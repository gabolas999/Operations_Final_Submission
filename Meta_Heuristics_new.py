from matplotlib import pyplot as plt
import random
import numpy as np
import pulp
import copy


def sanitize_for_yaml(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_yaml(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_yaml(v) for v in obj)
    elif isinstance(obj, np.generic):  # catches np.float64, np.int64, etc.
        return obj.item()
    else:
        return obj


def repair_route(assigned_containers, C_dict, Qk, T_ij):
    """
    MILP routing repair for one barge (Section 4.2.2).

    Parameters
    ----------
    assigned_containers : list
        List of container IDs assigned to this barge.
    C_dict : dict
        Container info dictionary:
        - "Terminal"   : int
        - "In_or_Out"  : 1 (import) or 2 (export)
        - "Wc"         : 1 or 2
        - "Rc"         : release time
        - "Oc"         : opening time
        - "Dc"         : due time
    Qk : int
        Single Barge capacity (TEU).
    T_ij : dict[(i,j) -> float] or 2D list
        Travel times.
    handling_time : float
        Handling time per container (hours).

    Returns
    -------
    route : list[int]
        Terminal visit sequence including dry port.
    arrival_times : dict[int, float]
        Arrival time at each terminal.

    Raises
    ------
    RuntimeError
        If no feasible route is found.
    """

    # --------------------------------------------------
    # STEP 0 — Terminal set and demands
    # --------------------------------------------------

    terminals = sorted({C_dict[c]["Terminal"] for c in assigned_containers})
    if 0 not in terminals:
        terminals = [0] + terminals

    N = terminals

    # Pickup / delivery quantities per terminal
    p = {j: 0 for j in N if j != 0}  # imports
    d = {j: 0 for j in N if j != 0}  # exports

    for c in assigned_containers:
        j = C_dict[c]["Terminal"]
        if j == 0:
            continue
        if C_dict[c]["In_or_Out"] == 1:
            p[j] += C_dict[c]["Wc"]
        else:
            d[j] += C_dict[c]["Wc"]

    # print("Import pickups per terminal:", p)
    # print("Export deliveries per terminal:", d)

    # Terminal time windows
    O = {}
    D = {}

    for j in N:
        if j == 0:
            continue
        related = [c for c in assigned_containers if C_dict[c]["Terminal"] == j]
        if related:
            O[j] = max(C_dict[c]["Oc"] for c in related)
            D[j] = min(C_dict[c]["Dc"] for c in related)
        else:
            O[j] = 0
            D[j] = 10**6

    R = max(
        [C_dict[c]["Rc"] for c in assigned_containers if C_dict[c]["In_or_Out"] == 2]
        or [0]
    )

    # --------------------------------------------------
    # STEP 1 — Build MILP
    # --------------------------------------------------

    prob = pulp.LpProblem("BargeRoutingRepair", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", (N, N), 0, 1, cat="Binary")
    y = pulp.LpVariable.dicts("y", (N, N), 0)
    z = pulp.LpVariable.dicts("z", (N, N), 0)
    t = pulp.LpVariable.dicts("t", N, 0)

    # Objective (19): minimize travel time
    prob += pulp.lpSum(T_ij[i][j] * x[i][j] for i in N for j in N)

    # --------------------------------------------------
    # STEP 2 — Flow conservation (20)
    # --------------------------------------------------
    for i in N:
        prob += (
            pulp.lpSum(x[i][j] for j in N if j != i)
            - pulp.lpSum(x[j][i] for j in N if j != i)
            == 0
        )

    # One departure from dry port (21)
    prob += pulp.lpSum(x[0][j] for j in N if j != 0) <= 1

    # --------------------------------------------------
    # STEP 3 — Pickup / delivery flows (22–23)
    # --------------------------------------------------

    for j in N:
        if j != 0:
            prob += (
                pulp.lpSum(y[i][j] for i in N if i != j)
                - pulp.lpSum(y[j][i] for i in N if i != j)
                == p[j]
            )
            prob += (
                pulp.lpSum(z[j][i] for i in N if i != j)
                - pulp.lpSum(z[i][j] for i in N if i != j)
                == d[j]
            )

    # --------------------------------------------------
    # STEP 4 — Capacity (24)
    # --------------------------------------------------

    for i in N:
        for j in N:
            prob += y[i][j] + z[i][j] <= Qk * x[i][j]

    # --------------------------------------------------
    # STEP 5 — Timing (25–29)
    # --------------------------------------------------

    prob += t[0] >= R

    M = 10**6

    for i in N:
        for j in N:
            prob += t[j] >= t[i] + T_ij[i][j] - M * (1 - x[i][j])

    for i in N:
        for j in N:
            prob += t[j] <= t[i] + T_ij[i][j] + M * (1 - x[i][j])

    for j in N:
        if j != 0:
            prob += t[j] >= O[j]
            prob += t[j] <= D[j]

    # --------------------------------------------------
    # STEP 6 — Solve
    # --------------------------------------------------

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[status] != "Optimal":
        return None
        # raise RuntimeError("No feasible MILP route found")

    # --------------------------------------------------
    # STEP 7 — Extract route
    # --------------------------------------------------

    route = [0]
    current = 0

    while True:
        next_nodes = [
            j for j in N if j not in route and pulp.value(x[current][j]) > 0.5
        ]
        if not next_nodes:
            break
        nxt = next_nodes[0]
        route.append(nxt)
        current = nxt
        if current == 0:
            break

    # arrival_times = {j: pulp.value(t[j]) for j in route}

    return route


class MetaHeuristic:
    def __init__(
        self,
        problem_instance,
        init_solution,
        get_route,
        get_timing,
        check_for_cap,
        delay_window,
        calculate_objective,
    ):

        self.get_route = get_route
        self.get_timing = get_timing
        self.check_for_cap = check_for_cap
        self.delay_window = delay_window
        self.calculate_objective = calculate_objective

        self.instance = problem_instance
        self.init_solution = init_solution

        # 1) compute slacks and pick top‐10% as critical
        slacks = {
            c: self.instance.C_dict[c]["Dc"] - self.instance.C_dict[c]["Oc"]
            for c in self.init_solution.C_ordered
        }
        ncrit = max(1, int(0.1 * len(self.init_solution.C_ordered)))
        crit_sorted = sorted(slacks, key=slacks.get)
        self.critical = set(crit_sorted[:ncrit])

        # parameters
        self.truck_move_prob = 0.6
        self.critical_move_prob = 0.6
        self.ten_crit = 30

        self.H_t_dict = {1: self.instance.H_t_20, 2: self.instance.H_t_40}

        self.K = len(self.instance.K_list[:-1])  # exclude the truck

        self.Barge_cap = self.init_solution.Barges
        self.H_b = self.init_solution.H_b

        # solution representation
        self.f_ck_greedy = self.init_solution.f_ck_init

        self.f_ck = copy.deepcopy(self.f_ck_greedy)

        # tabu structures (move_key -> tenure)
        self.T1 = {}
        self.T2 = {}
        self.T3 = {}

        self.route_dict = {}

        self.route_load_dict = {k: {} for k in range(self.K)}

        self.move_accepts = 0
        self.swap_accepts = 0
        self.milp_calls = 0
        self.shake_count = 0
        self.milp_repairs = 0

        # parameters (tune these!)
        self.ten_move = 20
        self.ten_crit = 20
        self.ten_barban = 10
        self.shake_thr = 60

    def _edge_loads_along_route(
        self,
        route,
        L_current,
        barge_idx,
    ):

        edge_load_list = []

        # Start at depot: all exports are loaded
        load = sum(c["Wc"] for c in L_current.values() if c["In_or_Out"] == 2)
        edge_load_list.append(load)

        # Visit terminals once in the given route
        for terminal in route:
            if terminal == 0:
                continue

            exports_unloaded = sum(
                c["Wc"]
                for c in L_current.values()
                if c["Terminal"] == terminal and c["In_or_Out"] == 2
            )
            imports_loaded = sum(
                c["Wc"]
                for c in L_current.values()
                if c["Terminal"] == terminal and c["In_or_Out"] == 1
            )

            load -= exports_unloaded
            load += imports_loaded
            edge_load_list.append(load)

        self.route_load_dict[barge_idx] = edge_load_list

    def _get_L_current_for_barge(self, barge_idx, fck=None):
        assigned = [c for c in range(self.instance.C) if fck[c, barge_idx] == 1]
        L_current = {c: self.instance.C_dict[c] for c in assigned}
        return L_current

    def _fill_dicts(self, fck):
        for k in range(self.K):
            L_current = self._get_L_current_for_barge(barge_idx=k, fck=fck)

            route = self.route_dict.get(k, self.get_route(L_current))

            self.route_dict[k] = route  # fills route dict

            self._edge_loads_along_route(
                route,
                L_current,
                k,
            )  # this autofills self.route_load_dict

            assert (
                len(self.route_dict[k]) >= 2
            ), "route must at least start and end at depot"

        assert (
            len(self.route_load_dict) == self.K
        ), "route_load_dict incomplete, missing barges or too many barges"

    def reassign_barges_by_requirements(self, required):
        """
        Deterministically assign barges to routes by sorted requirements.
        Assumes global dominance check has already passed.
        """

        req_sorted_idx = sorted(range(self.K), key=lambda k: required[k])
        cap_cost_sorted = sorted(zip(self.Barge_cap, self.H_b), key=lambda x: x[0])

        new_Barge_cap = [None] * self.K
        new_H_b = [None] * self.K

        for r_idx, (cap, cost) in zip(req_sorted_idx, cap_cost_sorted):
            new_Barge_cap[r_idx] = cap
            new_H_b[r_idx] = cost

        self.Barge_cap = new_Barge_cap
        self.H_b = new_H_b

    def _age_tabu(self):
        # decrement and purge expired tenures from T1, T2, T3
        for T in (self.T1, self.T2):
            expired = [m for m, t in T.items() if t <= 1]
            for m in expired:
                del T[m]
            for m in T:
                T[m] -= 1
        expired = [b for b, t in self.T3.items() if t <= 1]
        for b in expired:
            del self.T3[b]
        for b in self.T3:
            self.T3[b] -= 1

    def _shake(self):
        best_k = None
        worst = 1.0
        for k in range(self.K):
            if k in self.T3:
                continue
            assigned = [c for c in range(self.instance.C) if self.f_ck[c, k] == 1]
            if not assigned:
                continue
            Lcur = {c: self.instance.C_dict[c] for c in assigned}
            route = self.get_route(Lcur)
            loads = []
            load = sum(c["Wc"] for c in Lcur.values() if c["In_or_Out"] == 2)
            loads.append(load)
            for node in route[1:]:
                for cont in Lcur.values():
                    if cont["Terminal"] == node:
                        load += cont["Wc"] if cont["In_or_Out"] == 1 else -cont["Wc"]
                loads.append(load)
            util = sum(loads) / (len(loads) * self.Barge_cap[k])
            if util < worst:
                worst, best_k = util, k
        if best_k is not None:
            self.f_ck[:, best_k] = 0
            self.T3[best_k] = self.ten_barban

    def operator_move(self):

        # 1) pick container c (unchanged)
        if random.random() < self.critical_move_prob:
            trucked = [c for c in range(self.instance.C) if not any(self.f_ck[c])]
            crit_trucked = [c for c in trucked if c in self.critical]
            if crit_trucked:
                c = random.choice(crit_trucked)
            elif trucked:
                c = random.choice(trucked)
            else:
                c = random.randrange(self.instance.C)
        else:
            c = random.randrange(self.instance.C)

        # 2) locate current assignment
        from_b = next((k for k in range(self.K) if self.f_ck[c, k]), None)
        if from_b is None:
            from_b = "truck"
        choices = list(range(self.K)) + ["truck"]
        to_b = random.choice(choices)

        if to_b == from_b:
            return False

        move = (c, from_b, to_b)

        # 3) tabu check
        if (
            move in self.T1
            or move in self.T2
            or (from_b in self.T3)
            or (to_b in self.T3)
        ):
            return False

        # Save old state
        old_row = self.f_ck[c, :].copy()
        old_route_dict = self.route_dict.copy()
        old_Barge_cap = self.Barge_cap.copy()
        old_H_b = self.H_b.copy()

        # 4) tentative apply
        if from_b != "truck":
            self.f_ck[c, from_b] = 0
            self.route_dict.pop(from_b, None)

        if to_b != "truck":
            self.f_ck[c, to_b] = 1
            self.route_dict.pop(to_b, None)

        self._fill_dicts(fck=self.f_ck)

        # 5) QUICK CAPACITY CHECK
        required_capacity = [max(self.route_load_dict[k]) for k in range(self.K)]

        # dominance check
        req_sorted = sorted(required_capacity)
        cap_sorted = sorted(self.Barge_cap)

        if any(r > c for r, c in zip(req_sorted, cap_sorted)):
            # impossible no matter what
            self.f_ck[c, :] = old_row
            self.route_dict = old_route_dict
            self.Barge_cap = old_Barge_cap
            self.H_b = old_H_b
            self.T1[move] = self.ten_move
            return False

        # deterministic reassignment (upgrade or tighten)
        self.reassign_barges_by_requirements(required=required_capacity)

        # 6) TIMING CHECK (only for affected barge if not truck)
        if to_b != "truck":
            Lcur = self._get_L_current_for_barge(barge_idx=to_b, fck=self.f_ck)

            route = self.route_dict.get(to_b, self.get_route(Lcur))

            delay = 0.0
            feasible = False

            delay_per_terminal = {term: 0 for term in route}

            for attempt in range(2):  # attempt = 0 (no shift), attempt = 1 (shift)
                D_term, O_term = self.get_timing(route, Lcur, delay)
                arrival_by_terminal = dict(zip(route, O_term))

                for terminal in route:
                    Oj = max(
                        info["Oc"]
                        for c, info in Lcur.items()
                        if info["Terminal"] == terminal
                    )
                    Dj = min(
                        info["Dc"]
                        for c, info in Lcur.items()
                        if info["Terminal"] == terminal
                    )

                    arrival = arrival_by_terminal[terminal]

                    if arrival >= Oj and arrival <= Dj:
                        continue
                    elif arrival < Oj:
                        delay_per_terminal[terminal] = Oj - arrival
                    elif arrival > Dj:
                        late = True
                        break

                if late:
                    feasible = False
                    break

                if all(d == 0 for d in delay_per_terminal.values()):
                    feasible = True
                    break

                if attempt == 0:
                    delay += max(delay_per_terminal.values())
                else:
                    break

            # 7) MILP repair if timing failed
            if not feasible:
                assigned = [i for i in range(self.instance.C) if self.f_ck[i, to_b]]
                new_route = repair_route(
                    assigned,
                    self.instance.C_dict,
                    self.Barge_cap[to_b],
                    self.instance.T_ij_matrix,
                )
                self.milp_calls += 1

                if new_route is None:
                    # undo everything
                    self.f_ck[c, :] = old_row
                    self.route_dict = old_route_dict
                    self.Barge_cap = old_Barge_cap
                    self.H_b = old_H_b
                    self.T1[move] = self.ten_move
                    return False

                self.milp_repairs += 1
                self.route_dict[to_b] = new_route

        # 8) tabu bookkeeping
        if to_b == "truck" and c in self.critical:
            self.T2[move] = self.ten_crit

        return True

    def operator_swap(self):
        c1, c2 = random.sample(range(self.instance.C), 2)
        bs1 = [k for k in range(self.K) if self.f_ck[c1, k]]
        bs2 = [k for k in range(self.K) if self.f_ck[c2, k]]
        if not bs1 or not bs2 or bs1[0] == bs2[0]:
            return False
        b1, b2 = bs1[0], bs2[0]

        move = ((c1, b1, b2), (c2, b2, b1))
        if move in self.T1 or b1 in self.T3 or b2 in self.T3:
            return False

        # tentatively swap
        old1 = self.f_ck[c1].copy()
        old2 = self.f_ck[c2].copy()
        self.f_ck[c1, b1] = 0
        self.f_ck[c1, b2] = 1
        self.f_ck[c2, b2] = 0
        self.f_ck[c2, b1] = 1

        # invalidate routes affected by the swap
        self.route_dict.pop(b1, None)
        self.route_dict.pop(b2, None)

        def barge_ok(k):
            assigned = [i for i in range(self.instance.C) if self.f_ck[i, k]]
            if not assigned:
                return True
            Lcur = {i: self.instance.C_dict[i] for i in assigned}
            route = self.route_dict.get(k, self.get_route(Lcur))
            if not self.check_for_cap(route, Lcur, k, barges=self.Barge_cap):
                return False
            # one‐shift TW
            # ---- time-window check (identical logic to Greedy) ----
            delay = 0.0

            success = False

            for attempt in range(2):  # at most one shift
                D_term, O_term = self.get_timing(route, Lcur, delay)

                early_arrival_violations = []
                late = False

                for cont in Lcur.values():
                    t = cont["Terminal"]
                    arrival = O_term[route.index(t)]

                    if arrival < cont["Oc"]:
                        early_arrival_violations.append(cont)
                    elif arrival > cont["Dc"]:
                        late = True
                        break

                if late:
                    success = False
                    break

                if not early_arrival_violations:
                    success = True
                    break

                # apply the single allowed shift
                if attempt == 0:
                    delay_needed = [
                        self.delay_window(
                            container=v,
                            O_terminal=O_term,
                            route=route,
                            terminal=v["Terminal"],
                        )
                        for v in early_arrival_violations
                    ]
                    delay += max(delay_needed)

            return success

        ok1 = barge_ok(b1)
        ok2 = barge_ok(b2)
        if ok1 and ok2:
            return True

        # quick check failed on at least one barge: call repair on each
        for b in (b1, b2):
            assigned = [i for i in range(self.instance.C) if self.f_ck[i, b]]

            new_route = repair_route(
                assigned,
                self.instance.C_dict,
                self.Barge_cap[b],
                self.instance.T_ij_matrix,
            )
            # print("New route from MILP repair:", new_route)
            self.milp_calls += 1

            if new_route is None:
                # irreparable swap → undo + tabu
                self.f_ck[c1] = old1
                self.f_ck[c2] = old2
                self.T1[move] = self.ten_move
                return False
            else:
                self.milp_repairs += 1
                L_cur_temp = {i: self.instance.C_dict[i] for i in assigned}

                old_route = self.get_route(L_cur_temp)
                old_cost = sum(
                    self.instance.T_ij_matrix[old_route[i]][old_route[i + 1]]
                    for i in range(len(old_route) - 1)
                )

                new_cost = sum(
                    self.instance.T_ij_matrix[new_route[i]][new_route[i + 1]]
                    for i in range(len(new_route) - 1)
                )

                if new_cost >= old_cost:
                    print("MILP routing not better:", old_cost, "→", new_cost)
                self.route_dict[b] = new_route

        # both repairs succeeded
        return True

    def evaluate(self):
        total_cost = 0
        total_stops = 0
        utils = []
        self.x_ijk = np.zeros((len(self.Barge_cap), self.instance.N, self.instance.N))

        for k in range(self.K):
            assigned = np.where(self.f_ck[:, k] == 1)[0].tolist()
            if not assigned:
                continue
            Lcur = {c: self.instance.C_dict[c] for c in assigned}
            route = self.route_dict.get(k, self.get_route(Lcur))

            self.route_dict[k] = route

            for i in range(len(route) - 1):
                if route[i] != route[i + 1]:
                    self.x_ijk[k][route[i]][route[i + 1]] = 1

            # util
            load = sum(c["Wc"] for c in Lcur.values() if c["In_or_Out"] == 2)
            loads = [load]
            for node in route[1:]:
                for cont in Lcur.values():
                    if cont["Terminal"] == node:
                        load += cont["Wc"] if cont["In_or_Out"] == 1 else -cont["Wc"]
                loads.append(load)
            utils.append(sum(loads) / (len(loads) * self.Barge_cap[k]))
        # barge cost
        total_cost += self.calculate_objective()

        # truck
        unassigned = np.where(self.f_ck.sum(axis=1) == 0)[0]
        for c in unassigned:
            total_cost += self.H_t_dict[self.instance.C_dict[c]["Wc"]]

        return total_cost, total_stops, (sum(utils) / len(utils) if utils else 0)

    def local_search(self, max_iters=3000):
        print("\nStarting Meta-Heuristic Search...\n")
        self.best_cost, _, _ = self.evaluate()
        best_f = self.f_ck.copy()
        no_improve = 0

        self.it_list = []
        self.cost_list = []
        self.best_cost_list = []

        # # Set up interactive plotting
        # plt.ion()
        # fig, ax = plt.subplots()
        # (line1,) = ax.plot([], [], label="Current Cost")
        # (line2,) = ax.plot([], [], label="Best Cost", linestyle="--")
        # ax.set_xlabel("Iteration")
        # ax.set_ylabel("Cost")
        # ax.set_title("Meta-Heuristic Cost Over Iterations")
        # ax.legend()
        # ax.grid(True)

        for it in range(max_iters):
            if it % 100 == 0:
                print(f"Iteration {it}, Percent Complete: {100*it/max_iters:.1f}%")
                print(f"  Current best cost: {self.best_cost}")
            if random.random() < 0.8:
                moved = self.operator_move()
                if moved:
                    self.move_accepts += 1
            else:
                moved = self.operator_swap()
                if moved:
                    self.swap_accepts += 1

            # *always* age your tabu after every move‐attempt
            self._age_tabu()

            if not moved:
                continue

            cost, _, _ = self.evaluate()

            if cost < self.best_cost:
                self.best_cost, best_f = cost, self.f_ck.copy()
                no_improve = 0
            else:
                # self.f_ck = best_f.copy()
                no_improve += 1

            if no_improve >= self.shake_thr:
                self._shake()
                self.shake_count += 1
                no_improve = 0

            self.it_list.append(it)
            self.cost_list.append(cost)
            self.best_cost_list.append(self.best_cost)

        #     # Update plot every iteration
        #     line1.set_data(self.it_list, self.cost_list)
        #     line2.set_data(self.it_list, self.best_cost_list)
        #     ax.relim()
        #     ax.autoscale_view()
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()

        # plt.ioff()
        self.f_ck = copy.deepcopy(best_f)

        print("\nMeta-Heuristic Search Complete, search move analysis:")
        print(f"Total move accepts: {self.move_accepts}")
        print(f"Total swap accepts: {self.swap_accepts}")
        print(f"Total MILP repair calls: {self.milp_calls}")
        print(f"Total shakes performed: {self.shake_count}")
        print(f"MILP repairs succeeded: {self.milp_repairs}\n")

        return self.best_cost, self.it_list, self.cost_list, self.best_cost_list

    def build_final_allocation_report(self):
        report = {"summary": {}, "barges": [], "trucked_containers": {}}

        barge_assignments = {k: [] for k in range(self.K)}
        trucked_containers = []

        for c in range(self.instance.C):
            assigned = False
            for k in range(self.K):
                if self.f_ck[c, k] == 1:
                    barge_assignments[k].append(c)
                    assigned = True
                    break
            if not assigned:
                trucked_containers.append(c)

        total_containers_on_barges = 0

        for k in range(self.K):
            containers = barge_assignments[k]
            if not containers:
                continue

            Lcur = {c: self.instance.C_dict[c] for c in containers}
            route = self.route_dict.get(k, self.get_route(Lcur))

            cap = self.Barge_cap[k]
            load = sum(info["Wc"] for info in Lcur.values() if info["In_or_Out"] == 2)
            peak = load

            for node in route[1:]:
                exports_unloaded = sum(
                    info["Wc"]
                    for info in Lcur.values()
                    if info["Terminal"] == node and info["In_or_Out"] == 2
                )
                imports_loaded = sum(
                    info["Wc"]
                    for info in Lcur.values()
                    if info["Terminal"] == node and info["In_or_Out"] == 1
                )
                load = load - exports_unloaded + imports_loaded
                peak = max(peak, load)

            imports = [
                c for c in containers if self.instance.C_dict[c]["In_or_Out"] == 1
            ]
            exports = [
                c for c in containers if self.instance.C_dict[c]["In_or_Out"] == 2
            ]

            report["barges"].append(
                {
                    "barge_id": k + 1,
                    "capacity": cap,
                    "fixed_cost": self.init_solution.H_b[k],
                    "num_containers": len(containers),
                    "peak_load": peak,
                    "utilization_percent": round(100 * peak / cap, 1),
                    "imports": imports,
                    "exports": exports,
                    "container_ids": containers,
                }
            )

            total_containers_on_barges += len(containers)

        report["trucked_containers"] = {
            "container_ids": trucked_containers,
            "num_20ft": sum(
                1 for c in trucked_containers if self.instance.C_dict[c]["Wc"] == 1
            ),
            "num_40ft": sum(
                1 for c in trucked_containers if self.instance.C_dict[c]["Wc"] == 2
            ),
        }

        report["summary"] = {
            "total_containers": self.instance.C,
            "containers_on_barges": total_containers_on_barges,
            "containers_trucked": len(trucked_containers),
            "barges_used": len(report["barges"]),
            "final_cost": self.best_cost,
        }

        return report

    def display_final_allocations(
        self, scenario_name, yaml_dir="./Storage/theo_results"
    ):
        import yaml

        yaml_path = f"{yaml_dir}/{scenario_name}_final_allocations.yaml"
        report = self.build_final_allocation_report()

        # Pretty print (terminal)
        print("\n" + "=" * 80)
        print("FINAL CONTAINER-BARGE ALLOCATIONS")
        print("=" * 80)

        for b in report["barges"]:
            print(
                f"\nBARGE {b['barge_id']} (Capacity: {b['capacity']} TEU, Fixed cost: €{b['fixed_cost']}):"
            )
            print(f"  Assigned containers: {b['num_containers']}")
            print(
                f"  Peak onboard load: {b['peak_load']}/{b['capacity']} "
                f"({b['utilization_percent']}% utilization)"
            )
            print(f"  Import containers: {len(b['imports'])}")
            print(f"  Export containers: {len(b['exports'])}")

        print("\nSUMMARY:")
        for k, v in report["summary"].items():
            print(f"  {k.replace('_',' ').title()}: {v}")

        # Save YAML
        report = sanitize_for_yaml(report)
        with open(yaml_path, "w") as f:
            yaml.safe_dump(report, f, sort_keys=False)

        return report, yaml_path


if __name__ == "__main__":
    # usage
    from MILP import MILP_Algo
    from Greedy_Algo import GreedyOptimizer

    milp_instance = MILP_Algo(reduced=False)
    greedy = GreedyOptimizer(problem_instance=milp_instance)
    init_solution = greedy.solve_greedy()

    mh = MetaHeuristic(problem_instance=milp_instance, init_solution=init_solution)
    # mh.initial_solution()
    print("Greedy cost:", init_solution.total_cost, "\n")
    mh.local_search()
    print("Meta-heuristic cost:", mh.best_cost, "\n")

    # Display final allocations
    mh.display_final_allocations()
