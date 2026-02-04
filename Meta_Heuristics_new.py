from matplotlib import pyplot as plt
import numpy as np
import pulp
import copy

rng = np.random.default_rng(seed=2)


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


def inspect_x_solution(x, N, tol=1e-6):
    """
    Inspect x[i][j] values after MILP solve.

    Parameters
    ----------
    x : dict of dicts (PuLP variables)
        x[i][j] binary arc variables
    N : iterable
        Set/list of nodes
    tol : float
        Numerical tolerance
    """

    active_arcs = []
    out_degree = {i: 0 for i in N}
    in_degree = {i: 0 for i in N}

    for i in N:
        for j in N:
            if i != j and x[i][j].value() is not None:
                if x[i][j].value() > 1 - tol:
                    active_arcs.append((i, j))
                    out_degree[i] += 1
                    in_degree[j] += 1

    # print("\n=== ACTIVE ARCS (x[i][j] = 1) ===")
    # for i, j in active_arcs:
    #     print(f"{i} -> {j}")

    # print("\n=== NODE DEGREES ===")
    # print("Node | Out | In")
    # print("----------------")
    # for i in N:
    #     print(f"{i:>4} | {out_degree[i]:>3} | {in_degree[i]:>2}")

    # print("\n=== SYMMETRIC ARC CHECK (x[i][j] and x[j][i]) ===")
    # for i, j in active_arcs:
    #     if (j, i) in active_arcs:
    #         print(f"⚠️ 2-cycle detected: {i} <-> {j}")

    return active_arcs, out_degree, in_degree


def repair_route(assigned_containers, C_dict, Qk, T_ij, Handling_time=1 / 6):
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

    # print("\nRepairing route with MILP...")

    terminals = sorted({C_dict[cont]["Terminal"] for cont in assigned_containers})
    if 0 not in terminals:
        terminals = [0] + terminals

    N = terminals

    # Pickup / delivery quantities per terminal
    p = {j: 0 for j in N if j != 0}  # imports
    d = {j: 0 for j in N if j != 0}  # exports

    for cont in assigned_containers:
        j = C_dict[cont]["Terminal"]
        if j == 0:
            raise RuntimeError("Error: container assigned to dry port")
        if C_dict[cont]["In_or_Out"] == 1:
            p[j] += C_dict[cont]["Wc"]
        else:
            d[j] += C_dict[cont]["Wc"]

    # print("Import pickups per terminal:", p)
    # print("Export deliveries per terminal:", d)

    # Terminal time windows (start-of-service)
    O = {}
    D = {}
    service_time = {j: 0.0 for j in N}

    for j in N:
        related = [
            cont for cont in assigned_containers if C_dict[cont]["Terminal"] == j
        ]
        if j == 0:
            assert len(related) == 0, "Containers assigned to dry port in repair_route"
        if related:
            service_time[j] = sum(1 for cont in related) * Handling_time
            O[j] = max(C_dict[cont]["Oc"] for cont in related)
            D[j] = min(C_dict[cont]["Dc"] for cont in related)
        else:
            service_time[j] = 0
            O[j] = 0
            D[j] = 10**6

    R = max(
        [
            C_dict[cont]["Rc"]
            for cont in assigned_containers
            if C_dict[cont]["In_or_Out"] == 2
        ]
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
        prob += x[i][i] == 0  # no self-loops

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

    for j in N:
        if j != 0:
            for i in N:
                if i != j:
                    prob += t[j] >= t[i] + T_ij[i][j] - M * (1 - x[i][j])

            prob += t[j] >= O[j]
            prob += t[j] <= D[j]

    # Removed the upper bound as it forces tj = ti + Tij when xij = 1, which is not correct if we want to allow waiting
    # In any case, tj <= Dj already enforces an upper bound on tj, and tj >= O_j and tj >= ti + Tij when xij=1 enforces a lower bound
    # for i in N:
    #     for j in N:
    #         if i != j:
    #             if j != 0:
    #                 prob += t[j] <= t[i] + T_ij[i][j] + M * (1 - x[i][j])

    # --------------------------------------------------
    # STEP 6 — Solve
    # --------------------------------------------------
    # prob.writeLP("repair_debug.lp")

    # status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    status = prob.solve(
        pulp.PULP_CBC_CMD(
            msg=False,
            threads=1,  # 🔑 determinism
            timeLimit=None,
            options=["randomSeed 0"],  # 🔑 determinism
        )
    )

    if pulp.LpStatus[status] != "Optimal":
        # print("No feasible MILP route found")
        return None
        # raise RuntimeError("No feasible MILP route found")
    # else:
    #     print("Status:", pulp.LpStatus[prob.status])

    #     for v in prob.variables():
    #         if abs(v.varValue) > 1e-6:  # only nonzero vars
    #             print(f"{v.name} = {v.varValue}")

    # --------------------------------------------------
    # STEP 7 — Extract route
    # --------------------------------------------------

    for j in N:
        if j == 0:
            continue
        # print(
        #     f"Opening time at terminal: {O[j]}, Arrival time at terminal {j}: {pulp.value(t[j])}, Closing time at terminal: {D[j]}"
        # )

        assert O[j] <= pulp.value(t[j]) <= D[j], "Time window violated"

    _, degree_out, degree_in = inspect_x_solution(x, N)

    for i in N:
        assert degree_out[i] <= 1, "Invalid out-degree in repaired route"
        assert degree_in[i] <= 1, "Invalid in-degree in repaired route"
        assert degree_out[i] == degree_in[i], "Inconsistent degrees in repaired route"

    route = [0]
    current = 0

    while True:
        next_nodes = [j for j in N if j != current and pulp.value(x[current][j]) > 0.5]
        assert (
            len(next_nodes) <= 1
        ), "Invalid next nodes in repaired route, there are more than 1 next nodes"
        if not next_nodes:
            break
        nxt = next_nodes[0]
        route.append(nxt)
        current = nxt
        if nxt == 0:
            break

    # print("route", route)

    # print("Repaired route with MILP!!!")

    return route


class MetaHeuristic:
    def __init__(
        self,
        problem_instance,
        init_solution,
        get_route,
        get_timing,
        check_for_cap,
        calculate_objective,
    ):

        self.get_route = get_route
        self.get_timing = get_timing
        self.check_for_cap = check_for_cap
        self.calculate_objective = calculate_objective

        self.instance = problem_instance
        self.init_solution = init_solution

        self.H_t_dict = {1: self.instance.H_t_20, 2: self.instance.H_t_40}

        self.K = len(self.instance.K_list[:-1])  # exclude the truck

        self.full_choice_list = list(range(self.K)) + ["truck"]

        self.Barge_cap = self.init_solution.Barges
        self.H_b = self.init_solution.H_b

        # solution representation
        self.f_ck_greedy = self.init_solution.f_ck_init

        self.f_ck = copy.deepcopy(self.f_ck_greedy)

        # tabu structures (move_key -> tenure)
        self.T1 = {}
        self.T2 = {}
        self.T3 = {}

        self.critical = self._compute_critical_containers()

        self.non_critical = [
            cont for cont in range(self.instance.C) if cont not in self.critical
        ]

        self.route_dict = {}

        self.fill_initial_routes()

        self.route_load_dict = {k: {} for k in range(self.K)}

        self.move_accepts = 0
        self.swap_accepts = 0
        self.milp_calls = 0
        self.shake_count = 0
        self.milp_repairs = 0

        # parameters (tune these!)
        self.critical_move_prob = 0.6
        self.prob_operator_move = 0.8
        self.truck_move_prob = 0.6
        self.tenure_move_container = 15
        self.tenure_critical_container = 15
        self.tenure_barge_shake_ban = 80
        self.shake_threshold = 100

    def fill_initial_routes(self):
        for k in range(self.K):
            assigned = [
                cont for cont in range(self.instance.C) if self.f_ck[cont, k] == 1
            ]
            if not assigned:
                continue
            L_current = self._get_L_current_for_barge(barge_idx=k, fck=self.f_ck)

            route = self.get_route(L_current)

            self.route_dict[k] = route

    def _compute_critical_containers(self):
        """
        Compute the set of critical containers according to the paper logic:

        - Per sea terminal:
            * container with earliest opening time (min Oc)
            * container with latest closing time (max Dc)
        - Globally:
            * export container with latest release time (max Rc), if Rc > 0 exists
        """

        critical = set()
        C_dict = self.instance.C_dict

        # --- per-terminal critical containers (exclude dry port j = 0) ---
        terminals = {
            info["Terminal"] for info in C_dict.values() if info["Terminal"] != 0
        }

        for j in terminals:
            containers_at_j = [
                cont for cont, info in C_dict.items() if info["Terminal"] == j
            ]

            if not containers_at_j:
                continue

            # earliest opening
            c_earliest_O = min(containers_at_j, key=lambda cont: C_dict[cont]["Oc"])

            # latest closing
            c_latest_D = max(containers_at_j, key=lambda cont: C_dict[cont]["Dc"])

            critical.add(c_earliest_O)
            critical.add(c_latest_D)

        # --- export container with latest release ---
        export_containers = [
            cont for cont, info in C_dict.items() if info["In_or_Out"] == 2
        ]

        if export_containers:
            max_Rc = max(C_dict[cont]["Rc"] for cont in export_containers)
            if max_Rc > 0:  # ignore trivial all-zero case
                c_latest_R = max(export_containers, key=lambda cont: C_dict[cont]["Rc"])
                critical.add(c_latest_R)

        return critical

    def _edge_loads_along_route(
        self,
        route,
        L_current,
        barge_idx,
    ):

        edge_load_list = []

        # Start at depot: all exports are loaded
        load = sum(cont["Wc"] for cont in L_current.values() if cont["In_or_Out"] == 2)
        edge_load_list.append(load)

        # Visit terminals once in the given route
        for terminal in route:
            if terminal == 0:
                continue

            exports_unloaded = sum(
                cont["Wc"]
                for cont in L_current.values()
                if cont["Terminal"] == terminal and cont["In_or_Out"] == 2
            )
            imports_loaded = sum(
                cont["Wc"]
                for cont in L_current.values()
                if cont["Terminal"] == terminal and cont["In_or_Out"] == 1
            )

            load -= exports_unloaded
            load += imports_loaded
            edge_load_list.append(load)

        self.route_load_dict[barge_idx] = edge_load_list

    def _get_L_current_for_barge(self, barge_idx, fck):
        assigned = [
            cont for cont in range(self.instance.C) if fck[cont, barge_idx] == 1
        ]
        L_current = {cont: self.instance.C_dict[cont] for cont in assigned}
        return L_current

    def _fill_route_related_dictionaries(self, fck):
        for k in range(self.K):
            L_current = self._get_L_current_for_barge(barge_idx=k, fck=fck)

            route = self.route_dict[k]

            self._edge_loads_along_route(
                route,
                L_current,
                k,
            )  # this autofills self.route_load_dict

        assert (
            len(self.route_dict[k]) >= 2
        ), "route must at least start and end at depot after filling route_dict"

        assert (
            len(self.route_load_dict) == self.K
        ), "route_load_dict incomplete, missing barges or too many barges after filling route_load_dict"

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
        T1_expired = [m for m, t in self.T1.items() if t <= 1]
        for T1_m in T1_expired:
            del self.T1[T1_m]
        for T1_move in self.T1:
            self.T1[T1_move] -= 1

        T2_expired = [m for m, t in self.T2.items() if t <= 1]
        for T2_m in T2_expired:
            del self.T2[T2_m]
        for T2_move in self.T2:
            self.T2[T2_move] -= 1

        T3_expired = [b for b, t in self.T3.items() if t <= 1]
        for b in T3_expired:
            del self.T3[b]
        for b in self.T3:
            self.T3[b] -= 1

    def _released_container_tabu_reset(self, dumped_containers):
        """
        Remove tabu restrictions for containers released by a shake.
        """

        for tabu_list in (self.T1, self.T2):
            for move in list(tabu_list.keys()):
                if move[0] in dumped_containers:
                    del tabu_list[move]

    def _shake(self):
        best_k = None
        worst = 1.0

        if len(self.T3) == self.K:
            self.T3 = {}

        for k in range(self.K):
            if k in self.T3:
                continue
            assigned = [
                cont for cont in range(self.instance.C) if self.f_ck[cont, k] == 1
            ]
            if not assigned:
                continue
            Lcur = {cont: self.instance.C_dict[cont] for cont in assigned}
            route = self.route_dict[k]
            self._edge_loads_along_route(route, Lcur, k)
            util = max(self.route_load_dict[k]) / self.Barge_cap[k]
            if util < worst:
                worst, best_k = util, k
        if best_k is not None:
            dumped_containers = [
                cont for cont in range(self.instance.C) if self.f_ck[cont, best_k] == 1
            ]

            self.f_ck[:, best_k] = 0
            self.route_dict[best_k] = [0, 0]  # empty barge → trivial routes
            self.T3[best_k] = self.tenure_barge_shake_ban

            self._released_container_tabu_reset(dumped_containers)

            # greedy reassignment after shake
            for k in range(self.K):
                if k in self.T3:
                    continue
                self._randomized_greedy_reinsert(k)

    def _randomized_greedy_reinsert(self, barge_idx, max_trials=10):
        """
        Randomized greedy procedure:
        tries to insert trucked containers into a specific barge.
        Triggered after removing a critical container from that barge.
        """

        for _ in range(max_trials):
            old_route = copy.deepcopy(self.route_dict[barge_idx])
            trucked_containers = [
                cont for cont in range(self.instance.C) if not any(self.f_ck[cont, :])
            ]

            if not trucked_containers:
                print("No trucked containers to reinsert")
                return

            cont = rng.choice(trucked_containers)

            # tentative insertion
            self.f_ck[cont, barge_idx] = 1
            self.route_dict.pop(barge_idx)

            Lcur = self._get_L_current_for_barge(barge_idx=barge_idx, fck=self.f_ck)

            if len(Lcur) == 0:
                # empty barge → trivial route
                self.route_dict[barge_idx] = [0, 0]
            else:
                self.route_dict[barge_idx] = self.get_route(Lcur)

            self._fill_route_related_dictionaries(fck=self.f_ck)

            route = self.route_dict[barge_idx]
            assert (
                route is not None
            ), "Route missing for barge after fill method when trying to reinsert container from truck to barge"

            # capacity check
            if not self.check_for_cap(route, Lcur, barge_idx, barges=self.Barge_cap):
                self.f_ck[cont, barge_idx] = 0
                self.route_dict[barge_idx] = old_route

                continue

            result = self.get_timing(route, Lcur)

            if result is not None:
                # print("Successfully reinserted container from truck to barge")
                self.route_dict[barge_idx] = route
            else:
                # print("Failed to reinsert container from truck to barge")
                # revert
                self.f_ck[cont, barge_idx] = 0
                self.route_dict[barge_idx] = old_route

        return

    def operator_move(self):

        # 1) pick container container (unchanged)

        if rng.random() < self.critical_move_prob:
            crital_container_chosen = True
            trucked = [
                cont for cont in range(self.instance.C) if not any(self.f_ck[cont])
            ]
            crit_trucked = [cont for cont in trucked if cont in self.critical]
            if crit_trucked:
                container = rng.choice(crit_trucked)
            else:
                container = rng.choice(list(self.critical))
        else:
            crital_container_chosen = False
            container = rng.choice(self.non_critical)

        # 2) locate current assignment
        from_b = next((k for k in range(self.K) if self.f_ck[container, k]), "truck")

        choices = copy.deepcopy(self.full_choice_list)

        if from_b == "truck":
            choices.remove("truck")  # cannot move truck → truck
        elif not crital_container_chosen:
            choices.remove(from_b)  # cannot move to same barge
            choices.remove(
                "truck"
            )  # cannot move → truck as the container is non-critical
        elif crital_container_chosen:
            choices.remove(from_b)  # cannot move to same barge

        to_b = rng.choice(choices)

        if isinstance(to_b, str) and to_b != "truck":
            to_b = int(to_b)

        assert to_b != from_b, "from_b and to_b cannot be the same"

        move = (container, from_b, to_b)

        # print(f"Move (container, from_b, to_b): {move}")

        # 3) tabu check
        if (
            move in self.T1
            or move in self.T2
            or (from_b in self.T3)
            or (to_b in self.T3)
        ):
            return False

        assert len(self.route_dict) == self.K, "route_dict incomplete before move"
        assert all(
            len(self.route_dict[k]) >= 2 for k in range(self.K)
        ), "invalid route in route_dict before move"

        # Save old state
        old_row = self.f_ck[container, :].copy()
        old_Barge_cap = self.Barge_cap.copy()
        old_H_b = self.H_b.copy()
        old_route_from_b = self.route_dict[from_b] if from_b != "truck" else None
        old_route_to_b = self.route_dict[to_b] if to_b != "truck" else None

        # 4) tentative apply
        if from_b != "truck":
            self.f_ck[container, from_b] = 0
            self.route_dict.pop(from_b)

        if to_b != "truck":
            self.f_ck[container, to_b] = 1
            self.route_dict.pop(to_b)

        if to_b != "truck":
            Lcur_to = self._get_L_current_for_barge(barge_idx=to_b, fck=self.f_ck)

            if len(Lcur_to) == 0:
                # empty barge → trivial route
                self.route_dict[to_b] = [0, 0]
            else:
                self.route_dict[to_b] = self.get_route(Lcur_to)

        if from_b != "truck":
            Lcur_from = self._get_L_current_for_barge(barge_idx=from_b, fck=self.f_ck)

            if len(Lcur_from) == 0:
                # empty barge → trivial route
                self.route_dict[from_b] = [0, 0]
            else:
                self.route_dict[from_b] = self.get_route(Lcur_from)

        self._fill_route_related_dictionaries(fck=self.f_ck)

        # 5) CAPACITY CHECK
        required_capacity = [max(self.route_load_dict[k]) for k in range(self.K)]

        # dominance check
        req_sorted = sorted(required_capacity)
        cap_sorted = sorted(self.Barge_cap)

        if any(req > cap for req, cap in zip(req_sorted, cap_sorted)):
            # impossible no matter what
            self.f_ck[container, :] = old_row
            if from_b != "truck":
                self.route_dict[from_b] = old_route_from_b
            if to_b != "truck":
                self.route_dict[to_b] = old_route_to_b
            self.Barge_cap = old_Barge_cap
            self.H_b = old_H_b
            self.T1[move] = self.tenure_move_container
            return False

        # deterministic reassignment (upgrade or tighten)
        self.reassign_barges_by_requirements(required=required_capacity)

        # update old state after reassignment
        old_Barge_cap = self.Barge_cap.copy()
        old_H_b = self.H_b.copy()

        # 6) TIMING CHECK (only for affected barge if not truck)
        affected_barges = set()

        if from_b != "truck":
            affected_barges.add(from_b)

        if to_b != "truck":
            affected_barges.add(to_b)

        for barge_idx in affected_barges:
            Lcur = self._get_L_current_for_barge(barge_idx=barge_idx, fck=self.f_ck)

            route = self.route_dict[barge_idx]

            result = self.get_timing(route, Lcur)

            # 7) MILP repair if timing failed
            if result is None:
                assigned = [
                    i for i in range(self.instance.C) if self.f_ck[i, barge_idx]
                ]
                new_route = repair_route(
                    assigned,
                    self.instance.C_dict,
                    self.Barge_cap[barge_idx],
                    self.instance.T_ij_matrix,
                    self.instance.Handling_time,
                )
                self.milp_calls += 1

                if new_route is None:
                    # undo everything
                    self.f_ck[container, :] = old_row
                    if from_b != "truck":
                        self.route_dict[from_b] = old_route_from_b
                    if to_b != "truck":
                        self.route_dict[to_b] = old_route_to_b
                    self.Barge_cap = old_Barge_cap
                    self.H_b = old_H_b
                    self.T1[move] = self.tenure_move_container
                    return False
                if new_route is not None:
                    self.milp_repairs += 1
                    self.route_dict[barge_idx] = new_route
                    # hard capacity gate on the repaired route
                    Lcur = self._get_L_current_for_barge(
                        barge_idx=barge_idx, fck=self.f_ck
                    )
                    if not self.check_for_cap(
                        new_route, Lcur, barge_idx, barges=self.Barge_cap
                    ):
                        self.f_ck[container] = old_row
                        if from_b != "truck":
                            self.route_dict[from_b] = old_route_from_b
                        if to_b != "truck":
                            self.route_dict[to_b] = old_route_to_b
                        self.Barge_cap = old_Barge_cap
                        self.H_b = old_H_b
                        self.T1[move] = self.tenure_move_container
                        return False

        # 8) tabu bookkeeping
        if crital_container_chosen and from_b != "truck":
            # print("Critical container moved from barge → longer tabu on move")
            self.T2[move] = self.tenure_critical_container

            self._randomized_greedy_reinsert(from_b)

        return True

    def barge_fully_ok(self, barge_idx, move=None):
        assigned = [i for i in range(self.instance.C) if self.f_ck[i, barge_idx]]
        if not assigned:
            return True

        # one‐shift TW
        # ---- time-window check (identical logic to Greedy) ----
        Lcur = self._get_L_current_for_barge(barge_idx=barge_idx, fck=self.f_ck)

        route = self.route_dict[barge_idx]

        if not self.check_for_cap(route, Lcur, barge_idx, barges=self.Barge_cap):
            raise RuntimeError(
                "Capacity check failed in barge_ok method, for barge " + str(barge_idx),
                str(move),
            )

        result = self.get_timing(route, Lcur)

        if result is None:
            raise RuntimeError(
                "Timing check failed in barge_ok method, for barge " + str(barge_idx),
                str(move),
            )

    def operator_swap(self):
        c1, c2 = rng.choice(self.instance.C, size=2, replace=False)
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

        old_route_b1 = self.route_dict[b1]
        old_route_b2 = self.route_dict[b2]

        # invalidate routes affected by the swap
        self.route_dict.pop(b1)
        self.route_dict.pop(b2)

        Lcur_b1 = self._get_L_current_for_barge(barge_idx=b1, fck=self.f_ck)
        Lcur_b2 = self._get_L_current_for_barge(barge_idx=b2, fck=self.f_ck)

        if len(Lcur_b1) == 0:
            # empty barge → trivial route
            self.route_dict[b1] = [0, 0]
        else:
            self.route_dict[b1] = self.get_route(Lcur_b1)

        if len(Lcur_b2) == 0:
            # empty barge → trivial route
            self.route_dict[b2] = [0, 0]
        else:
            self.route_dict[b2] = self.get_route(Lcur_b2)

        self._fill_route_related_dictionaries(fck=self.f_ck)

        def barge_ok(k):
            assigned = [i for i in range(self.instance.C) if self.f_ck[i, k]]
            if not assigned:
                return True

            # one‐shift TW
            # ---- time-window check (identical logic to Greedy) ----
            Lcur = self._get_L_current_for_barge(barge_idx=k, fck=self.f_ck)

            route = self.route_dict[k]

            if not self.check_for_cap(route, Lcur, k, barges=self.Barge_cap):
                return False

            result = self.get_timing(route, Lcur)

            success = result is not None

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
                self.instance.Handling_time,
            )
            # print("New route from MILP repair:", new_route)
            self.milp_calls += 1

            if new_route is None:
                # irreparable swap → undo + tabu
                self.f_ck[c1] = old1
                self.f_ck[c2] = old2
                self.T1[move] = self.tenure_move_container
                return False
            else:
                self.milp_repairs += 1
                self.route_dict[b] = new_route
                # hard capacity gate on the repaired route
                Lcur = self._get_L_current_for_barge(barge_idx=b, fck=self.f_ck)
                if not self.check_for_cap(new_route, Lcur, b, barges=self.Barge_cap):
                    self.f_ck[c1] = old1
                    self.f_ck[c2] = old2
                    self.route_dict[b1] = old_route_b1
                    self.route_dict[b2] = old_route_b2
                    self.T1[move] = self.tenure_move_container
                    return False
        return True

    def evaluate(self):
        total_cost = 0
        total_stops = 0
        utils = []
        self.x_ijk = np.zeros((self.instance.N, self.instance.N, len(self.Barge_cap)))

        for k in range(self.K):
            assigned = np.where(self.f_ck[:, k] == 1)[0].tolist()
            if not assigned:
                continue
            Lcur = {cont: self.instance.C_dict[cont] for cont in assigned}
            route = self.route_dict[k]

            for i in range(len(route) - 1):
                if route[i] != route[i + 1]:
                    self.x_ijk[route[i]][route[i + 1]][k] = 1

            # util

            self._edge_loads_along_route(route, Lcur, k)
            util = max(self.route_load_dict[k]) / self.Barge_cap[k]
            assert util <= 1.0, "capacity violation detected in evaluation"
            utils.append(util)
            total_stops += len(route) - 2  # exclude depot visits

        # barge cost
        total_cost += self.calculate_objective()

        # truck
        unassigned = np.where(self.f_ck.sum(axis=1) == 0)[0]
        for cont in unassigned:
            total_cost += self.H_t_dict[self.instance.C_dict[cont]["Wc"]]

        return total_cost

    def local_search(self, max_iters=3000):
        print("\nStarting Meta-Heuristic Search...\n")
        self.best_cost = self.evaluate()
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
            if rng.random() < self.prob_operator_move:
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
                no_improve += 1
                continue

            cost = self.evaluate()

            if cost < self.best_cost:
                self.best_cost, best_f = cost, self.f_ck.copy()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.shake_threshold:
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

        final_route_dict = self.route_dict.copy()

        self.timing_window_plot(final_route_dict)

        print("\nMeta-Heuristic Search Complete, search move analysis:")
        print(f"Total move accepts: {self.move_accepts}")
        print(f"Total swap accepts: {self.swap_accepts}")
        print(f"Total MILP repair calls: {self.milp_calls}")
        print(f"Total shakes performed: {self.shake_count}")
        print(f"MILP repairs succeeded: {self.milp_repairs}\n")

        return self.best_cost, self.it_list, self.cost_list, self.best_cost_list

    def timing_window_plot(self, final_routes: dict):
        """
        Plot container time windows with actual barge arrival times.

        - One row per container
        - Grouped by barge
        - Green = import, Red = export
        - Square marker = export release time Rc
        - Cross marker = actual barge arrival time at container terminal
        """

        import matplotlib.pyplot as plt
        import math

        C_dict = self.instance.C_dict

        # --------------------------------------------------
        # 1) Collect containers per barge (final solution)
        # --------------------------------------------------
        barge_to_containers = {k: [] for k in range(self.K)}
        trucked = []

        for c in range(self.instance.C):
            assigned = False
            for k in range(self.K):
                if self.f_ck[c, k] == 1:
                    barge_to_containers[k].append(c)
                    assigned = True
                    break
            if not assigned:
                trucked.append(c)

        # --------------------------------------------------
        # 2) Compute global time horizon
        # --------------------------------------------------
        max_D = max(C_dict[c]["Dc"] for c in range(self.instance.C))
        Tmax = int(math.ceil(max_D / 50.0) * 50)

        # --------------------------------------------------
        # 3) Precompute arrival times per (barge, terminal)
        #    using waiting logic
        # --------------------------------------------------
        arrival_time = {}  # (k, terminal) -> time

        for k, route in final_routes.items():
            if not route or len(route) <= 1:
                continue

            containers = barge_to_containers[k]
            if not containers:
                continue

            Lcur = {c: C_dict[c] for c in containers}

            result = self.get_timing(route, Lcur)
            if result is None:
                print(
                    f"Warning: could not compute arrival times for barge {k} in final plot"
                )
                continue  # or mark route as infeasible
            D_term, O_term = result

            for node, arrival in zip(route, O_term):
                arrival_time[(k, node)] = arrival

        # --------------------------------------------------
        # 4) Build plot rows (barge, terminal, container)
        # --------------------------------------------------
        rows = []

        # --- barges in ascending order ---
        for k in sorted(barge_to_containers.keys()):
            containers = barge_to_containers[k]

            # sort by (terminal, container)
            containers_sorted = sorted(
                containers, key=lambda c: (C_dict[c]["Terminal"], c)
            )

            for c in containers_sorted:
                rows.append((k, C_dict[c]["Terminal"], c))

        # --- trucked containers last (optional) ---
        trucked_sorted = sorted(trucked, key=lambda c: (C_dict[c]["Terminal"], c))

        for c in trucked_sorted:
            rows.append(("Truck", C_dict[c]["Terminal"], c))

        # --------------------------------------------------
        # 5) Plot
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 0.3 * len(rows)))

        yticks = []
        ylabels = []

        for y, (k, terminal, c) in enumerate(rows):
            info = C_dict[c]
            Oc, Dc = info["Oc"], info["Dc"]

            color = "green" if info["In_or_Out"] == 1 else "red"

            # time window bar
            ax.barh(
                y,
                Dc - Oc,
                left=Oc,
                height=0.6,
                color=color,
                alpha=0.6,
                edgecolor="black",
            )

            # export release time
            if info["In_or_Out"] == 2 and info["Rc"] > 0:
                ax.scatter(info["Rc"], y, marker="s", color="black", zorder=3)

            # barge arrival time
            if k != "Truck":
                t_arr = arrival_time.get((k, terminal), None)
                if t_arr is not None:
                    ax.scatter(t_arr, y, marker="x", color="black", zorder=3)

            yticks.append(y)
            ylabels.append(f"B{k} | T{terminal} | C{c}")

        # --------------------------------------------------
        # 6) Final formatting
        # --------------------------------------------------
        ax.set_xlim(0, Tmax)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Time [hours]")
        ax.set_title("Container Time Windows and Barge Arrival Times")

        ax.grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()

    def build_final_allocation_report(self):
        report = {"summary": {}, "barges": [], "trucked_containers": {}}

        barge_assignments = {k: [] for k in range(self.K)}
        trucked_containers = []
        total_containers_on_barges = 0

        for cont in range(self.instance.C):
            assigned = False
            for k in range(self.K):
                if self.f_ck[cont, k] == 1:
                    barge_assignments[k].append(cont)
                    total_containers_on_barges += 1
                    assigned = True
                    break
            if not assigned:
                trucked_containers.append(cont)

        for k in range(self.K):
            containers = barge_assignments[k]
            if not containers:
                continue

            Lcur = {cont: self.instance.C_dict[cont] for cont in containers}
            route = self.route_dict[k]

            cap = self.Barge_cap[k]

            self._edge_loads_along_route(route, Lcur, k)
            peak = max(self.route_load_dict[k])

            imports = [
                cont
                for cont in containers
                if self.instance.C_dict[cont]["In_or_Out"] == 1
            ]
            exports = [
                cont
                for cont in containers
                if self.instance.C_dict[cont]["In_or_Out"] == 2
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

        report["trucked_containers"] = {
            "container_ids": trucked_containers,
            "num_20ft": sum(
                1
                for cont in trucked_containers
                if self.instance.C_dict[cont]["Wc"] == 1
            ),
            "num_40ft": sum(
                1
                for cont in trucked_containers
                if self.instance.C_dict[cont]["Wc"] == 2
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
