from matplotlib import pyplot as plt
import numpy as np
import pulp
import copy

from helpers import (
    _edge_loads_along_route,
    _get_L_current_for_barge,
    timing_window_plot,
)

rng = np.random.default_rng(seed=2)


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
                    prob += t[j] >= t[i] + service_time[i] + T_ij[i][j] - M * (
                        1 - x[i][j]
                    )

            prob += t[j] >= O[j]
            prob += t[j] + service_time[j] <= D[j]

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
        return None, None
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

        if not (O[j] <= pulp.value(t[j]) <= D[j]):
            print("Timing violation at terminal", j)
            print(
                f"Opening time at terminal: {O[j]}, Arrival time at terminal {j}: {pulp.value(t[j])}, Closing time at terminal: {D[j]}"
            )
            return None, None

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

    timing = dict(zip(route, [pulp.value(t[j]) for j in route]))

    return route, timing


class MetaHeuristic:
    def __init__(
        self,
        scenario_name,
        problem_instance,
        init_solution,
        get_route,
        get_timing,
        calculate_objective,
    ):

        self.cost_list = []
        self.scenario_name = scenario_name
        self.get_route = get_route
        self.get_timing = get_timing
        self.calculate_objective = calculate_objective

        self.H_t_dict = {1: problem_instance.H_t_20, 2: problem_instance.H_t_40}

        self.K = len(problem_instance.K_list[:-1])  # exclude the truck

        self.C = problem_instance.C
        self.C_dict = problem_instance.C_dict
        self.T_ij = problem_instance.T_ij_matrix
        self.Handling_time = problem_instance.Handling_time
        self.N = problem_instance.N

        self.full_choice_list = list(range(self.K)) + ["truck"]

        self.Barge_cap = init_solution.Barges
        self.H_b = init_solution.H_b

        # solution representation
        self.f_ck = copy.deepcopy(init_solution.f_ck)

        # tabu structures (move_key -> tenure)
        self.T1 = {}
        self.T2 = {}
        self.T3 = {}

        self.critical = self._compute_critical_containers()

        self.non_critical = [
            cont for cont in range(self.C) if cont not in self.critical
        ]

        self.route_dict = {}

        self.fill_initial_routes()

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
            assigned = [cont for cont in range(self.C) if self.f_ck[cont, k] == 1]
            if not assigned:
                continue
            L_current = _get_L_current_for_barge(
                barge_idx=k,
                f_ck=self.f_ck,
                C=self.C,
                C_dict=self.C_dict,
            )

            route = self.get_route(L_current)

            timing = self.get_timing(route, L_current)

            entry = self.route_dict.setdefault(k, {})

            entry.setdefault("route", route)
            entry.setdefault("repaired", False)
            entry.setdefault("timing", timing)

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
        C_dict = self.C_dict

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

    def reassign_barges_by_cost_requirements(self, req_cap_per_route):
        """
        Cost-first reassignment of barges to routes, with capacity feasibility check.

        - Routes with higher utilization get cheaper barges
        - Capacity feasibility is enforced locally
        - Global feasibility assumed but not blindly trusted
        """

        # Routes sorted by descending importance
        routes_sorted = sorted(
            range(self.K), key=lambda k: req_cap_per_route[k], reverse=True
        )

        # Barges sorted by ascending cost
        barges = sorted(
            [
                (cap, cost, i)
                for i, (cap, cost) in enumerate(zip(self.Barge_cap, self.H_b))
            ],
            key=lambda x: x[1],
        )

        new_Barge_cap = [None] * self.K
        new_H_b = [None] * self.K
        assigned_barges = set()

        for r in routes_sorted:
            req = req_cap_per_route[r]

            # find cheapest feasible unused barge
            for cap, cost, b_idx in barges:
                if b_idx in assigned_barges:
                    continue
                if cap >= req:
                    new_Barge_cap[r] = cap
                    new_H_b[r] = cost
                    assigned_barges.add(b_idx)
                    break
            else:
                raise RuntimeError(
                    f"No feasible barge found for route {r} with requirement {req}"
                )

        return new_Barge_cap, new_H_b

    def reassign_barges_by_capacity(self, req_cap_per_route):
        """
        Reassign barges to routes by global capacity matching.

        - Route order is preserved
        - self.Barge_cap and self.H_b are reordered in-place
        - Returns None if no feasible assignment exists
        """

        K = self.K

        # Routes sorted by increasing required capacity
        routes_sorted = sorted(range(K), key=lambda k: req_cap_per_route[k])

        # Barges sorted by increasing capacity
        barges_sorted = sorted(zip(self.Barge_cap, self.H_b), key=lambda x: x[0])

        new_Barge_cap = [None] * K
        new_H_b = [None] * K

        for r, (cap, cost) in zip(routes_sorted, barges_sorted):
            if cap < req_cap_per_route[r]:
                return None, None  # globally infeasible

            new_Barge_cap[r] = cap
            new_H_b[r] = cost

        return new_Barge_cap, new_H_b

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
            assigned = [cont for cont in range(self.C) if self.f_ck[cont, k] == 1]
            if not assigned:
                continue
            Lcur_k = {cont: self.C_dict[cont] for cont in assigned}
            route_k = self.route_dict[k]["route"]
            edge_loads = _edge_loads_along_route(
                route=route_k,
                L_current=Lcur_k,
            )
            util = max(edge_loads) / self.Barge_cap[k]
            if util < worst:
                worst, best_k = util, k
        if best_k is not None:
            dumped_containers = [
                cont for cont in range(self.C) if self.f_ck[cont, best_k] == 1
            ]

            self.f_ck[:, best_k] = 0
            self.route_dict[best_k]["route"] = [0, 0]  # empty barge → trivial routes
            self.route_dict[best_k]["repaired"] = False
            self.route_dict[best_k]["timing"] = {0: 0, 0: 0}
            self.T3[best_k] = self.tenure_barge_shake_ban

            self._released_container_tabu_reset(dumped_containers)

            cost = self.evaluate()

            self.cost_list.append(cost)

            if cost < self.best_cost:
                self.best_cost = cost
                self.best_fck = copy.deepcopy(self.f_ck)
                self.best_route_dict = copy.deepcopy(self.route_dict)
                self.best_Barge_cap = copy.deepcopy(self.Barge_cap)
                self.best_H_b = copy.deepcopy(self.H_b)

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
            old_route = copy.deepcopy(self.route_dict[barge_idx]["route"])
            old_timing = copy.deepcopy(self.route_dict[barge_idx]["timing"])
            old_repaired = self.route_dict[barge_idx]["repaired"]

            trucked_containers = [
                cont for cont in range(self.C) if not any(self.f_ck[cont, :])
            ]

            if not trucked_containers:
                print("No trucked containers to reinsert")
                return

            cont = rng.choice(trucked_containers)

            # tentative insertion
            self.f_ck[cont, barge_idx] = 1
            self.route_dict.pop(barge_idx)

            Lcur = _get_L_current_for_barge(
                barge_idx=barge_idx,
                f_ck=self.f_ck,
                C=self.C,
                C_dict=self.C_dict,
            )

            if len(Lcur) == 0:
                # empty barge → trivial route
                entry = self.route_dict.setdefault(barge_idx, {})
                entry.setdefault("route", [0, 0])
                entry.setdefault("repaired", False)
                entry.setdefault("timing", {0: 0, 0: 0})
            else:
                created_route = self.get_route(Lcur)
                timing = self.get_timing(created_route, Lcur)
                entry = self.route_dict.setdefault(barge_idx, {})
                entry.setdefault("route", created_route)
                entry.setdefault("repaired", False)
                entry.setdefault("timing", timing)

            route = self.route_dict[barge_idx]["route"]
            assert (
                route is not None
            ), "Route missing for barge after fill method when trying to reinsert container from truck to barge"

            # capacity check
            edge_list_barge_idx = _edge_loads_along_route(
                route=route,
                L_current=Lcur,
            )
            if self.Barge_cap[barge_idx] < max(edge_list_barge_idx):
                self.f_ck[cont, barge_idx] = 0
                self.route_dict[barge_idx]["route"] = old_route
                self.route_dict[barge_idx]["timing"] = old_timing
                self.route_dict[barge_idx]["repaired"] = old_repaired

                continue

            timing = self.get_timing(route, Lcur)

            if timing is not None:
                self.route_dict[barge_idx]["route"] = route
                self.route_dict[barge_idx]["timing"] = timing
                self.route_dict[barge_idx]["repaired"] = False

                # 🔴 EVALUATE INTERMEDIATE STATE
                cost = self.evaluate()

                self.cost_list.append(cost)

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_fck = copy.deepcopy(self.f_ck)
                    self.best_route_dict = copy.deepcopy(self.route_dict)
                    self.best_Barge_cap = copy.deepcopy(self.Barge_cap)
                    self.best_H_b = copy.deepcopy(self.H_b)

            else:
                # print("Failed to reinsert container from truck to barge")
                # revert
                self.f_ck[cont, barge_idx] = 0
                self.route_dict[barge_idx]["route"] = old_route
                self.route_dict[barge_idx]["timing"] = old_timing
                self.route_dict[barge_idx]["repaired"] = old_repaired

        return

    def operator_move(self):

        # 1) pick container container (unchanged)

        if rng.random() < self.critical_move_prob:
            crital_container_chosen = True
            trucked = [cont for cont in range(self.C) if not any(self.f_ck[cont])]
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
            len(self.route_dict[k]["route"]) >= 2 for k in range(self.K)
        ), "invalid route in route_dict before move"

        # Save old state
        old_row = copy.deepcopy(self.f_ck[container, :])
        old_Barge_cap = copy.deepcopy(self.Barge_cap)
        old_H_b = copy.deepcopy(self.H_b)
        old_route_from_b = (
            self.route_dict[from_b]["route"] if from_b != "truck" else None
        )
        old_route_to_b = self.route_dict[to_b]["route"] if to_b != "truck" else None

        old_timing_from_b = (
            self.route_dict[from_b]["timing"] if from_b != "truck" else None
        )
        old_timing_to_b = self.route_dict[to_b]["timing"] if to_b != "truck" else None
        old_repaired_from_b = (
            self.route_dict[from_b]["repaired"] if from_b != "truck" else None
        )
        old_repaired_to_b = (
            self.route_dict[to_b]["repaired"] if to_b != "truck" else None
        )

        # 4) tentative apply
        if from_b != "truck":
            self.f_ck[container, from_b] = 0
            self.route_dict.pop(from_b)

        if to_b != "truck":
            self.f_ck[container, to_b] = 1
            self.route_dict.pop(to_b)

        if to_b != "truck":
            Lcur_to = _get_L_current_for_barge(
                barge_idx=to_b,
                f_ck=self.f_ck,
                C=self.C,
                C_dict=self.C_dict,
            )

            if len(Lcur_to) == 0:
                # empty barge → trivial route
                entry_to_b = self.route_dict.setdefault(to_b, {})
                entry_to_b.setdefault("route", [0, 0])
                entry_to_b.setdefault("repaired", False)
                entry_to_b.setdefault("timing", {0: 0, 0: 0})

            else:
                route_to_b = self.get_route(Lcur_to)
                timing_to_b = self.get_timing(route_to_b, Lcur_to)
                entry_to_b = self.route_dict.setdefault(to_b, {})
                entry_to_b.setdefault("route", route_to_b)
                entry_to_b.setdefault("repaired", False)
                entry_to_b.setdefault("timing", timing_to_b)

        if from_b != "truck":
            Lcur_from = _get_L_current_for_barge(
                barge_idx=from_b,
                f_ck=self.f_ck,
                C=self.C,
                C_dict=self.C_dict,
            )

            if len(Lcur_from) == 0:
                # empty barge → trivial route
                entry_from_b = self.route_dict.setdefault(from_b, {})
                entry_from_b.setdefault("route", [0, 0])
                entry_from_b.setdefault("repaired", False)
                entry_from_b.setdefault("timing", {0: 0, 0: 0})
            else:
                entry_from_b = self.route_dict.setdefault(from_b, {})
                route_from_b = self.get_route(Lcur_from)
                timing_from_b = self.get_timing(route_from_b, Lcur_from)
                entry_from_b.setdefault("route", route_from_b)
                entry_from_b.setdefault("repaired", False)
                entry_from_b.setdefault("timing", timing_from_b)

        # 5) CAPACITY CHECK
        req_cap_per_route = []

        for k in range(self.K):
            edge_loads_k = _edge_loads_along_route(
                route=self.route_dict[k]["route"],
                L_current=_get_L_current_for_barge(
                    barge_idx=k,
                    f_ck=self.f_ck,
                    C=self.C,
                    C_dict=self.C_dict,
                ),
            )

            req_cap_per_route.append(max(edge_loads_k))

        has_violation = any(
            self.Barge_cap[k] < req_cap_per_route[k] for k in range(self.K)
        )

        if has_violation:
            result = self.reassign_barges_by_capacity(req_cap_per_route)
            if result != (None, None):
                self.Barge_cap, self.H_b = result
                self.Barge_cap, self.H_b = self.reassign_barges_by_cost_requirements(
                    req_cap_per_route
                )
            else:
                self.f_ck[container, :] = old_row
                if from_b != "truck":
                    self.route_dict[from_b]["route"] = old_route_from_b
                    self.route_dict[from_b]["repaired"] = old_repaired_from_b
                    self.route_dict[from_b]["timing"] = old_timing_from_b
                if to_b != "truck":
                    self.route_dict[to_b]["route"] = old_route_to_b
                    self.route_dict[to_b]["repaired"] = old_repaired_to_b
                    self.route_dict[to_b]["timing"] = old_timing_to_b
                self.Barge_cap = old_Barge_cap
                self.H_b = old_H_b
                self.T1[move] = self.tenure_move_container

                return False

        # update old state after reassignment
        old_Barge_cap = copy.deepcopy(self.Barge_cap)
        old_H_b = copy.deepcopy(self.H_b)

        # 6) TIMING CHECK (only for affected barge if not truck)
        affected_barges = set()

        if from_b != "truck":
            affected_barges.add(from_b)

        if to_b != "truck":
            affected_barges.add(to_b)

        for barge_idx in affected_barges:
            Lcur = _get_L_current_for_barge(
                barge_idx=barge_idx,
                f_ck=self.f_ck,
                C=self.C,
                C_dict=self.C_dict,
            )

            route = self.route_dict[barge_idx]["route"]

            timing = self.get_timing(route, Lcur)

            # 7) MILP repair if timing failed
            if timing is None:
                assigned = [i for i in range(self.C) if self.f_ck[i, barge_idx]]
                new_route, new_timing = repair_route(
                    assigned,
                    self.C_dict,
                    self.Barge_cap[barge_idx],
                    self.T_ij,
                    self.Handling_time,
                )
                self.milp_calls += 1

                if new_route is None:
                    # undo everything
                    self.f_ck[container, :] = old_row
                    if from_b != "truck":
                        self.route_dict[from_b]["route"] = old_route_from_b
                        self.route_dict[from_b]["repaired"] = old_repaired_from_b
                        self.route_dict[from_b]["timing"] = old_timing_from_b
                    if to_b != "truck":
                        self.route_dict[to_b]["route"] = old_route_to_b
                        self.route_dict[to_b]["repaired"] = old_repaired_to_b
                        self.route_dict[to_b]["timing"] = old_timing_to_b
                    self.Barge_cap = old_Barge_cap
                    self.H_b = old_H_b
                    self.T1[move] = self.tenure_move_container
                    return False
                if new_route is not None:
                    self.milp_repairs += 1
                    self.route_dict[barge_idx]["route"] = new_route
                    self.route_dict[barge_idx]["timing"] = new_timing
                    self.route_dict[barge_idx]["repaired"] = True
                    # hard capacity gate on the repaired route
                    Lcur = _get_L_current_for_barge(
                        barge_idx=barge_idx,
                        f_ck=self.f_ck,
                        C=self.C,
                        C_dict=self.C_dict,
                    )
                    edge_list_barge_idx = _edge_loads_along_route(
                        route=new_route,
                        L_current=Lcur,
                    )
                    if self.Barge_cap[barge_idx] < max(edge_list_barge_idx):
                        self.f_ck[container] = old_row
                        if from_b != "truck":
                            self.route_dict[from_b]["route"] = old_route_from_b
                            self.route_dict[from_b]["repaired"] = old_repaired_from_b
                            self.route_dict[from_b]["timing"] = old_timing_from_b
                        if to_b != "truck":
                            self.route_dict[to_b]["route"] = old_route_to_b
                            self.route_dict[to_b]["repaired"] = old_repaired_to_b
                            self.route_dict[to_b]["timing"] = old_timing_to_b
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

    def operator_swap(self):
        c1, c2 = rng.choice(self.C, size=2, replace=False)
        bs1 = [k for k in range(self.K) if self.f_ck[c1, k]]
        bs2 = [k for k in range(self.K) if self.f_ck[c2, k]]
        if not bs1 or not bs2 or bs1[0] == bs2[0]:
            return False
        b1, b2 = bs1[0], bs2[0]

        move = ((c1, b1, b2), (c2, b2, b1))
        if move in self.T1 or b1 in self.T3 or b2 in self.T3:
            return False

        # tentatively swap
        old1 = copy.deepcopy(self.f_ck[c1])
        old2 = copy.deepcopy(self.f_ck[c2])
        self.f_ck[c1, b1] = 0
        self.f_ck[c1, b2] = 1
        self.f_ck[c2, b2] = 0
        self.f_ck[c2, b1] = 1

        old_route_b1 = self.route_dict[b1]["route"]
        old_route_b2 = self.route_dict[b2]["route"]
        old_timing_b1 = self.route_dict[b1]["timing"]
        old_timing_b2 = self.route_dict[b2]["timing"]
        old_repaired_b1 = self.route_dict[b1]["repaired"]
        old_repaired_b2 = self.route_dict[b2]["repaired"]

        # invalidate routes affected by the swap
        self.route_dict.pop(b1)
        self.route_dict.pop(b2)

        Lcur_b1 = _get_L_current_for_barge(
            barge_idx=b1,
            f_ck=self.f_ck,
            C=self.C,
            C_dict=self.C_dict,
        )
        Lcur_b2 = _get_L_current_for_barge(
            barge_idx=b2,
            f_ck=self.f_ck,
            C=self.C,
            C_dict=self.C_dict,
        )

        if len(Lcur_b1) == 0:
            # empty barge → trivial route
            entry_b1 = self.route_dict.setdefault(b1, {})
            entry_b1.setdefault("route", [0, 0])
            entry_b1.setdefault("repaired", False)
            entry_b1.setdefault("timing", {0: 0, 0: 0})
        else:
            route_b1 = self.get_route(Lcur_b1)
            timing_b1 = self.get_timing(route_b1, Lcur_b1)
            entry_b1 = self.route_dict.setdefault(b1, {})
            entry_b1.setdefault("route", route_b1)
            entry_b1.setdefault("repaired", False)
            entry_b1.setdefault("timing", timing_b1)
        if len(Lcur_b2) == 0:
            # empty barge → trivial route
            entry_b2 = self.route_dict.setdefault(b2, {})
            entry_b2.setdefault("route", [0, 0])
            entry_b2.setdefault("repaired", False)
            entry_b2.setdefault("timing", {0: 0, 0: 0})
        else:
            route_b2 = self.get_route(Lcur_b2)
            timing_b2 = self.get_timing(route_b2, Lcur_b2)
            entry_b2 = self.route_dict.setdefault(b2, {})
            entry_b2.setdefault("route", route_b2)
            entry_b2.setdefault("repaired", False)
            entry_b2.setdefault("timing", timing_b2)

        def barge_ok(k):
            assigned = [i for i in range(self.C) if self.f_ck[i, k]]
            if not assigned:
                return True

            # one‐shift TW
            # ---- time-window check (identical logic to Greedy) ----
            Lcur = _get_L_current_for_barge(
                barge_idx=k,
                f_ck=self.f_ck,
                C=self.C,
                C_dict=self.C_dict,
            )

            route = self.route_dict[k]["route"]

            edge_list_barge_k = _edge_loads_along_route(
                route=route,
                L_current=Lcur,
            )
            if self.Barge_cap[k] < max(edge_list_barge_k):
                return False

            if self.route_dict[k]["repaired"]:
                # print("Route already repaired once by MILP, skipping timing check for barge", k)
                return True
            else:
                timing = self.get_timing(route, Lcur)

            success = timing is not None

            return success

        ok1 = barge_ok(b1)
        ok2 = barge_ok(b2)
        if ok1 and ok2:
            return True

        # quick check failed on at least one barge: call repair on each
        for b in (b1, b2):
            assigned = [i for i in range(self.C) if self.f_ck[i, b]]

            new_route, new_timing = repair_route(
                assigned,
                self.C_dict,
                self.Barge_cap[b],
                self.T_ij,
                self.Handling_time,
            )
            # print("New route from MILP repair:", new_route)
            self.milp_calls += 1

            if new_route is None:
                # irreparable swap → undo + tabu
                self.f_ck[c1] = old1
                self.f_ck[c2] = old2
                self.route_dict[b1]["route"] = old_route_b1
                self.route_dict[b1]["timing"] = old_timing_b1
                self.route_dict[b1]["repaired"] = old_repaired_b1
                self.route_dict[b2]["route"] = old_route_b2
                self.route_dict[b2]["timing"] = old_timing_b2
                self.route_dict[b2]["repaired"] = old_repaired_b2
                self.T1[move] = self.tenure_move_container
                return False
            else:
                self.milp_repairs += 1
                self.route_dict[b]["route"] = new_route
                self.route_dict[b]["timing"] = new_timing
                self.route_dict[b]["repaired"] = True
                # hard capacity gate on the repaired route
                Lcur = _get_L_current_for_barge(
                    barge_idx=b,
                    f_ck=self.f_ck,
                    C=self.C,
                    C_dict=self.C_dict,
                )
                edge_list_barge_b = _edge_loads_along_route(
                    route=new_route,
                    L_current=Lcur,
                )

                if self.Barge_cap[b] < max(edge_list_barge_b):
                    self.f_ck[c1] = old1
                    self.f_ck[c2] = old2
                    self.route_dict[b1]["route"] = old_route_b1
                    self.route_dict[b2]["route"] = old_route_b2
                    self.route_dict[b1]["timing"] = old_timing_b1
                    self.route_dict[b1]["repaired"] = old_repaired_b1
                    self.route_dict[b2]["timing"] = old_timing_b2
                    self.route_dict[b2]["repaired"] = old_repaired_b2
                    self.T1[move] = self.tenure_move_container
                    return False
        return True

    def evaluate(self):
        total_cost = 0
        total_stops = 0
        utils = []
        self.x_ijk = np.zeros((self.N, self.N, len(self.Barge_cap)))

        for k in range(self.K):
            assigned = np.where(self.f_ck[:, k] == 1)[0].tolist()
            if not assigned:
                continue
            Lcur_k = {cont: self.C_dict[cont] for cont in assigned}
            route_k = self.route_dict[k]["route"]

            for i in range(len(route_k) - 1):
                if route_k[i] != route_k[i + 1]:
                    self.x_ijk[route_k[i]][route_k[i + 1]][k] = 1

            # util

            edge_loads_k = _edge_loads_along_route(
                route=route_k,
                L_current=Lcur_k,
            )
            util = max(edge_loads_k) / self.Barge_cap[k]
            assert util <= 1.0, "capacity violation detected in evaluation"
            utils.append(util)
            total_stops += len(route_k) - 2  # exclude depot visits

        # barge cost
        total_cost += self.calculate_objective()

        # truck
        unassigned = np.where(self.f_ck.sum(axis=1) == 0)[0]
        for cont in unassigned:
            total_cost += self.H_t_dict[self.C_dict[cont]["Wc"]]

        return total_cost

    def local_search(self, max_iters=3000):
        print("\n ---- Starting Meta-Heuristic Search... ----\n")
        self.best_cost = self.evaluate()
        self.cost_list.append(self.best_cost)
        self.best_fck = copy.deepcopy(self.f_ck)
        self.best_route_dict = copy.deepcopy(self.route_dict)
        self.best_Barge_cap = copy.deepcopy(self.Barge_cap)
        self.best_H_b = copy.deepcopy(self.H_b)
        no_improve = 0

        it_list = []
        cost_list = []
        best_cost_list = []

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
            self.cost_list.append(cost)

            if cost < self.best_cost:
                self.best_cost, self.best_fck = cost, copy.deepcopy(self.f_ck)
                self.best_route_dict = copy.deepcopy(self.route_dict)
                self.best_Barge_cap = copy.deepcopy(self.Barge_cap)
                self.best_H_b = copy.deepcopy(self.H_b)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.shake_threshold:
                old_best = self.best_cost
                self._shake()

                if self.best_cost < old_best:
                    # restart from improved shaken solution
                    self.f_ck = copy.deepcopy(self.best_fck)
                    self.route_dict = copy.deepcopy(self.best_route_dict)
                    self.Barge_cap = copy.deepcopy(self.best_Barge_cap)
                    self.H_b = copy.deepcopy(self.best_H_b)
                self.shake_count += 1
                no_improve = 0

            it_list.append(it)
            cost_list.append(cost)
            best_cost_list.append(self.best_cost)

        #     # Update plot every iteration
        #     line1.set_data(it_list, cost_list)
        #     line2.set_data(it_list, best_cost_list)
        #     ax.relim()
        #     ax.autoscale_view()
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()

        # plt.ioff()

        plt.figure()
        plt.plot(it_list, cost_list, label="Cost")
        plt.show()

        self.f_ck = copy.deepcopy(self.best_fck)
        self.Barge_cap = copy.deepcopy(self.best_Barge_cap)
        self.H_b = copy.deepcopy(self.best_H_b)
        self.route_dict = copy.deepcopy(self.best_route_dict)

        print("\nFinal route dictionary:", self.best_route_dict, "\n")

        fig, file_path = timing_window_plot(
            C=self.C,
            K=self.K,
            C_dict=self.C_dict,
            f_ck=self.f_ck,
            MH_or_Greedy="MH",
            scenario_name=self.scenario_name,
            final_route_dict=self.route_dict,
        )

        print("\nMeta-Heuristic Search Complete, search move analysis:")
        print(f"Total move accepts: {self.move_accepts}")
        print(f"Total swap accepts: {self.swap_accepts}")
        print(f"Total MILP repair calls: {self.milp_calls}")
        print(f"Total shakes performed: {self.shake_count}")
        print(f"MILP repairs succeeded: {self.milp_repairs}\n")

        return self.best_cost, self.best_fck, self.best_route_dict, fig, file_path
