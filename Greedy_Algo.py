#!/usr/bin/env python3
"""
Greedy Algorithm for Container Allocation Optimization

This module provides a unified class-based implementation of the greedy algorithm
for container-to-barge allocation optimization.

This file does work #+#+# Gabo

TODO: Implement a re-run verification tool - and include the video in the report -> instant 10.

"""

import networkx as nx
import numpy as np
from dataclasses import dataclass

from MILP import MILP_Algo

from helpers import timing_window_plot, _edge_loads_along_route


@dataclass
class GreedySolution:
    total_cost: float
    barge_cost: float
    truck_cost: float
    f_ck: np.ndarray
    route_dict: dict
    trucked_containers: dict
    xijk: np.ndarray
    C_ordered: list
    H_b: list
    Barges: list


class GreedyOptimizer:
    """Unified greedy algorithm class for container allocation optimization"""

    def __init__(
        self,
        # reduced=False,
        scenario_name=None,
        problem_instance=None,
    ):

        self.scenario_name = scenario_name
        self.C = problem_instance.C
        self.C_dict = problem_instance.C_dict
        self.N = problem_instance.N
        self.T_ij_matrix = problem_instance.T_ij_matrix
        self.Gamma = problem_instance.Gamma
        self.K = len(problem_instance.K_list[:-1])  # exclude the truck
        self.Qk = problem_instance.Qk
        self.H_b = problem_instance.H_b
        self.Handling_time = problem_instance.Handling_time
        self.Ht20 = problem_instance.H_t_20
        self.Ht40 = problem_instance.H_t_40

        self.H_t_dict = {1: self.Ht20, 2: self.Ht40}

        self.generate_master_route()
        self.generate_ordered_containers()
        self._sort_barges_by_capacity_desc()

    def _sort_barges_by_capacity_desc(self):
        """Sort barges by decreasing capacity, keeping fixed costs paired (Algorithm 1, line 2)."""
        pairs = sorted(zip(self.Qk, self.H_b), key=lambda p: p[0], reverse=True)
        self.Barges = [q for q, _ in pairs]
        self.H_b = [h for _, h in pairs]

    def rotate_cycle_to_start(self, cycle, start_node=0):
        if cycle[0] == cycle[-1]:
            cycle = cycle[:-1]  # remove duplicate end

        idx = cycle.index(start_node)
        rotated = cycle[idx:] + cycle[:idx] + [start_node]
        return rotated

    def generate_master_route(self):
        """Generate master route using TSP approximation"""

        n = len(self.T_ij_matrix)

        assert n == self.N, "T_ij_matrix size mismatch with N"

        G = nx.complete_graph(n)

        for i in range(n):
            for j in range(n):
                if i != j:
                    G[i][j]["weight"] = self.T_ij_matrix[i][j]

        # Find approximate TSP cycle (returns to start)
        cycle = nx.approximation.traveling_salesman_problem(
            G, cycle=True, weight="weight"
        )
        self.master_route = self.rotate_cycle_to_start(cycle, start_node=0)

        assert len(self.master_route) == self.N + 1, "Invalid master route length"
        for terminal in self.master_route:
            assert isinstance(terminal, int), "Terminal indices must be integers"
        assert len(self.master_route[1:-1]) == len(
            set(self.master_route[1:-1])
        ), "Terminals repeated in TSP cycle"
        assert 0 in cycle, "Dry port (0) missing from TSP cycle"
        assert (
            self.master_route[0] == 0 and self.master_route[-1] == 0
        ), "Cycle must start and end at dry port (0)"

    def generate_ordered_containers(self):
        """Generate ordered list of containers based on master route"""
        self.C_ordered = []
        condit_satisfies_counter = 0

        for i in self.master_route:
            if i == 0:
                continue  # Skip first and last (depot)
            for c, info in self.C_dict.items():
                if info["Terminal"] == i:
                    condit_satisfies_counter += 1
                    self.C_ordered.append(c)

        assert (
            condit_satisfies_counter == self.C
        ), "Not all containers included in ordered list"

    def get_route(self, L_current):
        terminals = {c["Terminal"] for c in L_current.values() if c["Terminal"] != 0}

        route = [0]
        # Follow the global master route order (unique terminals)
        if hasattr(self, "master_route") and self.master_route:
            for t in self.master_route:
                if t != 0 and t in terminals and t not in route:
                    route.append(t)
        else:
            route.extend(sorted(terminals))

        route.append(0)  # return to depot

        return route

    def get_timing(self, route, L_current):
        """
        Returns
        -------
        D_terminal : list[float]
            Departure times at each route node (after service)
        O_terminal : list[float]
            Arrival times at each route node (before service)

        Returns None if the route is time-infeasible.

        Interpretation:
        - Departure from dry port is max export release time (or 0 if no exports)
        - Arrival times propagate forward
        - If arrival < opening time -> wait
        - If arrival > closing time -> infeasible
        """

        # -----------------------------
        # 1) Base departure from depot
        # -----------------------------
        export_release_times = [
            c["Rc"] for c in L_current.values() if c["In_or_Out"] == 2
        ]
        current_time = max(export_release_times) if export_release_times else 0.0

        O_terminal = [current_time]  # arrival at depot
        D_terminal = [current_time]  # departure from depot

        current_node = route[0]
        assert current_node == 0, "Route must start at dry port (0)"

        # -----------------------------
        # 2) Forward propagation
        # -----------------------------
        for node in route[1:]:

            # travel
            travel = self.T_ij_matrix[current_node][node]
            arrival = current_time + travel

            if node != 0:
                # containers handled at this terminal
                containers_here = [
                    c for c in L_current.values() if c["Terminal"] == node
                ]

                if containers_here:
                    Oj = max(c["Oc"] for c in containers_here)
                    Dj = min(c["Dc"] for c in containers_here)

                    # WAIT if early
                    arrival = max(arrival, Oj)

                    # FAIL if late
                    if arrival > Dj:
                        return None

                    service = self.Handling_time * len(containers_here)
                else:
                    service = 0.0
            else:
                # depot on return
                service = 0.0

            depart = arrival + service

            O_terminal.append(arrival)
            D_terminal.append(depart)

            current_time = depart
            current_node = node

        # -----------------------------
        # 3) Sanity checks
        # -----------------------------
        assert len(O_terminal) == len(D_terminal) == len(route), (
            f"Timing length mismatch: "
            f"O={len(O_terminal)}, D={len(D_terminal)}, route={len(route)}"
        )

        timing = dict(zip(route, O_terminal))

        return timing

    # def delay_window(self, container, O_terminal, route, terminal):
    #     """
    #     Calculate delay needed for container to fit in time window if the arrival is too early.
    #     Late arrivals are not adjusted as any delay would make the issue worse, so we return 0 in that case.

    #     Parameters:
    #     -----------
    #     container : dict
    #         Container information
    #     O_terminal : list
    #         Arrival times at each terminal
    #     route : list
    #         Route as list of terminal indices
    #     terminal : int
    #         Terminal index

    #     Returns:
    #     --------
    #     float : Required delay in hours
    #     """
    #     Oc = container["Oc"]
    #     arrival = O_terminal[route.index(terminal)]
    #     return max(0.0, Oc - arrival)

    def solve_greedy(self):
        """
        Solve the container allocation problem using greedy algorithm

        Returns:
        --------
        dict : Solution results including costs and assignments
        """
        self.f_ck = np.zeros(
            (self.C, len(self.Barges))
        )  # matrix for container to barge assignment

        barge_idx = 0
        to_ignore = []  # list to store containers that can be removed from C_ordered
        self.barge_departure_delay = []
        self.route_dict = {}

        while barge_idx < len(self.Barges):
            departure_delay = 0
            for c in self.C_ordered:
                if c in to_ignore:
                    continue

                # 1) Tentatively assign c to this barge
                self.f_ck[c, barge_idx] = 1

                # 2) Build the current load
                L_current = {
                    cont: self.C_dict[cont]
                    for cont in self.C_ordered
                    if self.f_ck[cont, barge_idx] == 1
                }
                route = self.get_route(L_current)

                # 3) Capacity check
                edge_list = _edge_loads_along_route(route, L_current)

                if self.Barges[barge_idx] < max(edge_list):
                    self.f_ck[c, barge_idx] = 0
                    continue

                timing = self.get_timing(route, L_current)
                if timing is not None:
                    to_ignore.append(c)
                else:
                    # undo assignment
                    self.f_ck[c, barge_idx] = 0

            # move on to next barge: store the FINAL route for this barge
            assigned_idx = np.where(self.f_ck[:, barge_idx] == 1)[0].tolist()
            if len(assigned_idx) > 0:
                L_final = {cont: self.C_dict[cont] for cont in assigned_idx}
                route_final = self.get_route(L_final)
                timing_final = self.get_timing(route_final, L_final)
            else:
                route_final = [0, 0]
                timing_final = {0: 0, 0: 0}

            assert (
                timing_final is not None
            ), f"Final route for barge {barge_idx} is infeasible: {route_final}"

            self.route_dict.setdefault(barge_idx, {}).setdefault("route", route_final)
            self.route_dict[barge_idx].setdefault("timing", timing_final)
            self.route_dict[barge_idx].setdefault("repaired", False)
            self.barge_departure_delay.append(departure_delay)
            barge_idx += 1

        # Calculate trucked containers

        sum_of_rows = np.sum(self.f_ck, axis=1)

        assert len(sum_of_rows) == self.C, "Mismatch in container count"

        index_to_be_trucked = np.where(sum_of_rows == 0)[0].tolist()

        if len(index_to_be_trucked) == 0:
            print("All containers assigned to barges.")
        else:
            print(f"{len(index_to_be_trucked)} containers will be trucked.")
            self.trucked_containers = {i: self.C_dict[i] for i in index_to_be_trucked}

        # Calculate trucking cost
        self.truck_cost = 0
        for i in self.trucked_containers:
            if self.C_dict[i]["Wc"] == 1:  # 20ft container
                self.truck_cost += self.Ht20
            else:  # 40ft container
                self.truck_cost += self.Ht40

        # Calculate barge routing matrix
        self.x_ijk = np.zeros(
            (self.N, self.N, len(self.Barges))
        )  # xijk[i][j][k] = 1 if barge k goes from terminal i to terminal j

        for barge_idx, route_info in self.route_dict.items():
            route = route_info["route"]
            for i in range(len(route) - 1):
                if route[i] != route[i + 1]:
                    self.x_ijk[route[i]][route[i + 1]][barge_idx] = 1

        # Calculate barge cost
        self.barge_cost = self.calculate_objective()

        # Calculate total cost
        self.total_cost = self.barge_cost + self.truck_cost

        fig, file_path = timing_window_plot(
            C=self.C,
            K=self.K,
            C_dict=self.C_dict,
            f_ck=self.f_ck,
            MH_or_Greedy="Greedy",
            scenario_name=self.scenario_name,
            final_route_dict=self.route_dict,
        )

        solution = GreedySolution(
            total_cost=self.total_cost,
            barge_cost=self.barge_cost,
            truck_cost=self.truck_cost,
            f_ck=self.f_ck,
            route_dict=self.route_dict,
            trucked_containers=self.trucked_containers,
            xijk=self.x_ijk,
            C_ordered=self.C_ordered,
            H_b=self.H_b,
            Barges=self.Barges,
        )

        print("\n---- Greedy solution computed ----")
        print("----------------------------------")
        print(f"Total cost: €{np.round(self.total_cost, 2)}")
        print(f"Barge cost: €{np.round(self.barge_cost, 2)}")
        print(f"Truck cost: €{np.round(self.truck_cost, 2)}")
        print("----------------------------------")
        number_of_trucked = len(self.trucked_containers)
        print(f"Number of trucked containers: {number_of_trucked}")
        number_of_barged = self.C - number_of_trucked
        print(f"Number of barged containers: {number_of_barged}")
        print("---------------------------------- \n")

        return solution

    def calculate_objective(self):
        total_cost = 0

        # =========================
        # BARGE COSTS
        # =========================
        for k in range(self.K):
            assigned = np.where(self.f_ck[:, k] == 1)[0]

            # barge not used
            if len(assigned) == 0:
                continue

            route = self.route_dict[k]["route"]

            # 1) fixed barge cost
            total_cost += self.H_b[k]

            # 2) travel cost
            for i in range(len(route) - 1):
                total_cost += self.T_ij_matrix[route[i]][route[i + 1]]

            # 3) stop cost (exclude depot)
            n_stops = len(route) - 2
            total_cost += n_stops * self.Gamma

        # =========================
        # TRUCK COSTS
        # =========================
        unassigned = np.where(self.f_ck.sum(axis=1) == 0)[0]
        for c in unassigned:
            total_cost += self.H_t_dict[self.C_dict[c]["Wc"]]

        return total_cost

    def print_results(self):
        """Print detailed results of the optimization"""
        print(f"Total cost: {self.total_cost:>10.0f} Euros")
        print(
            f"Barge cost: {self.barge_cost:>10.0f} Euros             ({self.barge_cost / self.total_cost * 100:>5.1f}% )"
        )
        print(
            f"Truck cost: {self.truck_cost:>10.0f} Euros             ({self.truck_cost / self.total_cost * 100:>5.1f}% )"
        )
        print(f"Containers: {self.C:>10d}")
        print(f"Terminals: {self.N:>10d}")
        print(
            f"Trucked containers: {len(self.trucked_containers):>10d}           ({len(self.trucked_containers) / self.C * 100:>5.1f}% )"
        )

        # Print barge utilization
        # TODO: fix the capacity check to make sure we dont exceed capacity,
        # because it should show peak utilization not absolute / total utilization
        for k, route in enumerate(self.route_list):
            if len(route) > 2:  # Only print if barge is used
                containers_on_barge = sum(
                    1 for c in range(self.C) if self.f_ck[c][k] == 1
                )
                teu_on_barge = sum(
                    self.C_dict[c]["Wc"] for c in range(self.C) if self.f_ck[c][k] == 1
                )
                print(
                    f"Barge {k:>3d}: "
                    f"{containers_on_barge:>4d} containers, "
                    f"{teu_on_barge:>4d}/{self.Barges[k]:<4d} TEU"
                )


if __name__ == "__main__":
    milp_instance = MILP_Algo(reduced=False)
    optimizer = GreedyOptimizer(problem_instance=milp_instance)
    results = optimizer.solve_greedy()
    optimizer.print_results()
