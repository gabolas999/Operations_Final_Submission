#!/usr/bin/env python3
"""
Greedy Algorithm for Container Allocation Optimization

This module provides a unified class-based implementation of the greedy algorithm
for container-to-barge allocation optimization.

This file does work #+#+# Gabo

TODO: Implement a re-run verification tool - and include the video in the report -> instant 10.

"""

import random
import math
import networkx as nx
import numpy as np
from dataclasses import dataclass

from MILP import MILP_Algo


@dataclass
class GreedySolution:
    total_cost: float
    barge_cost: float
    truck_cost: float
    f_ck_init: np.ndarray
    route_list: list
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
        problem_instance=None,
    ):

        self.instance = problem_instance

        self.generate_master_route()
        self.generate_ordered_containers()
        self._sort_barges_by_capacity_desc()

    def _sort_barges_by_capacity_desc(self):
        """Sort barges by decreasing capacity, keeping fixed costs paired (Algorithm 1, line 2)."""
        pairs = sorted(
            zip(self.instance.Qk, self.instance.H_b), key=lambda p: p[0], reverse=True
        )
        self.Barges = [q for q, _ in pairs]
        self.H_b = [h for _, h in pairs]

    def generate_master_route(self):
        """Generate master route using TSP approximation"""

        n = len(self.instance.T_ij_matrix)
        G = nx.complete_graph(n)

        for i in range(n):
            for j in range(n):
                if i != j:
                    G[i][j]["weight"] = self.instance.T_ij_matrix[i][j]

        # Find approximate TSP cycle (returns to start)
        self.master_route = nx.approximation.traveling_salesman_problem(
            G, cycle=True, weight="weight"
        )

    def generate_ordered_containers(self):
        """Generate ordered list of containers based on master route"""
        self.C_ordered = []
        condit_satisfies_counter = 0

        for i in self.master_route[1:-1]:  # Skip first and last (depot)
            for c, info in self.instance.C_dict.items():
                if info["Terminal"] == i:
                    condit_satisfies_counter += 1
                    self.C_ordered.append(c)

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

    def get_timing(self, route, L_current, departure_shift):
        """
        Returns:
        D_terminal: departure times at each route node (after service)
        O_terminal: arrival times at each route node (before service)

        Interpretation (per your spec):
        - Base departure from dry port is max export release time at dry port (or 0 if no exports).
        - Then we add a uniform departure_shift (>=0) to postpone the whole trip.
        - Arrival times are propagated iteratively along the route.
        """
        # 1) Base departure time from dry port
        export_release_times = [
            c["Rc"] for c in L_current.values() if c["In_or_Out"] == 2
        ]
        base_departure = max(export_release_times) if export_release_times else 0.0
        depart_time = base_departure + departure_shift

        # 2) Iterative propagation
        O_terminal = [0.0]  # arrival at dry port is time 0 reference
        D_terminal = [depart_time]  # depart dry port at computed time

        current_node = 0
        current_depart = depart_time

        for node in route[1:]:
            travel = self.instance.T_ij_matrix[current_node][node]
            arrival = current_depart + travel

            if node == 0:
                # returning to dry port: no service time
                service = 0
            else:
                # handling time at node
                n_containers_here = sum(
                    1 for c in L_current.values() if c["Terminal"] == node
                )
                service = self.instance.Handling_time * n_containers_here

            depart = arrival + service

            O_terminal.append(arrival)
            D_terminal.append(depart)

            current_node = node
            current_depart = depart

        return D_terminal, O_terminal

    def check_for_cap(self, route, L_current, barge_idx, barges=None):
        cap = barges[barge_idx] if barges is not None else self.Barges[barge_idx]

        # Start at depot: all exports are loaded
        load = sum(c["Wc"] for c in L_current.values() if c["In_or_Out"] == 2)
        if load > cap:
            return False
        if load < 0:
            return False

        # Visit terminals once in the given route
        for terminal in route[1:]:
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

            if load > cap:
                return False
            if load < 0:
                return False

        return True

    def delay_window(self, container, O_terminal, route, terminal):
        """
        Calculate delay needed for container to fit in time window if the arrival is too early.
        Late arrivals are not adjusted as any delay would make the issue worse, so we return 0 in that case.

        Parameters:
        -----------
        container : dict
            Container information
        O_terminal : list
            Arrival times at each terminal
        route : list
            Route as list of terminal indices
        terminal : int
            Terminal index

        Returns:
        --------
        float : Required delay in hours
        """
        Oc = container["Oc"]
        arrival = O_terminal[route.index(terminal)]
        return max(0.0, Oc - arrival)

    def solve_greedy(self):
        """
        Solve the container allocation problem using greedy algorithm

        Returns:
        --------
        dict : Solution results including costs and assignments
        """
        self.f_ck_init = np.zeros(
            (self.instance.C, len(self.Barges))
        )  # matrix for container to barge assignment

        barge_idx = 0
        to_ignore = []  # list to store containers that can be removed from C_ordered
        departure_delay = 0  # carry this forward across containers
        self.barge_departure_delay = []
        self.route_list = []

        while barge_idx < len(self.Barges):
            for c in self.C_ordered:
                if c in to_ignore:
                    continue

                # 1) Tentatively assign c to this barge
                self.f_ck_init[c, barge_idx] = 1

                # 2) Build the current load
                L_current = {
                    cont: self.instance.C_dict[cont]
                    for cont in self.C_ordered
                    if self.f_ck_init[cont, barge_idx] == 1
                }
                route = self.get_route(L_current)

                # 3) Capacity check
                if not self.check_for_cap(route, L_current, barge_idx):
                    self.f_ck_init[c, barge_idx] = 0
                    continue

                # 4) Time‐window check, with up to one "departure shift"
                success = False
                delay = departure_delay  # start from whatever delay we already have

                # Try once to adjust departure (max_tries = 1)
                for attempt in range(2):  # attempt = 0 (no shift), attempt = 1 (shift)
                    D_term, O_term = self.get_timing(route, L_current, delay)

                    # find any containers that now violate
                    early_arrival_violations = []
                    late = False
                    for cont in L_current.values():
                        t = cont["Terminal"]

                        arrival = O_term[route.index(t)]

                        if arrival < cont["Oc"]:
                            early_arrival_violations.append(cont)
                        elif arrival > cont["Dc"]:
                            late = True
                            break

                    if late:
                        break

                    if not early_arrival_violations:
                        # everyone fits under this `delay`
                        success = True
                        break

                    # if we still have our one "shift" left, compute the shift
                    if attempt == 0:
                        # largest extra wait delay_needed
                        delay_needed = [
                            self.delay_window(
                                container=v,
                                O_terminal=O_term,
                                route=route,
                                terminal=v["Terminal"],
                            )
                            for v in early_arrival_violations
                        ]
                        delay += max(delay_needed)  # accumulate shift
                    else:
                        # second pass and still violations → fail
                        break

                if success:
                    # commit this shift permanently for the rest of this barge
                    departure_delay = delay
                    to_ignore.append(c)
                else:
                    # undo assignment
                    self.f_ck_init[c, barge_idx] = 0

            # move on to next barge: store the FINAL route for this barge
            assigned_idx = np.where(self.f_ck_init[:, barge_idx] == 1)[0].tolist()
            if assigned_idx:
                L_final = {cont: self.instance.C_dict[cont] for cont in assigned_idx}
                route_final = self.get_route(L_final)
            else:
                route_final = [0, 0]

            self.route_list.append(route_final)
            self.barge_departure_delay.append(departure_delay)
            barge_idx += 1
            departure_delay = 0

        # Calculate trucked containers
        index_to_be_trucked = np.where(np.sum(self.f_ck_init, axis=1) == 0)[0].tolist()
        self.trucked_containers = {
            i: self.instance.C_dict[i] for i in index_to_be_trucked
        }

        # Calculate trucking cost
        self.truck_cost = 0
        for i in self.trucked_containers:
            if self.instance.C_dict[i]["Wc"] == 1:  # 20ft container
                self.truck_cost += self.instance.H_t_20
            else:  # 40ft container
                self.truck_cost += self.instance.H_t_40

        # Calculate barge routing matrix
        self.x_ijk = np.zeros(
            (len(self.Barges), self.instance.N, self.instance.N)
        )  # xijk[k][i][j] = 1 if barge k goes from terminal i to terminal j

        for barge_idx, route in enumerate(self.route_list):
            for i in range(len(route) - 1):
                if route[i] != route[i + 1]:
                    self.x_ijk[barge_idx][route[i]][route[i + 1]] = 1

        # Calculate barge cost
        self.barge_cost = self.calculate_objective()

        # Calculate total cost
        self.total_cost = self.barge_cost + self.truck_cost

        solution = GreedySolution(
            total_cost=self.total_cost,
            barge_cost=self.barge_cost,
            truck_cost=self.truck_cost,
            f_ck_init=self.f_ck_init,
            route_list=self.route_list,
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
        number_of_barged = self.instance.C - number_of_trucked
        print(f"Number of barged containers: {number_of_barged}")
        print("---------------------------------- \n")

        return solution

    def calculate_objective(self):
        """
        Calculate objective function value (barge costs)

        Returns:
        --------
        float : Total barge cost
        """
        K = len(self.H_b)  # number of barges
        cost = 0

        for k in range(K):
            # 1) fixed‐cost term: sum over j≠0 of x[0][j][k]*H_b[k]
            for j in range(self.instance.N):
                if j == 0:
                    continue
                cost += self.x_ijk[k][0][j] * self.H_b[k]

            # 2) travel‐time term: sum over all i,j of T[i][j]*x[i][j][k]
            for i in range(self.instance.N):
                for j in range(self.instance.N):
                    cost += self.instance.T_ij_matrix[i][j] * self.x_ijk[k][i][j]

            # 3) stop penalty: count once per visited sea terminal (j != 0)
            for j in range(1, self.instance.N):
                if self.x_ijk[k][:, j].sum() > 0:
                    cost += self.instance.Gamma

        return cost

    def print_results(self):
        """Print detailed results of the optimization"""
        print(f"Total cost: {self.total_cost:>10.0f} Euros")
        print(
            f"Barge cost: {self.barge_cost:>10.0f} Euros             ({self.barge_cost / self.total_cost * 100:>5.1f}% )"
        )
        print(
            f"Truck cost: {self.truck_cost:>10.0f} Euros             ({self.truck_cost / self.total_cost * 100:>5.1f}% )"
        )
        print(f"Containers: {self.instance.C:>10d}")
        print(f"Terminals: {self.instance.N:>10d}")
        print(
            f"Trucked containers: {len(self.trucked_containers):>10d}           ({len(self.trucked_containers) / self.instance.C * 100:>5.1f}% )"
        )

        # Print barge utilization
        for k, route in enumerate(self.route_list):
            if len(route) > 1:  # Only print if barge is used
                containers_on_barge = sum(
                    1 for c in range(self.instance.C) if self.f_ck_init[c][k] == 1
                )
                teu_on_barge = sum(
                    self.instance.C_dict[c]["Wc"]
                    for c in range(self.instance.C)
                    if self.f_ck_init[c][k] == 1
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
