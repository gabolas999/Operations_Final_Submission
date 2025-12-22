#!/usr/bin/env python3
"""
Greedy Algorithm for Container Allocation Optimization

This module provides a unified class-based implementation of the greedy algorithm
for container-to-barge allocation optimization.

This file does work #+#+# Gabo

TODO: Implement a re-run verification tool - and include the video in the report -> instant 10. 


THIS FILE DID NOT MANAGE TO WORK; SPECIFICALLY, THE two lines:

    # greedy.print_results_2()
    # greedy.plot_barge_solution_map()

broke the code

"""

import random
import math
import networkx as nx
import numpy as np
from sklearn.manifold import MDS
import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt

class GreedyOptimizer:
    """Unified greedy algorithm class for container allocation optimization"""
   
    def __init__(
            self,
            qk = [  # Barge capacities in TEU
                104,  # Barge 0
                99,   # Barge 1
                81,   # Barge 2
                52,   # Barge 3
                28,   # Barge 4
            ],
            h_b = [  # Barge fixed costs in euros
                3700,  # Barge 0
                3600,  # Barge 1
                3400,  # Barge 2
                2800,  # Barge 3
                1800,  # Barge 4
            ],
            seed=100,
            reduced=False,
            h_t_40=200,                     # 40ft container trucking cost in euros
            h_t_20=140,                     # 20ft container trucking cost in euros
            handling_time=1/6,              # Container handling time in hours
            C_range=(100, 600),             # (min, max) number of containers when reduced=False
            N_range=(10, 20),               # (min, max) number of terminals when reduced=False
            
            Oc_range=(24, 196),             # (min, max) opening time in hours
            Oc_offset_range=(24, 120),      # (min_offset, max_offset) such that Dc is drawn in [Oc + min_offset, Oc + max_offset]
            
            P40_range=(0.75, 0.9),          # (min, max) for uniform draw of probability of 40ft container
            PExport_range=(0.05, 0.7),      # (min, max) for uniform draw of probability of export 

            C_range_reduced=(33, 200),      # (min, max) number of containers when reduced=True
            N_range_reduced=(4, 8),         # (min, max) number of terminals when reduced=True
            gamma=100,  

            # NEW: make Greedy compatible with MILP travel-time parameters
            Travel_time_long_range=(5, 5),   # (min, max) travel time between dryport and sea terminals in hours
            Travel_time_short_range=(1, 1),  # (min, max) travel time between sea terminals in hours
        ):
        """
        Initialize the greedy optimizer (MILP_Algo-compatible instance structure).
        """
        # Parameters
        self.seed = seed
        self.reduced = reduced
        self.Qk = qk              # TEU
        self.H_b = h_b            # euros
        self.H_t_40 = h_t_40      # euros
        self.H_t_20 = h_t_20      # euros
        self.Handling_time = handling_time  # hours

        # Parameter ranges (hours / probabilities)
        self.C_range = C_range
        self.N_range = N_range
        self.Oc_range = Oc_range
        self.Oc_offset_range = Oc_offset_range
        self.P40_range = P40_range
        self.PExport_range = PExport_range
        
        self.C_range_reduced = C_range_reduced
        self.N_range_reduced = N_range_reduced

        self.Gamma = gamma

        # NEW: travel time ranges (mirror MILP_Algo)
        self.travel_time_long_range = Travel_time_long_range
        self.travel_time_short_range = Travel_time_short_range

        # --- Instance data (MILP-style) ---
        self.C = 0              # Number of containers
        self.N = 0              # Number of terminals
        self.C_list = []        # list of container indices
        self.N_list = []        # list of terminal indices

        self.K_list = []        # list of vehicle indices (barges + 1 truck)
        self.K_b = []           # barge indices
        self.K_t = None         # truck index (for compatibility)

        self.E = []             # list of export containers
        self.I = []             # list of import containers
        self.W_c = []           # container size in TEU (1 or 2)
        self.R_c = []           # release times [hours]
        self.O_c = []           # opening times [hours]
        self.D_c = []           # closing times [hours]
        self.H_T = []           # trucking cost per container [euros]
        self.Z_cj = []          # assignment of container c to terminal j

        self.C_dict = {}        # Container information dictionary (same keys as MILP_Algo)

        # Travel times
        self.T_matrix = []      # numpy matrix (N x N)
        self.T_ij_matrix = []   # list-of-lists (N x N), MILP-style

        # Greedy-specific structures
        self.Barges = []        # Barge capacities (mutable copy of Qk)
        self.C_ordered = []     # Ordered containers
        self.master_route = []  # Master route (for greedy heuristic)
        
        # Results storage
        self.f_ck_init = None   # container-to-barge assignment matrix
        self.route_list = []
        self.barge_departure_delay = []
        self.trucked_containers = {}
        self.total_cost = 0     # Total cost of the solution
        self.truck_cost = 0     # Total trucking cost
        self.barge_cost = 0     # Total barge cost
        self.xijk = None        # Barge routing matrix
        
        # Generate instance automatically (MILP_Algo-compatible generation)
        self.generate_instance()

    def generate_instance(self):
        """
        Generate complete problem instance in a MILP_Algo-compatible way:
        - same container logic as MILP_Algo.generate_instance()
        - same travel-time logic as MILP_Algo.generate_travel_times()
        - plus Greedy-specific master route and container ordering.
        """
        self.generate_container_info()   # MILP-style container generation

        # Define vehicles like in MILP_Algo (barges + 1 truck)
        num_barges = len(self.Qk)
        self.K_list = list(range(num_barges + 1))
        self.K_b = self.K_list[:-1]
        self.K_t = self.K_list[-1]

        self.generate_travel_times()     # MILP-style travel times
        self.generate_master_route()     # Greedy-specific
        self.generate_ordered_containers()
        self.Barges = self.Qk.copy()     # Set barge capacities

    def generate_container_info(self):
        """
        Generate container information using the SAME logic as MILP_Algo.generate_instance
        for the instance part (C, N, E, I, W_c, R_c, O_c, D_c, H_T, Z_cj, C_dict).
        """
        rng = random.Random(self.seed)

        # Choose ranges based on reduced flag
        if self.reduced:
            C_min, C_max = self.C_range_reduced
            N_min, N_max = self.N_range_reduced
        else:
            C_min, C_max = self.C_range
            N_min, N_max = self.N_range

        num_C = rng.randint(C_min, C_max)
        num_N = rng.randint(N_min, N_max)

        self.C = num_C
        self.N = num_N
        self.C_list = list(range(num_C))
        self.N_list = list(range(num_N))

        # Unpack ranges
        Oc_min_hr, Oc_max_hr = self.Oc_range
        Oc_off_min_hr, Oc_off_max_hr = self.Oc_offset_range
        P40_min, P40_max = self.P40_range
        PExp_min, PExp_max = self.PExport_range

        # Reset storage
        self.E = []
        self.I = []
        self.W_c = []
        self.R_c = []
        self.O_c = []
        self.D_c = []
        self.H_T = []
        self.Z_cj = [[0 for _ in self.N_list] for _ in self.C_list]
        self.C_dict = {}

        # Container loop (identical structure to MILP_Algo)
        for c in self.C_list:
            # Opening time in hours
            Oc_hr = rng.randint(Oc_min_hr, Oc_max_hr)

            # Closing time in hours (after opening)
            Dc_hr = rng.randint(Oc_hr + Oc_off_min_hr, Oc_hr + Oc_off_max_hr)

            # Probabilities drawn per-container
            P_40 = rng.uniform(P40_min, P40_max)
            P_Export = rng.uniform(PExp_min, PExp_max)

            # Size in TEU: 1 -> 20ft, 2 -> 40ft
            if rng.random() < P_40:
                W_teu = 2  # 40ft
            else:
                W_teu = 1  # 20ft

            # Import / export, terminal, release time
            if rng.random() < P_Export:
                In_or_Out = 2  # Export
                Rc_hr = rng.randint(0, 24)  # release window for exports
                Terminal = rng.randint(1, self.N - 1)
                self.E.append(c)
            else:
                In_or_Out = 1  # Import
                Rc_hr = 0
                Terminal = rng.randint(1, self.N - 1)
                self.I.append(c)

            # Trucking cost per container
            if W_teu == 1:
                truck_cost = self.H_t_20
            else:
                truck_cost = self.H_t_40

            # Store MILP-style arrays
            self.W_c.append(W_teu)
            self.R_c.append(Rc_hr)
            self.O_c.append(Oc_hr)
            self.D_c.append(Dc_hr)
            self.H_T.append(truck_cost)
            self.Z_cj[c][Terminal] = 1

            # Container dict compatible with both Greedy and MILP
            self.C_dict[c] = {
                "Rc": Rc_hr,          # ready time in hours
                "Dc": Dc_hr,          # closing time in hours
                "Oc": Oc_hr,          # opening time in hours
                "Wc": W_teu,          # TEU (1 or 2)
                "In_or_Out": In_or_Out,
                "Terminal": Terminal,
            }

    def generate_travel_times(self, aleatorio=False):
        """
        Build travel time matrix T_ij [hours] using the SAME logic as MILP_Algo.generate_travel_times:

        i = origin terminal
        j = destination terminal
        - long range: travel time between dryport (0) and any other node
        - short range: travel time between non-0 nodes
        - if aleatorio=True, generate random symmetric times in hours (except diagonal)
        """
        num_nodes = self.N
        rng = random.Random(self.seed + 1)

        long_time_min, long_time_max = self.travel_time_long_range
        short_time_min, short_time_max = self.travel_time_short_range

        T_ij_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    T_ij = 0
                elif i == 0 or j == 0:
                    # From/to dryport (0)
                    T_ij = rng.randint(long_time_min, long_time_max)
                else:
                    T_ij = rng.randint(short_time_min, short_time_max)
                T_ij_matrix[i][j] = T_ij    # T_ij in hours

        if aleatorio:
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    dist = rng.randint(1, 10)
                    T_ij_matrix[i][j] = dist
                    T_ij_matrix[j][i] = dist
            for i in range(num_nodes):
                T_ij_matrix[i][i] = 0

        # Store both list-of-lists (MILP style) and numpy matrix (Greedy usage)
        self.T_ij_matrix = T_ij_matrix
        self.T_matrix = np.array(T_ij_matrix, dtype=int)
    def generate_master_route(self):
        """Generate master route using TSP approximation"""
        n = len(self.T_matrix)
        G = nx.complete_graph(n)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    G[i][j]['weight'] = self.T_matrix[i][j]

        # Find approximate TSP cycle (returns to start)
        self.master_route = nx.approximation.traveling_salesman_problem(G, cycle=True, weight='weight')
    def generate_ordered_containers(self):
        """Generate ordered list of containers based on master route"""
        self.C_ordered = []
        
        for i in self.master_route[1:-1]:  # Skip first and last (depot)
            for j in range(self.C):
                if self.C_dict[j]["Terminal"] == i:
                    self.C_ordered.append(j)
    
    
    def get_route(self, L_current):
        """
        Generate route for current barge load
        
        Parameters:
        -----------
        L_current : dict
            Current containers assigned to this barge
            
        Returns:
        --------
        list : Route as list of terminal indices
        """
        route = [0]  # start at terminal 0 (main terminal)
        current_terminal = 0  # start at terminal 0 (main terminal)

        for c in L_current.values():
            if c["Terminal"] == current_terminal:
                continue
            else:
                route.append(c["Terminal"])
                current_terminal = c["Terminal"]
        
        return route
    
    def get_timing(self, route, L_current, delay):
        """
        Calculate timing for barge route
        
        Parameters:
        -----------
        route : list
            Route as list of terminal indices
        L_current : dict
            Current containers assigned to this barge
        delay : float
            Additional delay in hours
            
        Returns:
        --------
        tuple : (departure_times, arrival_times)
        """
        departure_time = 0
        current_max = 0
        dry_port_handling_time = 0

        for c in L_current.values():
            if c["In_or_Out"] == 2:  # Export
                dry_port_handling_time += self.Handling_time
                if c["Rc"] > current_max:
                    current_max = c["Rc"]
                    
        departure_time = current_max + dry_port_handling_time

        D_terminal = [departure_time]  # time of departure from each terminal
        O_terminal = [0]  # time of arrival at each terminal

        current_terminal = 0  # start at terminal 0 (dry port)
        term_departure_time = departure_time  # start time at departure time
        term_arrival_time = 0  # start time at 0

        for terminal in route[1:]:
            travel_time = self.T_matrix[current_terminal][terminal]
            handling_time_total = self.Handling_time * sum(1 for c in L_current.values() if c["Terminal"] == terminal)

            term_departure_time += travel_time  # add travel time to next terminal
            term_departure_time += handling_time_total  # add handling time at terminal

            term_arrival_time += travel_time + D_terminal[-1]  # arrival time is the same as departure time after handling

            D_terminal.append(term_departure_time)  # append the time of arrival at the terminal
            O_terminal.append(term_arrival_time)  # append the time of arrival at the terminal

            current_terminal = terminal
        
        D_terminal[-1] += delay  # add delay to the last terminal's departure time

        return D_terminal, O_terminal
    
    def check_for_cap(self, route, L_current, barge_idx):
        """
        Check if current assignment respects barge capacity
        
        Parameters:
        -----------
        route : list
            Route as list of terminal indices
        L_current : dict
            Current containers assigned to this barge
        barge_idx : int
            Index of the barge
            
        Returns:
        --------
        bool : True if capacity is respected, False otherwise
        """
        teu_used = []

        for i in range(len(route)):
            terminal = route[i]
            sum_teu = 0
            
            for c in L_current.values():
                if c["Terminal"] == terminal and terminal > 0 and c["In_or_Out"] == 1:  # import containers are loaded on the barge
                    sum_teu += c["Wc"]
                elif c["Terminal"] == terminal and terminal > 0 and c["In_or_Out"] == 2:  # export containers are unloaded from the barge
                    sum_teu -= c["Wc"]
                elif i == 0 and c["In_or_Out"] == 2:  # export at depot
                    sum_teu += c["Wc"]

            teu_used.append(sum_teu)

        cap = [True if teu <= self.Barges[barge_idx] else False for teu in teu_used]

        return all(cap)
    
    def delay_window(self, container, D_terminal, route, terminal):
        """
        Calculate delay needed for time window constraint
        
        Parameters:
        -----------
        container : dict
            Container information
        D_terminal : list
            Departure times at each terminal
        route : list
            Route as list of terminal indices
        terminal : int
            Terminal index
            
        Returns:
        --------
        float : Required delay in hours
        """
        Oc = container["Oc"]
        D_term = D_terminal[route.index(terminal)]

        if Oc - D_term > 0:
            delay = (Oc - D_term) + self.Handling_time
            return delay
        else:
            return 0
    
    def solve_greedy(self):
        """
        Solve the container allocation problem using greedy algorithm
        
        Returns:
        --------
        dict : Solution results including costs and assignments
        """
        self.f_ck_init = np.zeros((self.C, len(self.Barges)))  # matrix for container to barge assignment
        
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
                    cont: self.C_dict[cont]
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
                    violations = []
                    for cont in L_current.values():
                        t = cont["Terminal"]
                        if not (
                            O_term[route.index(t)] <= cont["Oc"] <= D_term[route.index(t)]
                            or O_term[route.index(t)] <= cont["Dc"] <= D_term[route.index(t)]
                        ):
                            violations.append(cont)

                    if not violations:
                        # everyone fits under this `delay`
                        success = True
                        break

                    # if we still have our one "shift" left, compute the shift
                    if attempt == 0:
                        # largest extra wait needed
                        needed = [
                            self.delay_window(v, O_term, route, v["Terminal"])
                            for v in violations
                        ]
                        delay += max(needed)  # accumulate shift
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

            # move on to next barge (with its own fresh departure_delay)
            self.route_list.append(route)
            self.barge_departure_delay.append(departure_delay)
            barge_idx += 1
            departure_delay = 0

        # Calculate trucked containers
        index_to_be_trucked = np.where(np.sum(self.f_ck_init, axis=1) == 0)[0].tolist()
        self.trucked_containers = {i: self.C_dict[i] for i in index_to_be_trucked}

        # Calculate trucking cost
        self.truck_cost = 0
        for i in self.trucked_containers:
            if self.C_dict[i]["Wc"] == 1:  # 20ft container
                self.truck_cost += self.H_t_20
            else:  # 40ft container
                self.truck_cost += self.H_t_40

        # Calculate barge routing matrix
        self.xijk = np.zeros((len(self.Barges), self.N, self.N))  # xijk[k][i][j] = 1 if barge k goes from terminal i to terminal j

        for barge_idx, route in enumerate(self.route_list):
            for i in range(len(route) - 1):
                self.xijk[barge_idx][route[i]][route[i + 1]] = 1
                self.xijk[barge_idx][route[i + 1]][route[i]] = 1

        # Calculate barge cost
        self.barge_cost = self.calculate_objective()
        
        # Calculate total cost
        self.total_cost = self.barge_cost + self.truck_cost
        
        return {
            'total_cost': self.total_cost,
            'barge_cost': self.barge_cost,
            'truck_cost': self.truck_cost,
            'f_ck_init': self.f_ck_init,
            'route_list': self.route_list,
            'trucked_containers': self.trucked_containers,
            'xijk': self.xijk
        }
    
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
            for j in range(self.N):
                if j == 0:
                    continue
                cost += self.xijk[k][0][j] * self.H_b[k]

            # 2) travel‐time term: sum over all i,j of T[i][j]*x[i][j][k]
            for i in range(self.N):
                for j in range(self.N):
                    cost += self.T_matrix[i][j] * self.xijk[k][i][j]

            # 3) extra‐stop term: sum over j≠0, i≠j of x[j][i][k]
            for j in range(self.N):
                if j == 0:
                    continue
                for i in range(self.N):
                    if i == j:
                        continue
                    cost += self.xijk[k][i][j] * self.Handling_time

        return cost
    
    def print_results(self):
        """Print detailed results of the optimization"""
        print(f"Total cost: {self.total_cost:>10.0f} Euros")
        print(f"Barge cost: {self.barge_cost:>10.0f} Euros             ({self.barge_cost / self.total_cost * 100:>5.1f}% )")
        print(f"Truck cost: {self.truck_cost:>10.0f} Euros             ({self.truck_cost / self.total_cost * 100:>5.1f}% )")
        print(f"Containers: {self.C:>10d}")
        print(f"Terminals: {self.N:>10d}")
        print(f"Trucked containers: {len(self.trucked_containers):>10d}           ({len(self.trucked_containers) / self.C * 100:>5.1f}% )")
        
        # Print barge utilization
        for k, route in enumerate(self.route_list):
            if len(route) > 1:  # Only print if barge is used
                containers_on_barge = sum(1 for c in range(self.C) if self.f_ck_init[c][k] == 1)
                teu_on_barge = sum(self.C_dict[c]["Wc"] for c in range(self.C) if self.f_ck_init[c][k] == 1)
                print(
                        f"Barge {k:>3d}: "
                        f"{containers_on_barge:>4d} containers, "
                        f"{teu_on_barge:>4d}/{self.Barges[k]:<4d} TEU"
                        )

    @property
    def T_ij_list(self):
        """Alias for T_matrix for backward compatibility"""
        return self.T_matrix

    def print_pre_run_results(self):
        """
        Prints a concise summary of the generated instance before optimization starts.
        Provides an overview of model size: nodes, containers, barges, and key stats.
        (Adapted from MILP_Algo, but independent of any solver.)
        """
        print("\nPre-Run Instance Summary")
        print("========================")

        # Basic counts
        num_nodes = len(self.N_list)
        num_containers = len(self.C_list)
        num_barges = len(self.Qk)
        num_vehicles = len(self.K_list) if self.K_list else len(self.Qk) + 1

        # Container type counts
        num_imports = len(self.I)
        num_exports = len(self.E)

        # TEU totals
        total_teu = sum(self.W_c) if self.W_c else 0
        import_teu = sum(self.W_c[c] for c in self.I) if self.I else 0
        export_teu = sum(self.W_c[c] for c in self.E) if self.E else 0

        # Time-window statistics (in hours)
        earliest_open = min(self.O_c) if self.O_c else None
        latest_close = max(self.D_c) if self.D_c else None

        print(f"Nodes (terminals):         {num_nodes}")
        print(f"Containers:                {num_containers}  "
              f"(Imports: {num_imports}, Exports: {num_exports})")
        print(f"Barges available:          {num_barges}")
        print(f"Total vehicles (incl. truck): {num_vehicles}")
        print(f"Total TEU:                 {total_teu}  "
              f"(Import TEU: {import_teu}, Export TEU: {export_teu})")

        if earliest_open is not None and latest_close is not None:
            print(f"Container time windows:     earliest open = {earliest_open:.1f} h, "
                  f"latest close = {latest_close:.1f} h")
        else:
            print("Container time windows:     (no containers)")

        print(f"Handling time per container: {self.Handling_time:.2f} hours")
        print(f"Opening time range (param): {self.Oc_range} hours")
        print(f"Opening offset range:        {self.Oc_offset_range} hours")

        print("Summary complete.\n--\n")
    def print_node_table(self):
        """
        Prints a table summarizing how many imports/exports are associated to each node.
        Compatible with MILP_Algo.print_node_table() structure.
        """
        N = self.N_list
        Z_cj = self.Z_cj
        E = self.E
        I = self.I

        node_data = []
        for node in N:
            import_count = sum(1 for c in I if Z_cj[c][node] == 1)
            export_count = sum(1 for c in E if Z_cj[c][node] == 1)
            node_data.append({
                "Node ID": f"Node {node}",
                "Import Containers": import_count,
                "Export Containers": export_count,
            })

        df = pd.DataFrame(node_data)
        print("\n\nNode Table")
        print("==========")
        print(tabulate(df, headers="keys", tablefmt="grid"))
    def print_distance_table(self):
        """
        Prints a table summarizing the travel times (distances) between node pairs.
        Only unique pairs (i < j) are shown, since T[i][j] = T[j][i].
        Distances are in hours. Compatible with MILP_Algo.print_distance_table().
        """
        if not self.T_ij_matrix:
            print("Travel time matrix is empty. Did you call generate_travel_times()?") 
            return

        N = self.N_list
        T = self.T_ij_matrix

        distance_data = []
        for i in N:
            for j in N:
                if j <= i:
                    continue  # avoid self-pairs and duplicates
                distance_data.append({
                    "From": f"Node {i}",
                    "To": f"Node {j}",
                    "Distance [hours]": T[i][j],
                })

        df = pd.DataFrame(distance_data)
        print("\n\nDistance Table (Unique Node Pairs)")
        print("==================================")
        print(tabulate(df, headers="keys", tablefmt="grid"))



    def print_results_2(self):
        """
        Print detailed results in the same style as GreedyAlgo.print_results,
        with additional global metrics and a compact barge summary table.

        Computes:
        - Cost decomposition (truck, barge fixed, travel, terminal penalty)
        - Container and TEU breakdown (truck vs barge)
        - Barge usage and capacity utilization
        - Per-barge utilization (containers and TEU)
        """

        C    = self.C_list
        N    = self.N_list
        K_b  = self.K_b
        K_t  = self.K_t

        # Data
        H_T  = self.H_T
        H_b  = self.H_b
        T    = self.T_ij_matrix
        Gamma = self.Gamma
        Qk   = self.Qk
        W_c  = self.W_c

        # Variables
        f_ck  = self.f_ck
        x_ijk = self.x_ijk

        # -------------------------
        # Cost decomposition
        # -------------------------
        truck_cost = sum(H_T[c] * f_ck[c, K_t].X for c in C)

        barge_fixed_cost = sum(
            H_b[k] * x_ijk[0, j, k].X
            for k in K_b for j in N if j != 0
        )

        travel_cost = sum(
            T[i][j] * x_ijk[i, j, k].X
            for k in K_b for i in N for j in N if i != j
        )

        terminal_penalty_cost = sum(
            Gamma * x_ijk[i, j, k].X
            for k in K_b for i in N for j in N if j != 0
        )

        barge_cost = barge_fixed_cost + travel_cost + terminal_penalty_cost
        total_cost = truck_cost + barge_cost  # should match m.objVal

        # Guard against division by zero
        total_cost_safe = total_cost if total_cost != 0 else 1.0
        barge_cost_safe = barge_cost if barge_cost != 0 else 1.0

        # -------------------------
        # Container / TEU breakdown
        # -------------------------
        total_containers = self.C
        total_terminals  = self.N

        trucked_containers = sum(1 for c in C if f_ck[c, K_t].X > 0.5)
        barge_containers   = total_containers - trucked_containers

        total_teu = sum(W_c[c] for c in C)
        truck_teu = sum(W_c[c] for c in C if f_ck[c, K_t].X > 0.5)
        barge_teu = total_teu - truck_teu

        trucked_ratio = trucked_containers / total_containers * 100 if total_containers > 0 else 0.0

        # -------------------------
        # Barge usage / capacity
        # -------------------------
        barge_rows = []
        barge_teu_check = 0
        used_barges = []

        for k in K_b:
            containers_on_barge = sum(1 for c in C if f_ck[c, k].X > 0.5)
            if containers_on_barge == 0:
                continue  # barge unused

            used_barges.append(k)
            teu_on_barge = sum(W_c[c] for c in C if f_ck[c, k].X > 0.5)
            barge_teu_check += teu_on_barge

            utilization = teu_on_barge / Qk[k] * 100 if Qk[k] > 0 else 0.0

            barge_rows.append({
                "Barge": k,
                "Containers": containers_on_barge,
                "TEU used": teu_on_barge,
                "Capacity (TEU)": Qk[k],
                "Utilization [%]": f"{utilization:5.1f}",
            })

        # (barge_teu_check should equal barge_teu if everything is consistent)
        num_barges_total = len(K_b)
        num_barges_used  = len(used_barges)

        total_barge_capacity = sum(Qk[k] for k in K_b)
        overall_capacity_util = (
            barge_teu / total_barge_capacity * 100
            if total_barge_capacity > 0 else 0.0
        )

        # -------------------------
        # Print summary
        # -------------------------
        print("\n\nResults Table")
        print("=============")
        print(f"Total containers:               {total_containers:>10d}")
        print(f"Total terminals:                {total_terminals:>10d}")
        print(f"Total TEU demand:               {total_teu:>10d}")
        print()
        print(f"Available barges (K_b):         {num_barges_total:>10d}  {self.K_b}")
        print(f"Used barges:                    {num_barges_used:>10d}")
        print(f"Total barge capacity [TEU]:     {total_barge_capacity:>10d}")
        print(f"TEU on barges (model):          {barge_teu:>10d}")
        print(f"Overall barge utilization:      {overall_capacity_util:>9.1f} %")
        print()
        print(f"Trucked containers:             {trucked_containers:>10d}  "
              f"({trucked_ratio:>5.1f} % of all containers)")
        print(f"TEU on trucks:                  {truck_teu:>10d}")
        print(f"TEU on barges (via rows):       {barge_teu_check:>10d}")
        print()
        print(f"Total cost:                     {total_cost:>10.0f} Euros")
        print(f"  ├─ Truck cost:                {truck_cost:>10.0f} Euros  "
              f"({truck_cost / total_cost_safe * 100:>5.1f} % of total)")
        print(f"  └─ Barge cost:                {barge_cost:>10.0f} Euros  "
              f"({barge_cost / total_cost_safe * 100:>5.1f} % of total)")
        print()
        print(f"     Barge fixed cost:          {barge_fixed_cost:>10.0f} Euros  "
              f"({barge_fixed_cost / barge_cost_safe * 100:>5.1f} % of barge)")
        print(f"     Travel term:               {travel_cost:>10.0f} Euros  "
              f"({travel_cost / barge_cost_safe * 100:>5.1f} % of barge)")
        print(f"     Terminal penalty term:     {terminal_penalty_cost:>10.0f} Euros  "
              f"({terminal_penalty_cost / barge_cost_safe * 100:>5.1f} % of barge)")

        # -------------------------
        # Per-barge utilization summary (no table)
        # -------------------------
        print("\nBarge Utilization Summary")

        if barge_rows:
            for row in barge_rows:
                print(
                    f"Barge {row['Barge']:>2d}:  "
                    f"{row['Containers']:>3d} containers   |  "
                    f"TEU used: {row['TEU used']:>4d}/{row['Capacity (TEU)']:<4d}   "
                    f"({row['Utilization [%]']:>5s} %)"
                )
        else:
            print("No barges were used in the optimal solution.")



    def plot_barge_solution_map(self):
        """
        Plots a 2D map of terminals and barge routes for the final solution.

        - Nodes (terminals) are positioned using MDS on the travel-time matrix,
          then normalized to a unit square for a clear layout.
        - Node size roughly reflects total number of containers at that terminal.
        - Node annotations:
            * node index inside the circle
            * 'I:x  E:y' under the node (imports/exports).
        - Barge paths:
            * one color per barge
            * line width proportional to TEU flow on that arc (imports + exports)
            * lateral offset per barge so overlapping arcs are clearly distinguishable.

        The figure is saved as 'Figures/barge_solution_map.png'.
        """



        # Data / sets
        T_ij_matrix = np.array(self.T_ij_matrix)
        N = self.N_list
        K_b = self.K_b
        C = self.C_list
        E = self.E
        I = self.I
        Z_cj = self.Z_cj
        W_c = self.W_c
        x_ijk = self.x_ijk
        y_ijk = self.y_ijk
        z_ijk = self.z_ijk

        # -----------------------------
        # Compute 2D node positions via MDS
        # -----------------------------
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
        node_positions_raw = mds.fit_transform(T_ij_matrix)  # shape (N, 2)

        # Normalize positions to a [0, 1] x [0, 1] box for a clean aspect
        xs_raw = node_positions_raw[:, 0]
        ys_raw = node_positions_raw[:, 1]

        x_min0, x_max0 = xs_raw.min(), xs_raw.max()
        y_min0, y_max0 = ys_raw.min(), ys_raw.max()
        x_rng = x_max0 - x_min0 if x_max0 > x_min0 else 1.0
        y_rng = y_max0 - y_min0 if y_max0 > y_min0 else 1.0

        xs = (xs_raw - x_min0) / x_rng
        ys = (ys_raw - y_min0) / y_rng

        node_positions = np.column_stack([xs, ys])

        # -----------------------------
        # Node-level container stats
        # -----------------------------
        import_counts = {j: 0 for j in N}
        export_counts = {j: 0 for j in N}
        total_counts = {j: 0 for j in N}

        for c in C:
            for j in N:
                if Z_cj[c][j] == 1:
                    if c in I:
                        import_counts[j] += 1
                    elif c in E:
                        export_counts[j] += 1
                    total_counts[j] += 1
                    break

        max_containers = max(total_counts.values()) if total_counts else 1
        base_size = 120.0    # base marker size
        size_scale = 280.0   # extra size per relative container load

        node_sizes = {}
        for j in N:
            if max_containers > 0:
                node_sizes[j] = base_size + size_scale * (total_counts[j] / max_containers)
            else:
                node_sizes[j] = base_size

        # -----------------------------
        # Arc-level TEU flows (for line widths)
        # -----------------------------
        arc_flows = {}  # (i, j, k) -> TEU on arc i->j for barge k
        max_flow = 0.0
        for k in K_b:
            for i in N:
                for j in N:
                    if i == j:
                        continue
                    if x_ijk[i, j, k].X > 0.5:
                        flow_teu = y_ijk[i, j, k].X + z_ijk[i, j, k].X
                        arc_flows[(i, j, k)] = flow_teu
                        if flow_teu > max_flow:
                            max_flow = flow_teu

        if max_flow == 0:
            max_flow = 1.0  # avoids division by zero; all arcs get minimal width

        # Line width mapping
        min_lw = 0.6
        max_lw = 2.5

        def flow_to_lw(flow):
            return min_lw + (max_lw - min_lw) * (flow / max_flow)

        # -----------------------------
        # Plotting
        # -----------------------------
        fig, ax = plt.subplots(figsize=(8, 6))

        # For label offset we work in normalized coordinates
        y_span = ys.max() - ys.min() if ys.size > 0 else 1.0
        label_offset = 0.05 * y_span

        # Plot nodes
        for j in N:
            x, y = node_positions[j]
            size = node_sizes[j]

            # Draw node (dryport visually distinct)
            if j == 0:
                facecolor = "white"
                edgecolor = "black"
                linewidth = 1.5
            else:
                facecolor = "white"
                edgecolor = "black"
                linewidth = 1.0

            ax.scatter(
                x, y,
                s=size,
                edgecolor=edgecolor,
                facecolor=facecolor,
                linewidth=linewidth,
                zorder=3,
            )

            # Node id in the centre
            ax.text(
                x, y,
                f"{j}",
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                zorder=4,
            )

            # I/E counts below the node
            imp = import_counts[j]
            exp = export_counts[j]
            ax.text(
                x, y - label_offset,
                f"I:{imp}  E:{exp}",
                ha="center", va="top",
                fontsize=7,
                zorder=4,
            )

        # -----------------------------
        # Plot barge paths
        # -----------------------------
        cmap = plt.get_cmap("tab10")
        barge_index_map = {k: idx for idx, k in enumerate(K_b)}
        offset_scale = 0.04  # lateral separation in normalized coordinates

        for k in K_b:
            color = cmap(barge_index_map[k] % 10)
            idx_k = barge_index_map[k]

            for i in N:
                for j in N:
                    if i == j:
                        continue
                    if x_ijk[i, j, k].X <= 0.5:
                        continue

                    x1, y1 = node_positions[i]
                    x2, y2 = node_positions[j]

                    dx = x2 - x1
                    dy = y2 - y1
                    length = (dx**2 + dy**2) ** 0.5
                    if length == 0:
                        continue

                    # Unit perpendicular vector
                    nx = -dy / length
                    ny = dx / length

                    # Lateral offset per barge
                    offset = offset_scale * (idx_k - (len(K_b) - 1) / 2.0)
                    x1_off = x1 + nx * offset
                    y1_off = y1 + ny * offset
                    x2_off = x2 + nx * offset
                    y2_off = y2 + ny * offset

                    flow_teu = arc_flows.get((i, j, k), 0.0)
                    lw = flow_to_lw(flow_teu)

                    # Shrink arrow slightly so it doesn't cover node centres
                    shrink = 0.10
                    x_start = x1_off + shrink * dx
                    y_start = y1_off + shrink * dy
                    x_end = x2_off - shrink * dx
                    y_end = y2_off - shrink * dy

                    arrow = FancyArrowPatch(
                        (x_start, y_start), (x_end, y_end),
                        arrowstyle="-|>",
                        linewidth=lw,
                        color=color,
                        alpha=0.75,
                        mutation_scale=8,
                        zorder=2,
                    )
                    ax.add_patch(arrow)

        # -----------------------------
        # Final styling
        # -----------------------------
        # Legend for barges
        for k in K_b:
            color = cmap(barge_index_map[k] % 10)
            ax.plot([], [], color=color, label=f"Barge {k}", linewidth=2)

        ax.set_title("Barge Routes and Terminal Containers (Final Solution)")
        ax.set_xlabel("MDS dimension 1 (normalized)")
        ax.set_ylabel("MDS dimension 2 (normalized)")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
        ax.set_aspect("equal", adjustable="box")

        # Nice padding around [0,1]x[0,1]
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
            framealpha=0.8,
            fontsize=8,
        )

        plt.tight_layout()
        plt.savefig("Figures/barge_solution_map.png", dpi=300)
        # plt.show()



# Create global instance for backward compatibility
_global_optimizer = GreedyOptimizer()

# Global variables for backward compatibility
C_dict = _global_optimizer.C_dict
C = _global_optimizer.C
N = _global_optimizer.N
T_ij_list = _global_optimizer.T_matrix
Barges = _global_optimizer.Barges
C_ordered = _global_optimizer.C_ordered
Qk = _global_optimizer.Qk
H_b = _global_optimizer.H_b
H_t_40 = _global_optimizer.H_t_40
H_t_20 = _global_optimizer.H_t_20
Handling_time = _global_optimizer.Handling_time

# Functions for backward compatibility
def container_info(seed, reduced):
    """Backward compatibility function"""
    optimizer = GreedyOptimizer(seed=seed, reduced=reduced)
    return optimizer.C_dict, optimizer.C, optimizer.N

def get_route(L_current):
    """Backward compatibility function"""
    return _global_optimizer.get_route(L_current)

def get_timing(route, T_ij_list, handling_time, L_current, delay):
    """Backward compatibility function"""
    return _global_optimizer.get_timing(route, L_current, delay)

def check_for_cap(route, L_current, barges, idx):
    """Backward compatibility function"""
    # Create temporary optimizer with given barges
    temp_optimizer = GreedyOptimizer()
    temp_optimizer.Barges = barges
    return temp_optimizer.check_for_cap(route, L_current, idx)

def delay_window(container, D_terminal, route, terminal, handling_time):
    """Backward compatibility function"""
    return _global_optimizer.delay_window(container, D_terminal, route, terminal)

# Run the algorithm and print results if this file is executed directly
if __name__ == "__main__":
    # optimizer = GreedyOptimizer()
    # results = optimizer.solve_greedy()
    # optimizer.print_results()


    greedy = GreedyOptimizer()

    greedy.print_pre_run_results()
    greedy.print_node_table()
    greedy.print_distance_table()

    res = greedy.solve_greedy()
    greedy.print_results()
    # greedy.print_results_2()
    # greedy.plot_barge_solution_map()