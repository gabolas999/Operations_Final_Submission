#!/usr/bin/env python3
"""
Mixed-Integer Linear Programming (MILP) model for container allocation optimization.
Implemented as the MILP_Algo class, structurally aligned with GreedyAlgo.
Requires Gurobi (gurobipy) with a valid license.

"""

import os
from datetime import datetime
import random
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib.patches import FancyArrowPatch
import math

class MILP_Algo:
    def __init__(
            self,
            qk=[  # Barge capacities in TEU
                20,        # Barge 0
                20,         # Barge 1
                20,         # Barge 2
                20,         # Barge 3
                20,         # Barge 4
                20,         # Barge 5
                20,         # Barge 6
            ],
            h_b=[  # Barge fixed costs in euros
                1400,      # Barge 0
                1500,      # Barge 1
                1600,      # Barge 2
                1700,      # Barge 3
                3000,      # Barge 4
                5000,      # Barge 5
                5000,      # Barge 6
            ],
            seed=100,
            reduced=False,
            h_t_40=200_0000,                     # 40ft container trucking cost in euros
            h_t_20=140_0000,                     # 20ft container trucking cost in euros
            handling_time=1/6,              # Container handling time in hours
            C_range=(100, 200),             # (min, max) number of containers when reduced=False
            N_range=(5, 5),                 # (min, max) number of terminals when reduced=False

            Oc_range=(24, 196),             # (min, max) opening time in hours
            Oc_offset_range=(50, 220),      # (min_offset, max_offset) such that
                                            # Dc is drawn in [Oc + min_offset, Oc + max_offset]

            Travel_time_long_range=(2, 3),   # (min, max) travel time between dryport and sea terminals in hours
            # Travel_time_short_range=(1, 1),  # (min, max) travel time between sea terminals in hours

            P40_range=(0.75, 0.9),          # (min, max) probability of 40ft container
            PExport_range=(0.05, 0.7),      # (min, max) probability of export
            C_range_reduced=(60, 100),      # (min, max) containers when reduced=True
            N_range_reduced=(4, 4),         # (min, max) terminals when reduced=True
            gamma=100,                      # penalty per sea terminal visit [euros]
            big_m=1_000_000                 # big-M
    ):
        """
        Initialize the MILP optimizer.

        Parameters mirror GreedyAlgo so that both can be constructed in the same way.
        Time-related ranges and all internal time variables are in hours.
        """
        # Parameters
        self.seed = seed
        self.reduced = reduced
        self.Qk = qk          # barge capacities in TEU
        self.H_b = h_b        # barge fixed costs in euros
        self.H_t_40 = h_t_40  # euros per 40ft container
        self.H_t_20 = h_t_20  # euros per 20ft container
        self.Handling_time = handling_time  # hours per container


        # Ranges (hours / probabilities)
        self.C_range = C_range
        self.N_range = N_range
        self.Oc_range = Oc_range
        self.Oc_offset_range = Oc_offset_range
        self.P40_range = P40_range
        self.PExport_range = PExport_range

        self.travel_time_long_range = Travel_time_long_range
        # self.travel_time_short_range = Travel_time_short_range

        self.C_range_reduced = C_range_reduced
        self.N_range_reduced = N_range_reduced

        # MILP-specific constants
        self.Gamma = gamma
        self.M = big_m

        # Instance data (populated by generate_instance)
        self.C = 0                  # number of containers
        self.N = 0                  # number of terminals
        self.C_list = []            # list of container indices
        self.N_list = []            # list of terminal indices
        self.K_list = []            # list of vehicle indices
        self.K_b = []               # barge indices
        self.K_t = None             # truck index

        self.E = []                 # list of export containers
        self.I = []                 # list of import containers
        self.W_c = []               # container size in TEU (1 or 2)
        self.R_c = []               # release times [hours]
        self.O_c = []               # opening times [hours]
        self.D_c = []               # closing times [hours]
        self.H_T = []               # trucking cost per container [euros]
        self.Z_cj = []              # assignment of container c to terminal j
        self.C_dict = {}            # same structure as in GreedyAlgo for compatibility

        self.T_ij_matrix = []       # travel time matrix [hours]

        # Gurobi model and variables
        self.model = None
        self.f_ck = None
        self.x_ijk = None
        self.p_jk = None
        self.d_jk = None
        self.y_ijk = None
        self.z_ijk = None
        self.t_jk = None

        # Generate instance automatically (data only, no optimization yet)
        self.generate_instance()
        self.generate_travel_times()

    # -----------------------
    # Instance generation
    # -----------------------

    def generate_instance(self):
        """
        Generate the problem instance using the same parameter logic
        as GreedyAlgo.generate_container_info.
        and extended with MILP-specific sets (E, I, Z_cj, etc.).
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

        # Vehicles: len(Qk) barges + 1 truck
        num_barges = len(self.Qk)
        self.K_list = list(range(num_barges + 1))
        self.K_b = self.K_list[:-1]
        self.K_t = self.K_list[-1]

        # Prepare containers
        Oc_minim_hr, Oc_max_hr = self.Oc_range
        Oc_off_min_hr, Oc_off_max_hr = self.Oc_offset_range
        P40_min, P40_max = self.P40_range
        PExp_min, PExp_max = self.PExport_range

        self.E = []
        self.I = []
        self.W_c = []
        self.R_c = []
        self.O_c = []
        self.D_c = []
        self.H_T = []
        self.Z_cj = [[0 for _ in self.N_list] for _ in self.C_list]
        self.C_dict = {}

        for c in self.C_list:

            # Opening time in hours
            Oc_hr = rng.randint(Oc_minim_hr, Oc_max_hr)

            # Closing time in hours
            Dc_hr = rng.randint(Oc_hr + Oc_off_min_hr, Oc_hr + Oc_off_max_hr)


            # Probabilities
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

            # Keep times in hours for MILP model
            Rc = Rc_hr
            Oc = Oc_hr
            Dc = Dc_hr

            # Trucking cost per container
            if W_teu == 1:
                truck_cost = self.H_t_20
            else:
                truck_cost = self.H_t_40

            # Store
            self.W_c.append(W_teu)
            self.R_c.append(Rc)
            self.O_c.append(Oc)
            self.D_c.append(Dc)
            self.H_T.append(truck_cost)
            self.Z_cj[c][Terminal] = 1

            self.C_dict[c] = {
                "Rc": Rc_hr,          # ready time in hours (for consistency with GreedyAlgo)
                "Dc": Dc_hr,          # closing time in hours
                "Oc": Oc_hr,          # opening time in hours
                "Wc": W_teu,          # TEU (1 or 2)
                "In_or_Out": In_or_Out,
                "Terminal": Terminal,
            }

    # -----------------------
    # Travel time matrix
    # -----------------------
    def generate_travel_times(self):
        """
        Generates:
        - node_xy: list of (x, y) coordinates for each terminal.
        - T_ij_matrix: Euclidean travel times between nodes.
        
        Rules:
        - Dryport (node 0) fixed at (0, 0).
        - Each other node gets:
            * A radial distance sampled from travel_time_long_range.
            * A random angle in [0, 2π).
            * Coordinates computed from polar -> Cartesian.
        """

        rng = random.Random(self.seed + 123)
        num_nodes = self.N

        long_min, long_max = self.travel_time_long_range

        node_xy = []
        node_xy.append((0.0, 0.0))      # Node 0 fixed

        # ---------------------------
        # Assign coordinates to nodes
        # ---------------------------
        for j in range(1, num_nodes):
            # Sample travel time distance from dryport
            r = rng.randint(long_min, long_max)

            # Random angle
            # theta = rng.uniform(0, 2 * math.pi)
            theta = rng.uniform(math.pi*3/4, math.pi*5/4)

            # Cartesian coordinates
            x = r * math.cos(theta)
            y = r * math.sin(theta)

            node_xy.append((x, y))

        self.node_xy = node_xy  # store coordinates

        # ---------------------------
        # Build travel time matrix
        # ---------------------------
        T = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]

        for i in range(num_nodes):
            xi, yi = node_xy[i]
            for j in range(num_nodes):
                if i == j:
                    T[i][j] = 0.0
                    continue
                xj, yj = node_xy[j]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                T[i][j] = int(dist)

        self.T_ij_matrix = T

    # -----------------------
    # Model setup
    # -----------------------

    def setup_model(self):
        """Create Gurobi model and decision variables."""
        self.model = Model("BargeScheduling")


        # --------------- Gurobi configuration ---------------
        os.makedirs("Logs", exist_ok=True)

        # Log file with timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model.Params.LogFile = f"Logs/gurobi_log_{timestamp}.log"

        # Stopping criterion: 1% relative MIP gap
        self.model.Params.MIPGap = 0.01

        # Re-enable Gurobi's own console log
        self.model.Params.OutputFlag = 1

        # (Optional) If you want fewer log lines, uncomment this:
        # self.model.Params.DisplayInterval = 5  # print progress every 5 seconds
        # ----------------------------------------------------




        C = self.C_list
        N = self.N_list
        K = self.K_list
        K_b = self.K_b

        # f_ck = 1 if container c is allocated to vehicle k (barges or truck)
        self.f_ck = self.model.addVars(C, K, vtype=GRB.BINARY, name="f_ck")

        # x_ijk = 1 if barge k sails from terminal i to j
        self.x_ijk = self.model.addVars(N, N, K_b, vtype=GRB.BINARY, name="x_ijk")

        # p_jk: import quantity loaded by barge k at terminal j
        # d_jk: export quantity unloaded by barge k at terminal j
        self.p_jk = self.model.addVars(N, K_b, vtype=GRB.INTEGER, lb=0, name="p_jk")
        self.d_jk = self.model.addVars(N, K_b, vtype=GRB.INTEGER, lb=0, name="d_jk")

        # y_ijk: import quantity carried by barge k from i to j
        # z_ijk: export quantity carried by barge k from i to j
        self.y_ijk = self.model.addVars(N, N, K_b, vtype=GRB.INTEGER, lb=0, name="y_ijk")
        self.z_ijk = self.model.addVars(N, N, K_b, vtype=GRB.INTEGER, lb=0, name="z_ijk")

        # t_jk: time barge k is at terminal j
        self.t_jk = self.model.addVars(N, K_b, vtype=GRB.CONTINUOUS, name="t_jk")

        self.model.update()

    # -----------------------
    # Objective function
    # -----------------------

    def set_objective(self):
        """Set the MILP objective function."""
        m = self.model

        C = self.C_list
        N = self.N_list
        K_b = self.K_b
        K_t = self.K_t

        H_T = self.H_T
        HkB = self.H_b
        Gamma = self.Gamma
        T = self.T_ij_matrix

        f_ck = self.f_ck
        x_ijk = self.x_ijk

        objective = (
            # 1. Trucking cost
            quicksum(f_ck[c, K_t] * H_T[c] for c in C)
            +
            # 2. Barge fixed cost (if barge leaves dryport)
            quicksum(x_ijk[0, j, k] * HkB[k] for k in K_b for j in N if j != 0)
            +
            # 3. Travel cost between terminals
            quicksum(T[i][j] * x_ijk[i, j, k]
                     for k in K_b for i in N for j in N if i != j)
            +
            # 4. Sea terminal visit penalty (j != 0 are sea terminals)
            quicksum(x_ijk[i, j, k] * Gamma
                     for k in K_b for i in N for j in N if j != 0)
        )

        m.setObjective(objective, GRB.MINIMIZE)

    # -----------------------
    # Constraints
    # -----------------------

    def add_constraints(
            self,
            limit_total_trucked_containers=True,
            include_time_constraints=True
    ):
        """Add all MILP constraints to the model."""
        m = self.model

        C = self.C_list
        N = self.N_list
        K = self.K_list
        K_b = self.K_b
        K_t = self.K_t

        E = self.E
        I = self.I
        W_c = self.W_c
        R_c = self.R_c
        O_c = self.O_c
        D_c = self.D_c
        Z_cj = self.Z_cj
        Qk = self.Qk
        L = self.Handling_time  # in hours
        M = self.M
        T = self.T_ij_matrix

        f_ck = self.f_ck
        x_ijk = self.x_ijk
        p_jk = self.p_jk
        d_jk = self.d_jk
        y_ijk = self.y_ijk
        z_ijk = self.z_ijk
        t_jk = self.t_jk

        # Optional: limit total trucked containers
        if limit_total_trucked_containers:
            m.addConstr(
                quicksum(f_ck[c, K_t] for c in C) <= len(C) - 10,
                name="Trucked_Limit"
            )

        # 1. Each container is assigned to exactly one vehicle
        for c in C:
            m.addConstr(
                quicksum(f_ck[c, k] for k in K) == 1,
                name=f"Container_Assignment_{c}"
            )

        # 2. Flow conservation for barges at each node
        for i in N:
            for k in K_b:
                m.addConstr(
                    quicksum(x_ijk[i, j, k] for j in N if j != i)
                    - quicksum(x_ijk[j, i, k] for j in N if j != i) == 0,
                    name=f"Flow_Conservation_{i}_{k}"
                )

        # 3. Each barge leaves dryport (0) at most once
        for k in K_b:
            m.addConstr(
                quicksum(x_ijk[0, j, k] for j in N if j != 0) <= 1,
                name=f"Departures_{k}"
            )

        # 4. No self-loops (i -> i)
        for i in N:
            for k in K_b:
                m.addConstr(x_ijk[i, i, k] == 0, name=f"No_Self_Loop_{i}_{k}")

        # 5. Import quantity at terminal j for barge k
        for k in K_b:
            for j in N[1:]:
                m.addConstr(
                    p_jk[j, k] == quicksum(W_c[c] * Z_cj[c][j] * f_ck[c, k]
                                           for c in I),
                    name=f"Import_Quantity_{j}_{k}"
                )

        # 6. Export quantity at terminal j for barge k
        for k in K_b:
            for j in N[1:]:
                m.addConstr(
                    d_jk[j, k] == quicksum(W_c[c] * Z_cj[c][j] * f_ck[c, k]
                                           for c in E),
                    name=f"Export_Quantity_{j}_{k}"
                )

        # 7. Flow balance for import quantities at terminal j for barge k
        for j in N[1:]:
            for k in K_b:
                m.addConstr(
                    quicksum(y_ijk[j, i, k] for i in N if i != j)
                    - quicksum(y_ijk[i, j, k] for i in N if i != j)
                    == p_jk[j, k],
                    name=f"Import_Balance_{j}_{k}"
                )

        # 8. Flow balance for export quantities at terminal j for barge k
        for j in N[1:]:
            for k in K_b:
                m.addConstr(
                    quicksum(z_ijk[i, j, k] for i in N if i != j)
                    - quicksum(z_ijk[j, i, k] for i in N if i != j)
                    == d_jk[j, k],
                    name=f"Export_Balance_{j}_{k}"
                )

        # 9. Barge trip capacity constraint
        for i in N:
            for j in N:
                if i == j:
                    continue
                for k in K_b:
                    m.addConstr(
                        y_ijk[i, j, k] + z_ijk[i, j, k] <= Qk[k] * x_ijk[i, j, k],
                        name=f"Flow_Capacity_{i}_{j}_{k}"
                    )

        # 10. Export containers: departure time at dryport >= release time
        for c in E:
            for k in K_b:
                m.addConstr(
                    t_jk[0, k] >= R_c[c] * f_ck[c, k],
                    name=f"Vehicle_Departure_{c}_{k}"
                )

        # Time constraints
        if include_time_constraints:
            # 11 & 12. Time propagation along arcs with handling time at arrival
            for i in N:
                for j in N[1:]:
                    if i == j:
                        continue
                    for k in K_b:
                        handling_term = quicksum(L * Z_cj[c][j] * f_ck[c, k] for c in C)

                        m.addConstr(
                            t_jk[j, k] >= t_jk[i, k] + handling_term + T[i][j]
                            - (1 - x_ijk[i, j, k]) * M,
                            name=f"Time_LB_{i}_{j}_{k}"
                        )
                        m.addConstr(
                            t_jk[j, k] <= t_jk[i, k] + handling_term + T[i][j]
                            + (1 - x_ijk[i, j, k]) * M,
                            name=f"Time_UB_{i}_{j}_{k}"
                        )

            # 13. Export container service cannot start before opening time
            for c in C:
                for j in N[1:]:
                    for k in K_b:
                        m.addConstr(
                            t_jk[j, k] >= O_c[c] * Z_cj[c][j] - (1 - f_ck[c, k]) * M,
                            name=f"Export_Time_{c}_{j}_{k}"
                        )

            # 14. All containers must be served before closing time
            for c in C:
                for j in N[1:]:
                    for k in K_b:
                        m.addConstr(
                            t_jk[j, k] * Z_cj[c][j] <= D_c[c] + (1 - f_ck[c, k]) * M,
                            name=f"Demand_Fulfillment_{c}_{j}_{k}"
                        )

    # -----------------------
    # Solve
    # -----------------------
    def solve(self):
        """Run the optimization (standard Gurobi log) and print a short completion banner."""
        if self.model is None:
            raise RuntimeError(
                "Model not set up. Call setup_model(), set_objective(), add_constraints() first."
            )

        # Optional: briefly explain Gurobi's MIP progress table columns
        print("\nGurobi MIP progress table columns:")
        print("  Expl Unexpl : explored / unexplored nodes in the search tree")
        print("  Obj         : objective of the current node's LP relaxation")
        print("  Depth       : depth of the current node in the search tree")
        print("  IntInf      : integer infeasibilities at the current node")
        print("  Incumbent   : best feasible (integer) objective found so far")
        print("  BestBd      : best bound on the optimal objective (minimization)")
        print("  Gap         : relative gap between Incumbent and BestBd")
        print("  It/Node     : average LP iterations per processed node")
        print("  Time        : elapsed wall-clock time (seconds)\n")

        # Standard Gurobi optimization with built-in log
        self.model.optimize()



        print("\n#######################################################################################################################################################")
        print("#######################################################################################################################################################")
        print("################################################################## Optimization Complete ##############################################################")
        print("#######################################################################################################################################################")
        print("#######################################################################################################################################################")


    # -----------------------
    # Result printing helpers
    # -----------------------
    def print_pre_run_results(self):
        """
        Prints a concise summary of the generated instance before optimization starts.
        Provides an overview of model size: nodes, containers, barges, and key stats.
        """
        print("\nPre-Run Instance Summary")
        print("========================")

        # Basic counts
        num_nodes = len(self.N_list)
        num_containers = len(self.C_list)
        num_barges = len(self.K_b)
        num_vehicles = len(self.K_list)

        # Container type counts
        num_imports = len(self.I)
        num_exports = len(self.E)

        # TEU totals
        total_teu = sum(self.W_c)
        import_teu = sum(self.W_c[c] for c in self.I)
        export_teu = sum(self.W_c[c] for c in self.E)

        # Time-window statistics (in hours)
        earliest_open = min(self.O_c)  if self.O_c else None
        latest_close = max(self.D_c)   if self.D_c else None

        print(f"Nodes (terminals):         {num_nodes}")
        print(f"Containers:                {num_containers}  "
              f"(Imports: {num_imports}, Exports: {num_exports})")
        print(f"Barges available:          {num_barges}")
        print(f"Total vehicles (incl. truck): {num_vehicles}")
        print(f"Total TEU:                 {total_teu}  "
              f"(Import TEU: {import_teu}, Export TEU: {export_teu})")

        print(f"Container time windows:     earliest open = {earliest_open:.1f} h, latest close = {latest_close:.1f} h")
        print(f"Handling time per container: {self.Handling_time:.2f} hours")
        print(f"Opening time range (param): {self.Oc_range} hours")
        print(f"Opening offset range:        {self.Oc_offset_range} hours")

        print("Summary complete.\n--\n\n\n\n")

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
        m = self.model
        if m is None or m.status != GRB.OPTIMAL:
            print("No optimal solution found. Status:", m.status if m is not None else "No model")
            return

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

    def print_node_table(self):
        """
        Prints a table summarizing how many imports/exports are associated to each node.
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
        Distances are in hours.
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
        print(    "==================================")
        print(tabulate(df, headers="keys", tablefmt="grid"))

    def print_barge_table(self):
        """
        Prints a table summarizing barge routes and capacity utilization per arc.
        """
        print(f"\n")
        m = self.model
        if m is None or m.status != GRB.OPTIMAL:
            print("No optimal solution available to print barge table.")
            return

        K_b = self.K_b
        Qk = self.Qk
        N = self.N_list
        x_ijk = self.x_ijk
        y_ijk = self.y_ijk
        z_ijk = self.z_ijk
        print("\n\n================")
        print(   f"All Barge tables")
        print(    "================")
        for k in K_b:
            total_capacity = Qk[k]
            routes = []

            for i in N:
                for j in N:
                    if i != j and x_ijk[i, j, k].X > 0.5:
                        capacity_used = y_ijk[i, j, k].X + z_ijk[i, j, k].X
                        utilization_percent = (
                            capacity_used / total_capacity * 100 if total_capacity > 0 else 0
                        )
                        routes.append({
                            "Route": f"Node {i} -> Node {j}",
                            "Capacity Used (TEU)": capacity_used,
                            "Capacity (TEU)": total_capacity,
                            "Utilization (%)": f"{utilization_percent:.0f}",
                        })

            if routes:
                df = pd.DataFrame(routes)
                print(f"\nBarge {k} Route & Capacity Usage")
                print("==============================")
                print(tabulate(df, headers="keys", tablefmt="grid"))

    def print_container_table(self):
        """
        Prints a table summarizing container properties and assigned barge/truck.
        Exports: Node 0 -> Node j
        Imports: Node j -> Node 0
        """
        m = self.model
        if m is None or m.status != GRB.OPTIMAL:
            print("No optimal solution available to print container table.")
            return

        C = self.C_list
        E = self.E
        I = self.I
        W_c = self.W_c
        R_c = self.R_c
        O_c = self.O_c
        D_c = self.D_c
        Z_cj = self.Z_cj
        K = self.K_list
        K_t = self.K_t
        f_ck = self.f_ck

        container_data = []
        for c in C:
            container_type = "Export" if c in E else "Import"
            node = Z_cj[c].index(1)  # associated terminal

            if container_type == "Export":
                origin = "Node 0"
                destination = f"Node {node}"
                sort_node = node
            else:
                origin = f"Node {node}"
                destination = "Node 0"
                sort_node = node

            assigned_vehicle = next((k for k in K if f_ck[c, k].X > 0.5), None)
            assigned_label = (
                f"Truck {assigned_vehicle}"
                if assigned_vehicle == K_t
                else f"Barge {assigned_vehicle}"
            )

            # TEU size -> ft just for display
            size_ft = 20 if W_c[c] == 1 else 40

            container_data.append({
                "Container ID": c,
                "Size (ft)": size_ft,
                "Type": container_type,
                "Origin": origin,
                "Destination": destination,
                "Release Time [hours]": R_c[c],
                "Opening Time [hours]": O_c[c],
                "Closing Time [hours]": D_c[c],
                "Assigned Vehicle": assigned_label,
                "Sort Node": sort_node,
                "Sort Type": 0 if container_type == "Export" else 1
            })

        df = pd.DataFrame(container_data)
        df = df.sort_values(by=["Sort Node", "Sort Type"]).drop(columns=["Sort Node", "Sort Type"])

        print("\n\nContainer Table (Grouped by Node and Type)")
        print("==========================================")
        print(tabulate(df, headers="keys", tablefmt="grid"))


    def plot_barge_displacements(self):
        """
        Plots displacements of barges based on x_ijk using MDS layout.
        (Trucked containers are not plotted.)
        """
        m = self.model
        if m is None or m.status != GRB.OPTIMAL:
            print("No optimal solution available for plotting.")
            return

        T_ij_matrix = np.array(self.T_ij_matrix)
        N = self.N_list
        K_b = self.K_b
        x_ijk = self.x_ijk

        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
        node_positions = mds.fit_transform(T_ij_matrix)

        plt.figure(figsize=(8, 6))

        # Plot nodes
        for idx, (x, y) in enumerate(node_positions):
            plt.scatter(x, y, s=100)
            plt.text(x + 0.01, y + 0.01, f"{idx}", fontsize=9)

        # Plot barge paths
        for k in K_b:
            for i in N:
                for j in N:
                    if i != j and x_ijk[i, j, k].X > 0.5:
                        x1, y1 = node_positions[i]
                        x2, y2 = node_positions[j]

                        offset = 0.01 * k
                        x1 += offset
                        y1 += offset
                        x2 += offset
                        y2 += offset

                        plt.plot([x1, x2], [y1, y2], linewidth=1.5, alpha=0.7, label=f"Barge {k}")
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        plt.text(mid_x, mid_y, f"B{k}", fontsize=8)

        plt.title("Barge Displacements Between Terminals")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.legend(loc="best")
        plt.savefig(f"Figures/barge_displacements.png")
        # plt.show()

    def plot_barge_solution_map(self):
        """
        Customized 2D barge route map using true coordinates self.node_xy.
        Figure aesthetics:
        - figsize (12, 8)
        - dark blue background (#003d80)
        - node 0 = large solid black square
        - other nodes = thin black hollow circles
        - barge paths = thin black lines with unique dash patterns
        - legend on the right, only for barges actually used
        - no title, no container count labels
        """

        m = self.model
        if m is None or m.status != GRB.OPTIMAL:
            print("No optimal solution available for plotting.")
            return

        # --------------------------
        # Extract data
        # --------------------------
        node_xy = self.node_xy
        N = self.N_list
        K_b = self.K_b
        x_ijk = self.x_ijk
        y_ijk = self.y_ijk
        z_ijk = self.z_ijk

        # --------------------------
        # Determine which barges are used
        # --------------------------
        used_barges = []
        for k in K_b:
            used = False
            for i in N:
                for j in N:
                    if i != j and x_ijk[i, j, k].X > 0.5:
                        used = True
                        break
                if used:
                    break
            if used:
                used_barges.append(k)

        # Define dash patterns (cycled)
        dash_patterns = [
            (None, None),          # solid
            (5, 5),                # dashed
            (2, 3),                # dotted
            (8, 4, 2, 4),          # dash-dot
            (10, 3),               # long dash
            (3, 2, 3, 2, 8, 2),    # complex pattern
        ]
        # Enough patterns for many barges
        dash_patterns = dash_patterns * 10

        # --------------------------
        # Plotting setup
        # --------------------------
        fig, ax = plt.subplots(figsize=(14, 8))

        # background color

        ax.set_facecolor("#003d80")

        # --------------------------
        # Plot nodes
        # --------------------------
        for j in N:
            x, y = node_xy[j]

            if j == 0:
                # large solid black square
                ax.scatter(
                    x, y,
                    s=600,
                    marker="s",
                    facecolor="none",
                    edgecolor="white",
                    linewidth=1.2,
                    zorder=4,
                )
            else:
                # thin hollow black circle
                ax.scatter(
                    x, y,
                    s=300,
                    facecolor="none",
                    edgecolor="white",
                    linewidth=1.2,
                    zorder=3,
                )

            ax.text(
                x, y,
                f"{j}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
                zorder=5,
            )

        # --------------------------
        # Plot barge paths
        # --------------------------
        offset_scale = 0.03
        barge_index_map = {k: idx for idx, k in enumerate(used_barges)}

        for k_idx, k in enumerate(used_barges):
            dash = dash_patterns[k_idx]
            idx_k = barge_index_map[k]

            for i in N:
                for j in N:
                    if i == j or x_ijk[i, j, k].X <= 0.5:
                        continue

                    x1, y1 = node_xy[i]
                    x2, y2 = node_xy[j]
                    dx, dy = x2 - x1, y2 - y1
                    length = (dx**2 + dy**2) ** 0.5
                    if length == 0:
                        continue

                    # perpendicular offset
                    nx, ny = -dy / length, dx / length
                    offset = offset_scale * (idx_k - (len(used_barges) - 1) / 2)
                    x1o, y1o = x1 + nx * offset, y1 + ny * offset
                    x2o, y2o = x2 + nx * offset, y2 + ny * offset

                    # shrink endpoints slightly
                    shrink = 0.02
                    xs = x1o + shrink * dx
                    ys_ = y1o + shrink * dy
                    xe = x2o - shrink * dx
                    ye = y2o - shrink * dy

                    ax.plot(
                        [xs, xe], [ys_, ye],
                        color="black",
                        linewidth=1.2,
                        linestyle='-' if dash == (None, None) else (0, dash),
                        zorder=2,
                    )

        # --------------------------
        # Final styling
        # --------------------------
        ax.set_xlabel("X coordinate (hours)", color="black", fontsize=10)
        ax.set_ylabel("Y coordinate (hours)", color="black", fontsize=10)

        # No title
        ax.set_title("")

        # White grid on blue background
        ax.grid(True, linestyle=":", linewidth=0.5, color="white", alpha=0.4)

        ax.set_aspect("equal", adjustable="datalim")

        # Ticks in white
        ax.tick_params(colors="black")

        # --------------------------
        # Legend (only used barges)
        # --------------------------
        legend_lines = []
        legend_labels = []

        for k_idx, k in enumerate(used_barges):
            dash = dash_patterns[k_idx]
            line = ax.plot(
                [], [],
                color="black",
                linewidth=1.5,
                linestyle='-' if dash == (None, None) else (0, dash),
            )[0]
            legend_lines.append(line)
            legend_labels.append(f"Barge {k}")

        ax.legend(
            legend_lines,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            framealpha=0.9,
            fontsize=9,
            facecolor="white",
            edgecolor="black",
        )

        plt.tight_layout()
        plt.savefig("Figures/barge_solution_map.png", dpi=300)

    def plot_barge_solution_map_report(self):
        """
        Minimalistic 2D barge route map using true coordinates self.node_xy.

        Aesthetics:
        - figsize (12, 8)
        - white background
        - node 0 = solid black square
        - other nodes = hollow black circles
        - barge paths = thin colored lines with subtle dash variations
        - legend on the right, only for barges actually used
        - no title, no container/count labels
        - axes and ticks removed for a clean, schematic look
        """

        m = self.model
        if m is None or m.status != GRB.OPTIMAL:
            print("No optimal solution available for plotting.")
            return

        # --------------------------
        # Extract data
        # --------------------------
        node_xy = self.node_xy
        N = self.N_list
        K_b = self.K_b
        x_ijk = self.x_ijk
        y_ijk = self.y_ijk
        z_ijk = self.z_ijk

        # --------------------------
        # Determine which barges are used
        # --------------------------
        used_barges = []
        for k in K_b:
            used = False
            for i in N:
                for j in N:
                    if i != j and x_ijk[i, j, k].X > 0.5:
                        used = True
                        break
                if used:
                    break
            if used:
                used_barges.append(k)

        if not used_barges:
            print("No barges used in the solution; nothing to plot.")
            return

        # Dash patterns (cycled)
        dash_patterns = [
            (None, None),          # solid
            (4, 3),                # dashed
            (1, 2),                # dotted
            (6, 3, 1, 3),          # dash-dot
            (8, 4),                # long dash
            (2, 2, 6, 2),          # mixed
        ]
        dash_patterns = dash_patterns * 10  # ensure enough

        # Color palette (soft)
        cmap = plt.get_cmap("tab10")

        # --------------------------
        # Plotting setup
        # --------------------------
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # --------------------------
        # Plot nodes
        # --------------------------
        for j in N:
            x, y = node_xy[j]

            if j == 0:
                # solid black square for dryport
                ax.scatter(
                    x, y,
                    s=220,
                    marker="s",
                    facecolor="black",
                    edgecolor="black",
                    linewidth=1.2,
                    zorder=4,
                )
            else:
                # hollow circle for sea terminals
                ax.scatter(
                    x, y,
                    s=160,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=3,
                )

            # node index label
            ax.text(
                x, y,
                f"{j}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="medium",
                color="black",
                zorder=5,
            )

        # --------------------------
        # Plot barge paths
        # --------------------------
        offset_scale = 0.03
        barge_index_map = {k: idx for idx, k in enumerate(used_barges)}

        for k_idx, k in enumerate(used_barges):
            dash = dash_patterns[k_idx]
            color = cmap(k_idx % 10)
            idx_k = barge_index_map[k]

            for i in N:
                for j in N:
                    if i == j or x_ijk[i, j, k].X <= 0.5:
                        continue

                    x1, y1 = node_xy[i]
                    x2, y2 = node_xy[j]
                    dx, dy = x2 - x1, y2 - y1
                    length = (dx**2 + dy**2) ** 0.5
                    if length == 0:
                        continue

                    # perpendicular offset
                    nx, ny = -dy / length, dx / length
                    offset = offset_scale * (idx_k - (len(used_barges) - 1) / 2)
                    x1o, y1o = x1 + nx * offset, y1 + ny * offset
                    x2o, y2o = x2 + nx * offset, y2 + ny * offset

                    # slightly shrink to avoid covering node centers
                    shrink = 0.04
                    xs = x1o + shrink * dx
                    ys_ = y1o + shrink * dy
                    xe = x2o - shrink * dx
                    ye = y2o - shrink * dy

                    ax.plot(
                        [xs, xe], [ys_, ye],
                        color=color,
                        linewidth=1.4,
                        linestyle='-' if dash == (None, None) else (0, dash),
                        zorder=2,
                    )

        # --------------------------
        # Minimal styling
        # --------------------------
        # no title, very light axes
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

        # remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # subtle spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#aaaaaa")

        ax.set_aspect("equal", adjustable="datalim")

        # --------------------------
        # Legend (only used barges)
        # --------------------------
        legend_lines = []
        legend_labels = []

        for k_idx, k in enumerate(used_barges):
            dash = dash_patterns[k_idx]
            color = cmap(k_idx % 10)
            line = ax.plot(
                [], [],
                color=color,
                linewidth=1.6,
                linestyle='-' if dash == (None, None) else (0, dash),
            )[0]
            legend_lines.append(line)
            legend_labels.append(f"Barge {k}")

        ax.legend(
            legend_lines,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=9,
        )

        plt.tight_layout()
        plt.savefig("Figures/barge_solution_map_report.png", dpi=300)

    def plot_barge_solution_map_report_2(self):
        """
        Minimalistic curved-edge barge route map.
        Each barge path is drawn as a quadratic Bezier curve:
        - Nodes: true coordinates self.node_xy
        - Barge arcs: curved, colored, no offset at endpoints
        - Figure aesthetics: simple, clean, minimal
        """

        def bezier_quad(P0, P1, P2, t):
            """Quadratic Bézier interpolation."""
            return (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2

        m = self.model
        if m is None or m.status != GRB.OPTIMAL:
            print("No optimal solution available for plotting.")
            return

        # --------------------------
        # Extract data
        # --------------------------
        node_xy = self.node_xy
        N = self.N_list
        K_b = self.K_b
        x_ijk = self.x_ijk

        # --------------------------
        # Determine which barges are used
        # --------------------------
        used_barges = []
        for k in K_b:
            if any(x_ijk[i, j, k].X > 0.5 for i in N for j in N if i != j):
                used_barges.append(k)

        if not used_barges:
            print("No barges used; nothing to plot.")
            return

        # Color palette and curvature magnitudes
        cmap = plt.get_cmap("tab10")
        curvature_values = [0.12, -0.12, 0.18, -0.18, 0.25, -0.25, 0.32, -0.32]

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # --------------------------
        # Draw nodes
        # --------------------------
        for j in N:
            x, y = node_xy[j]

            if j == 0:
                # Dryport = black solid square
                ax.scatter(
                    x, y,
                    s=260,
                    marker="s",
                    facecolor="black",
                    edgecolor="black",
                    zorder=4,
                )
            else:
                ax.scatter(
                    x, y,
                    s=160,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.2,
                    zorder=3,
                )

            ax.text(
                x, y,
                f"{j}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                zorder=5,
            )

        # --------------------------
        # Draw curved barge paths
        # --------------------------
        for idx, k in enumerate(used_barges):
            color = cmap(idx % 10)
            curvature = curvature_values[idx % len(curvature_values)]

            for i in N:
                for j in N:
                    if i == j or x_ijk[i, j, k].X <= 0.5:
                        continue

                    # Endpoints
                    x1, y1 = node_xy[i]
                    x2, y2 = node_xy[j]
                    P0 = np.array([x1, y1])
                    P2 = np.array([x2, y2])

                    # Direction vector and orthogonal normal
                    d = P2 - P0
                    L = np.linalg.norm(d)
                    if L == 0:
                        continue

                    n = np.array([-d[1], d[0]]) / L  # orthogonal unit vector

                    # Control point displaced by curvature
                    P1 = (P0 + P2) / 2 + curvature * L * n

                    # Generate Bezier curve points
                    ts = np.linspace(0, 1, 60)
                    curve = np.array([bezier_quad(P0, P1, P2, t) for t in ts])

                    ax.plot(
                        curve[:, 0], curve[:, 1],
                        color=color,
                        linewidth=1.5,
                        zorder=2,
                    )

        # --------------------------
        # Styling
        # --------------------------
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Legend for used barges
        legend_lines = []
        legend_labels = []
        for idx, k in enumerate(used_barges):
            color = cmap(idx % 10)
            dummy, = ax.plot([], [], color=color, linewidth=1.8)
            legend_lines.append(dummy)
            legend_labels.append(f"Barge {k}")

        ax.legend(
            legend_lines,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=9,
        )

        plt.tight_layout()
        plt.savefig("Figures/barge_solution_map_report_2.png", dpi=300)
    def plot_barge_solution_map_report_2(self):
        """
        Minimalistic curved-edge barge route map.
        - Nodes: true coordinates self.node_xy
        - All barge paths: black curves with different line patterns per barge
        - No title, no ticks, clean white background
        - Legend only for barges that are actually used
        """

        def bezier_quad(P0, P1, P2, t):
            """Quadratic Bézier interpolation."""
            return (1 - t)**2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2

        m = self.model
        if m is None or m.status != GRB.OPTIMAL:
            print("No optimal solution available for plotting.")
            return

        # --------------------------
        # Extract data
        # --------------------------
        node_xy = self.node_xy
        N = self.N_list
        K_b = self.K_b
        x_ijk = self.x_ijk

        # --------------------------
        # Determine which barges are used
        # --------------------------
        used_barges = []
        for k in K_b:
            if any(x_ijk[i, j, k].X > 0.5 for i in N for j in N if i != j):
                used_barges.append(k)

        if not used_barges:
            print("No barges used; nothing to plot.")
            return

        # Different line patterns for barges
        line_styles = [
            "-",            # solid
            "--",           # dashed
            ":",            # dotted
            "-.",           # dash-dot
            (0, (5, 2, 1, 2)),   # custom: long-short-short
            (0, (3, 3, 1, 3)),   # custom: dot-dash
            (0, (1, 2)),         # sparse dots
            (0, (8, 4)),         # long dash
        ]
        line_styles = line_styles * 10  # just in case there are many barges

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # --------------------------
        # Draw nodes
        # --------------------------
        for j in N:
            x, y = node_xy[j]

            if j == 0:
                # Dryport = solid black square
                ax.scatter(
                    x, y,
                    s=260,
                    marker="s",
                    facecolor="black",
                    edgecolor="black",
                    zorder=4,
                )
            else:
                # Other terminals = hollow circles
                ax.scatter(
                    x, y,
                    s=160,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=1.2,
                    zorder=3,
                )

            ax.text(
                x, y,
                f"{j}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                zorder=5,
            )

        # --------------------------
        # Draw curved barge paths
        # --------------------------
        for idx, k in enumerate(used_barges):
            linestyle = line_styles[idx % len(line_styles)]
            # alternate curvature sign/magnitude for barges
            curvature_values = [0.12, -0.12, 0.18, -0.18, 0.25, -0.25]
            curvature = curvature_values[idx % len(curvature_values)]

            for i in N:
                for j in N:
                    if i == j or x_ijk[i, j, k].X <= 0.5:
                        continue

                    # Endpoints
                    x1, y1 = node_xy[i]
                    x2, y2 = node_xy[j]
                    P0 = np.array([x1, y1])
                    P2 = np.array([x2, y2])

                    # Direction vector and orthogonal normal
                    d = P2 - P0
                    L = np.linalg.norm(d)
                    if L == 0:
                        continue

                    n = np.array([-d[1], d[0]]) / L  # orthogonal unit vector

                    # Control point displaced by curvature
                    P1 = (P0 + P2) / 2 + curvature * L * n

                    # Generate Bezier curve points
                    ts = np.linspace(0, 1, 60)
                    curve = np.array([bezier_quad(P0, P1, P2, t) for t in ts])

                    ax.plot(
                        curve[:, 0], curve[:, 1],
                        color="black",
                        linewidth=1.4,
                        linestyle=linestyle,
                        zorder=2,
                    )

        # --------------------------
        # Styling
        # --------------------------
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Legend for used barges (black lines, different patterns)
        legend_lines = []
        legend_labels = []
        for idx, k in enumerate(used_barges):
            linestyle = line_styles[idx % len(line_styles)]
            line, = ax.plot(
                [], [],
                color="black",
                linewidth=1.6,
                linestyle=linestyle,
            )
            legend_lines.append(line)
            legend_labels.append(f"Barge {k}")

        ax.legend(
            legend_lines,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=9,
        )

        plt.tight_layout()
        plt.savefig("Figures/barge_solution_map_report_2.png", dpi=300)
    def plot_barge_solution_map_report_3(self):
        """
        Curved-edge barge route map with segment-order alpha encoding.
        Improvements over _2:
        - Each barge's path segments fade in as the barge progresses:
              early arcs → low alpha
              late arcs → alpha=1
        - Minimalistic aesthetic
        """

        import numpy as np
        import matplotlib.pyplot as plt

        def bezier_quad(P0, P1, P2, t):
            """Quadratic Bézier interpolation."""
            return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2

        m = self.model
        if m is None or m.status != GRB.OPTIMAL:
            print("No optimal solution available for plotting.")
            return

        # --------------------------
        # Extract data
        # --------------------------
        node_xy = self.node_xy
        N = self.N_list
        K_b = self.K_b
        x_ijk = self.x_ijk

        # --------------------------
        # Determine which barges are used
        # --------------------------
        used_barges = []
        for k in K_b:
            if any(x_ijk[i, j, k].X > 0.5 for i in N for j in N if i != j):
                used_barges.append(k)

        if not used_barges:
            print("No barges used; nothing to plot.")
            return

        # --------------------------
        # Linestyles for barges
        # --------------------------
        line_styles = [
            "-",          # solid
            "--",         # dashed
            ":",          # dotted
            "-.",         # dash-dot
            (0, (1, 1, 2, 1)),    # dot-dash fine
            (0, (8, 4)),          # long dash
            (0, (1, 2, 1, 2, 1, 6)),    # custom pattern
            (0, (2, 6)),                # custom pattern
            (0, (6, 2, 1, 2, 1, 2)),    # custom pattern
        ]
        line_styles = line_styles * 10  # repeat if many barges

        # --------------------------
        # Prepare figure
        # --------------------------
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # --------------------------
        # Draw nodes
        # --------------------------
        for j in N:
            x, y = node_xy[j]

            if j == 0:
                # Dryport = solid square
                ax.scatter(
                    x, y, s=650, marker="s",
                    facecolor="white", edgecolor="black",
                    zorder=4,
                )
            else:
                ax.scatter(
                    x, y, s=400,
                    facecolor="white", edgecolor="black",
                    linewidth=1.2, zorder=3
                )

            ax.text(
                x, y, f"{j}",
                ha="center", va="center",
                fontsize=9, color="black", zorder=5
            )

        # --------------------------
        # Build barge segment order
        # --------------------------
        barge_arcs_ordered = {}  # k -> ordered list of (i,j)

        for k in used_barges:
            # Collect all used arcs for this barge
            arcs = [(i, j) for i in N for j in N if i != j and x_ijk[i, j, k].X > 0.5]

            # -----------------------------------------------------------
            # Try to reconstruct the path order
            # Assumes a route exists (no complex branching)
            # -----------------------------------------------------------
            successors = {i: [] for i in N}
            predecessors = {j: [] for j in N}

            for (i, j) in arcs:
                successors[i].append(j)
                predecessors[j].append(i)

            # A start node has no predecessors
            start_candidates = [i for i in N if successors[i] and len(predecessors[i]) == 0]

            if start_candidates:
                start = start_candidates[0]
            else:
                # fallback: pick a node appearing as a 'from' node
                start = arcs[0][0]

            ordered = []
            current = start
            visited = set()

            while True:
                next_nodes = [j for j in successors[current] if (current, j) in arcs]
                next_nodes = [j for j in next_nodes if (current, j) not in visited]

                if not next_nodes:
                    break

                j = next_nodes[0]
                ordered.append((current, j))
                visited.add((current, j))
                current = j

            # If something was missed, append remaining arcs arbitrarily
            remaining = [a for a in arcs if a not in ordered]
            ordered.extend(remaining)

            barge_arcs_ordered[k] = ordered

        # --------------------------
        # Draw curved barge paths
        # --------------------------
        multiplier = 2.2
        curvature_values = [0.12*multiplier, -0.12*multiplier, 0.18*multiplier, -0.18*multiplier, 0.25*multiplier, -0.25*multiplier]

        legend_lines = []
        legend_labels = []

        for idx, k in enumerate(used_barges):
            linestyle = line_styles[idx]
            curvature = curvature_values[idx % len(curvature_values)]

            arcs = barge_arcs_ordered[k]
            S = max(1, len(arcs))  # number of segments

            # alpha schedule from 0.25 → 1.0
            alphas = np.linspace(0.25, 1.0, S)

            for s, ((i, j), alpha) in enumerate(zip(arcs, alphas)):
                x1, y1 = node_xy[i]
                x2, y2 = node_xy[j]

                P0 = np.array([x1, y1])
                P2 = np.array([x2, y2])

                d = P2 - P0
                L = np.linalg.norm(d)
                if L == 0:
                    continue

                # perpendicular orthonormal vector
                n = np.array([-d[1], d[0]]) / L

                # Bezier control point for curvature
                P1 = (P0 + P2) / 2 + curvature * L * n

                ts = np.linspace(0, 1, 60)
                curve = np.array([bezier_quad(P0, P1, P2, t) for t in ts])

                ax.plot(
                    curve[:, 0], curve[:, 1],
                    color="black",
                    linewidth=2.5,
                    linestyle=linestyle,
                    alpha=float(alpha),
                    zorder=2,
                )

            # legend entry for this barge
            line, = ax.plot([], [], color="black", linewidth=2.5, linestyle=linestyle)
            legend_lines.append(line)
            legend_labels.append(f"Barge {k}")

        # --------------------------
        # Styling
        # --------------------------
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Legend
        # ax.legend(
        #     legend_lines, legend_labels,
        #     loc="center left",
        #     bbox_to_anchor=(1.02, 0.5),
        #     frameon=False,
        #     fontsize=9,
        # )
        # ax.legend(
        #     legend_lines, legend_labels,
        #     loc="upper right",
        #     frameon=True,
        #     fontsize=15,
        # )
        ax.legend(
        legend_lines, legend_labels,
        loc="upper right",
        frameon=True,
        fontsize=14,
        handlelength=5,      # default is 2 — increase for longer patterns
        handletextpad=0.8,   # spacing between line and text
    )


        plt.tight_layout()
        plt.savefig("Figures/barge_solution_map_report_3.png", dpi=400)







    # -----------------------
    # Convenience pipeline
    # -----------------------

    def run(self, with_plots=True):
        """
        Convenience method to run the full MILP pipeline:
        - setup_model
        - set_objective
        - add_constraints
        - solve
        - print node/container/barge tables
        - plot displacements (optional)
        """
        self.print_pre_run_results()   
        
        self.setup_model()
        self.set_objective()
        self.add_constraints()

        # Solve
        self.solve()


        if self.model.status == GRB.OPTIMAL:
            # self.print_results_old_format()
            # print("\n\n\n\n\n\n")
            self.print_results_2()
            self.print_node_table()
            self.print_distance_table()
            self.print_barge_table()
            self.print_container_table()
            if with_plots:
                self.plot_barge_displacements()
                # self.plot_barge_solution_map()
                # self.plot_barge_solution_map_report()
                # self.plot_barge_solution_map_report_2()
                self.plot_barge_solution_map_report_3()

# Optional quick test if you run MILP.py directly:
if __name__ == "__main__":
    print("\n\n\n\n\n\n\n\n\n\n\n")
    milp = MILP_Algo(reduced=True)   # e.g. smaller instances
    milp.run(with_plots=True)








# continuing with the development. 






























    # def plot_barge_solution_map(self):
    #     """
    #     Plots a 2D map using *true* node coordinates from self.node_xy.
    #     Figure size is always (12, 8).
    #     Arrow widths represent TEU flow.
    #     """

    #     m = self.model
    #     if m is None or m.status != GRB.OPTIMAL:
    #         print("No optimal solution available for plotting.")
    #         return

    #     # Data
    #     node_xy = self.node_xy      # dict: node_id -> (x, y)
    #     N = self.N_list             # list of node IDs
    #     K_b = self.K_b              # list of barge IDs
    #     C = self.C_list             # list of container IDs
    #     E = self.E                  # set of export container IDs
    #     I = self.I                  # set of import container IDs
    #     Z_cj = self.Z_cj            # container-to-node assignment matrix
    #     W_c = self.W_c              # container size (TEU)
    #     x_ijk = self.x_ijk          # barge route binary variables
    #     y_ijk = self.y_ijk          # barge route TEU flow variables (imports)
    #     z_ijk = self.z_ijk          # barge route TEU flow variables (exports)

    #     # Extract X,Y arrays for convenience
    #     xs = np.array([p[0] for p in node_xy])
    #     ys = np.array([p[1] for p in node_xy])

    #     # Node-level container stats
    #     import_counts = {j: 0 for j in N}
    #     export_counts = {j: 0 for j in N}
    #     total_counts = {j: 0 for j in N}

    #     for c in C:
    #         for j in N:
    #             if Z_cj[c][j] == 1:
    #                 if c in I:
    #                     import_counts[j] += 1
    #                 elif c in E:
    #                     export_counts[j] += 1
    #                 total_counts[j] += 1
    #                 break

    #     max_containers = max(total_counts.values()) if total_counts else 1
    #     base_size = 120.0
    #     size_scale = 280.0

    #     node_sizes = {
    #         j: base_size + size_scale * (total_counts[j] / max_containers)
    #         for j in N
    #     }

    #     # Compute TEU flows for arrow thickness
    #     arc_flows = {}
    #     max_flow = 0.0
    #     for k in K_b:
    #         for i in N:
    #             for j in N:
    #                 if i != j and x_ijk[i, j, k].X > 0.5:
    #                     flow = y_ijk[i, j, k].X + z_ijk[i, j, k].X
    #                     arc_flows[(i, j, k)] = flow
    #                     max_flow = max(max_flow, flow)

    #     if max_flow == 0:
    #         max_flow = 1.0

    #     def flow_to_lw(flow):
    #         return 0.6 + (2.5 - 0.6) * (flow / max_flow)

    #     # -----------------------------
    #     # Plotting
    #     # -----------------------------
    #     fig, ax = plt.subplots(figsize=(12, 8))

    #     # vertical offset for I/E labels
    #     y_span = ys.max() - ys.min()
    #     label_offset = 0.05 * (y_span if y_span > 0 else 1.0)

    #     # Plot nodes
    #     for j in N:
    #         x, y = node_xy[j]
    #         size = node_sizes[j]

    #         # dryport special style
    #         edgecolor = "black"
    #         linewidth = 1.5 if j == 0 else 1.0

    #         ax.scatter(
    #             x, y,
    #             s=size,
    #             edgecolor=edgecolor,
    #             facecolor="white",
    #             linewidth=linewidth,
    #             zorder=3,
    #         )

    #         ax.text(
    #             x, y,
    #             f"{j}",
    #             ha="center", va="center",
    #             fontsize=9, fontweight="bold",
    #             zorder=4,
    #         )

    #         ax.text(
    #             x, y - label_offset,
    #             f"I:{import_counts[j]}  E:{export_counts[j]}",
    #             ha="center", va="top",
    #             fontsize=7,
    #             zorder=4,
    #         )

    #     # Plot barge arcs
    #     cmap = plt.get_cmap("tab10")
    #     barge_index_map = {k: idx for idx, k in enumerate(K_b)}
    #     offset_scale = 0.04

    #     for k in K_b:
    #         color = cmap(barge_index_map[k] % 10)
    #         idx_k = barge_index_map[k]

    #         for i in N:
    #             for j in N:
    #                 if i == j or x_ijk[i, j, k].X <= 0.5:
    #                     continue

    #                 x1, y1 = node_xy[i]
    #                 x2, y2 = node_xy[j]

    #                 dx, dy = (x2 - x1), (y2 - y1)
    #                 length = (dx**2 + dy**2)**0.5
    #                 if length == 0:
    #                     continue

    #                 # perpendicular offset per barge
    #                 nx, ny = -dy / length, dx / length
    #                 offset = offset_scale * (idx_k - (len(K_b) - 1) / 2)
    #                 x1o, y1o = x1 + nx * offset, y1 + ny * offset
    #                 x2o, y2o = x2 + nx * offset, y2 + ny * offset

    #                 shrink = 0.10
    #                 xs = x1o + shrink * dx
    #                 ys_ = y1o + shrink * dy
    #                 xe = x2o - shrink * dx
    #                 ye = y2o - shrink * dy

    #                 flow = arc_flows.get((i, j, k), 0.0)
    #                 lw = flow_to_lw(flow)

    #                 arrow = FancyArrowPatch(
    #                     (xs, ys_), (xe, ye),
    #                     arrowstyle="-|>",
    #                     linewidth=lw,
    #                     color=color,
    #                     alpha=0.75,
    #                     mutation_scale=8,
    #                     zorder=2,
    #                 )
    #                 ax.add_patch(arrow)

    #     # Final layout
    #     ax.set_title("Barge Routes and Terminal Containers (True Coordinates)")
    #     ax.set_xlabel("X coordinate (hours)")
    #     ax.set_ylabel("Y coordinate (hours)")
    #     ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    #     ax.set_aspect("equal", adjustable="datalim")

    #     # Add legend
    #     for k in K_b:
    #         ax.plot([], [], color=cmap(barge_index_map[k] % 10),
    #                 label=f"Barge {k}", linewidth=2)

    #     ax.legend(loc="upper right", frameon=True, fontsize=8)

    #     plt.tight_layout()
    #     plt.savefig("Figures/barge_solution_map.png", dpi=300)







    # def plot_barge_solution_map(self):
    #     """
    #     Plots a 2D map of terminals and barge routes for the final solution.

    #     - Nodes (terminals) are positioned using MDS on the travel-time matrix,
    #       then normalized to a unit square for a clear layout.
    #     - Node size roughly reflects total number of containers at that terminal.
    #     - Node annotations:
    #         * node index inside the circle
    #         * 'I:x  E:y' under the node (imports/exports).
    #     - Barge paths:
    #         * one color per barge
    #         * line width proportional to TEU flow on that arc (imports + exports)
    #         * lateral offset per barge so overlapping arcs are clearly distinguishable.

    #     The figure is saved as 'Figures/barge_solution_map.png'.
    #     """
    #     m = self.model
    #     if m is None or m.status != GRB.OPTIMAL:
    #         print("No optimal solution available for plotting.")
    #         return


    #     # Data / sets
    #     T_ij_matrix = np.array(self.T_ij_matrix)
    #     N = self.N_list
    #     K_b = self.K_b
    #     C = self.C_list
    #     E = self.E
    #     I = self.I
    #     Z_cj = self.Z_cj
    #     W_c = self.W_c
    #     x_ijk = self.x_ijk
    #     y_ijk = self.y_ijk
    #     z_ijk = self.z_ijk

    #     # -----------------------------
    #     # Compute 2D node positions via MDS
    #     # -----------------------------
    #     mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    #     node_positions_raw = mds.fit_transform(T_ij_matrix)  # shape (N, 2)

    #     # Normalize positions to a [0, 1] x [0, 1] box for a clean aspect
    #     xs_raw = node_positions_raw[:, 0]
    #     ys_raw = node_positions_raw[:, 1]

    #     x_min0, x_max0 = xs_raw.min(), xs_raw.max()
    #     y_min0, y_max0 = ys_raw.min(), ys_raw.max()
    #     x_rng = x_max0 - x_min0 if x_max0 > x_min0 else 1.0
    #     y_rng = y_max0 - y_min0 if y_max0 > y_min0 else 1.0

    #     xs = (xs_raw - x_min0) / x_rng
    #     ys = (ys_raw - y_min0) / y_rng

    #     node_positions = np.column_stack([xs, ys])

    #     # -----------------------------
    #     # Node-level container stats
    #     # -----------------------------
    #     import_counts = {j: 0 for j in N}
    #     export_counts = {j: 0 for j in N}
    #     total_counts = {j: 0 for j in N}

    #     for c in C:
    #         for j in N:
    #             if Z_cj[c][j] == 1:
    #                 if c in I:
    #                     import_counts[j] += 1
    #                 elif c in E:
    #                     export_counts[j] += 1
    #                 total_counts[j] += 1
    #                 break

    #     max_containers = max(total_counts.values()) if total_counts else 1
    #     base_size = 120.0    # base marker size
    #     size_scale = 280.0   # extra size per relative container load

    #     node_sizes = {}
    #     for j in N:
    #         if max_containers > 0:
    #             node_sizes[j] = base_size + size_scale * (total_counts[j] / max_containers)
    #         else:
    #             node_sizes[j] = base_size

    #     # -----------------------------
    #     # Arc-level TEU flows (for line widths)
    #     # -----------------------------
    #     arc_flows = {}  # (i, j, k) -> TEU on arc i->j for barge k
    #     max_flow = 0.0
    #     for k in K_b:
    #         for i in N:
    #             for j in N:
    #                 if i == j:
    #                     continue
    #                 if x_ijk[i, j, k].X > 0.5:
    #                     flow_teu = y_ijk[i, j, k].X + z_ijk[i, j, k].X
    #                     arc_flows[(i, j, k)] = flow_teu
    #                     if flow_teu > max_flow:
    #                         max_flow = flow_teu

    #     if max_flow == 0:
    #         max_flow = 1.0  # avoids division by zero; all arcs get minimal width

    #     # Line width mapping
    #     min_lw = 0.6
    #     max_lw = 2.5

    #     def flow_to_lw(flow):
    #         return min_lw + (max_lw - min_lw) * (flow / max_flow)

    #     # -----------------------------
    #     # Plotting
    #     # -----------------------------
    #     fig, ax = plt.subplots(figsize=(8, 6))

    #     # For label offset we work in normalized coordinates
    #     y_span = ys.max() - ys.min() if ys.size > 0 else 1.0
    #     label_offset = 0.05 * y_span

    #     # Plot nodes
    #     for j in N:
    #         x, y = node_positions[j]
    #         size = node_sizes[j]

    #         # Draw node (dryport visually distinct)
    #         if j == 0:
    #             facecolor = "white"
    #             edgecolor = "black"
    #             linewidth = 1.5
    #         else:
    #             facecolor = "white"
    #             edgecolor = "black"
    #             linewidth = 1.0

    #         ax.scatter(
    #             x, y,
    #             s=size,
    #             edgecolor=edgecolor,
    #             facecolor=facecolor,
    #             linewidth=linewidth,
    #             zorder=3,
    #         )

    #         # Node id in the centre
    #         ax.text(
    #             x, y,
    #             f"{j}",
    #             ha="center", va="center",
    #             fontsize=9, fontweight="bold",
    #             zorder=4,
    #         )

    #         # I/E counts below the node
    #         imp = import_counts[j]
    #         exp = export_counts[j]
    #         ax.text(
    #             x, y - label_offset,
    #             f"I:{imp}  E:{exp}",
    #             ha="center", va="top",
    #             fontsize=7,
    #             zorder=4,
    #         )

    #     # -----------------------------
    #     # Plot barge paths
    #     # -----------------------------
    #     cmap = plt.get_cmap("tab10")
    #     barge_index_map = {k: idx for idx, k in enumerate(K_b)}
    #     offset_scale = 0.04  # lateral separation in normalized coordinates

    #     for k in K_b:
    #         color = cmap(barge_index_map[k] % 10)
    #         idx_k = barge_index_map[k]

    #         for i in N:
    #             for j in N:
    #                 if i == j:
    #                     continue
    #                 if x_ijk[i, j, k].X <= 0.5:
    #                     continue

    #                 x1, y1 = node_positions[i]
    #                 x2, y2 = node_positions[j]

    #                 dx = x2 - x1
    #                 dy = y2 - y1
    #                 length = (dx**2 + dy**2) ** 0.5
    #                 if length == 0:
    #                     continue

    #                 # Unit perpendicular vector
    #                 nx = -dy / length
    #                 ny = dx / length

    #                 # Lateral offset per barge
    #                 offset = offset_scale * (idx_k - (len(K_b) - 1) / 2.0)
    #                 x1_off = x1 + nx * offset
    #                 y1_off = y1 + ny * offset
    #                 x2_off = x2 + nx * offset
    #                 y2_off = y2 + ny * offset

    #                 flow_teu = arc_flows.get((i, j, k), 0.0)
    #                 lw = flow_to_lw(flow_teu)

    #                 # Shrink arrow slightly so it doesn't cover node centres
    #                 shrink = 0.10
    #                 x_start = x1_off + shrink * dx
    #                 y_start = y1_off + shrink * dy
    #                 x_end = x2_off - shrink * dx
    #                 y_end = y2_off - shrink * dy

    #                 arrow = FancyArrowPatch(
    #                     (x_start, y_start), (x_end, y_end),
    #                     arrowstyle="-|>",
    #                     linewidth=lw,
    #                     color=color,
    #                     alpha=0.75,
    #                     mutation_scale=8,
    #                     zorder=2,
    #                 )
    #                 ax.add_patch(arrow)

    #     # -----------------------------
    #     # Final styling
    #     # -----------------------------
    #     # Legend for barges
    #     for k in K_b:
    #         color = cmap(barge_index_map[k] % 10)
    #         ax.plot([], [], color=color, label=f"Barge {k}", linewidth=2)

    #     ax.set_title("Barge Routes and Terminal Containers (Final Solution)")
    #     ax.set_xlabel("MDS dimension 1 (normalized)")
    #     ax.set_ylabel("MDS dimension 2 (normalized)")
    #     ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    #     ax.set_aspect("equal", adjustable="box")

    #     # Nice padding around [0,1]x[0,1]
    #     ax.set_xlim(-0.1, 1.1)
    #     ax.set_ylim(-0.1, 1.1)

    #     ax.legend(
    #         loc="upper right",
    #         bbox_to_anchor=(1.02, 1.0),
    #         frameon=True,
    #         framealpha=0.8,
    #         fontsize=8,
    #     )

    #     plt.tight_layout()
    #     plt.savefig("Figures/barge_solution_map.png", dpi=300)
    #     # plt.show()
 











    # def generate_travel_times(self, aleatorio=False):
    #     """
    #     Build travel time matrix T_ij [hours].

    #     i = origin terminal
    #     j = destination terminal
    #     - long: travel time [hours] between dryport (0) and any other node
    #     - short: travel time [hours] between non-0 nodes
    #     - if aleatorio=True, generate random symmetric times in hours (except diagonal)
    #     """
    #     num_nodes = self.N
    #     rng = random.Random(self.seed + 1)

    #     long_time_min, long_time_max = self.travel_time_long_range
    #     short_time_min, short_time_max = self.travel_time_short_range

    #     T_ij_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

    #     for i in range(num_nodes):
    #         for j in range(num_nodes):
    #             if i == j:
    #                 T_ij = 0
    #             elif i == 0 or j == 0:
    #                 # From/to dryport (0)
    #                 T_ij = rng.randint(long_time_min, long_time_max)
    #             else:
    #                 T_ij = rng.randint(short_time_min, short_time_max)
    #             T_ij_matrix[i][j] = T_ij    # T_ij in hours

    #     if aleatorio:
    #         for i in range(num_nodes):
    #             for j in range(i + 1, num_nodes):
    #                 dist = rng.randint(1, 10)
    #                 T_ij_matrix[i][j] = dist
    #                 T_ij_matrix[j][i] = dist
    #         for i in range(num_nodes):
    #             T_ij_matrix[i][i] = 0

    #     self.T_ij_matrix = T_ij_matrix
















# # print_results is not being used currently. 
# def print_results(self):
#     """
#     Print detailed results in the same style as GreedyAlgo.print_results.
#     Computes cost decomposition and barge utilization from the MILP solution.
#     """
#     m = self.model
#     if m is None or m.status != GRB.OPTIMAL:
#         print("No optimal solution found. Status:", m.status if m is not None else "No model")
#         return

#     C = self.C_list
#     N = self.N_list
#     K_b = self.K_b
#     K_t = self.K_t

#     # Data
#     H_T = self.H_T
#     H_b = self.H_b
#     T = self.T_ij_matrix
#     Gamma = self.Gamma
#     Qk = self.Qk
#     W_c = self.W_c

#     # Variables
#     f_ck = self.f_ck
#     x_ijk = self.x_ijk

#     # Cost decomposition
#     truck_cost = sum(H_T[c] * f_ck[c, K_t].X for c in C)
#     barge_fixed_cost = sum(
#         H_b[k] * x_ijk[0, j, k].X for k in K_b for j in N if j != 0
#     )
#     travel_cost = sum(
#         T[i][j] * x_ijk[i, j, k].X
#         for k in K_b for i in N for j in N if i != j
#     )
#     terminal_penalty_cost = sum(
#         Gamma * x_ijk[i, j, k].X
#         for k in K_b for i in N for j in N if j != 0
#     )

#     barge_cost = barge_fixed_cost + travel_cost + terminal_penalty_cost
#     total_cost = truck_cost + barge_cost  # should match m.objVal

#     # Trucked containers
#     trucked_containers = sum(1 for c in C if f_ck[c, K_t].X > 0.5)

#     # Header-style summary
#     print("\n\nResults Table")
#     print(    "=============")
#     print(f"Total containers: {self.C}")
#     print(f"K_b (barges): {self.K_b}")
#     print(f"K_t (truck): {self.K_t}")
#     print(f"Total cost: {total_cost:>10.0f} Euros")
#     print(
#         f"Barge cost: {barge_cost:>10.0f} Euros             "
#         f"({barge_cost / total_cost * 100:>5.1f}% )"
#     )
#     print(
#         f"Truck cost: {truck_cost:>10.0f} Euros             "
#         f"({truck_cost / total_cost * 100:>5.1f}% )"
#     )
#     print(f"Containers: {self.C:>10d}")
#     print(f"Terminals: {self.N:>10d}")
#     print(
#         f"Trucked containers: {trucked_containers:>10d}           "
#         f"({trucked_containers / self.C * 100:>5.1f}% )"
#     )

#     # Barge utilization (same spirit as Greedy version)
#     for k in K_b:
#         # Count containers assigned to barge k
#         containers_on_barge = sum(1 for c in C if f_ck[c, k].X > 0.5)
#         if containers_on_barge == 0:
#             continue  # barge unused

#         # TEU on barge k (W_c is in TEU units)
#         teu_on_barge = sum(W_c[c] for c in C if f_ck[c, k].X > 0.5)

#         print(
#             f"Barge {k:>3d}: "
#             f"{containers_on_barge:>4d} containers, "
#             f"{teu_on_barge:>4d}/{Qk[k]:<4d} TEU"
#         )







# def print_results_old_format(self):
#     """Print basic optimization results."""
#     m = self.model
#     if m is None or m.status != GRB.OPTIMAL:
#         print("No optimal solution found. Status:", m.status if m is not None else "No model")
#         return

#     C = self.C_list
#     N = self.N_list
#     K = self.K_list
#     K_b = self.K_b
#     K_t = self.K_t
#     Qk = self.Qk

#     f_ck = self.f_ck
#     x_ijk = self.x_ijk
#     y_ijk = self.y_ijk
#     z_ijk = self.z_ijk

#     print("\n\nOptimal solution found!")
#     print(f"Optimal objective value: {m.objVal:.2f}")

#     # Container -> vehicle assignment
#     print("\nContainer to Vehicle Assignments (f_ck)")
#     print("========================================")
#     print(f"Number of containers: {len(C)}")

#     for c in C:
#         for k in K:
#             if f_ck[c, k].X > 0.5:
#                 if k == K_t:
#                     print(f"Container {c} is assigned to TRUCK (k={k})")
#                 else:
#                     print(f"Container {c} is assigned to BARGE {k}")

#     # Barge routes
#     print("\nBarge Travel Decisions (x_ijk)")
#     print("================================")
#     for k in K_b:
#         for i in N:
#             for j in N:
#                 if i != j and x_ijk[i, j, k].X > 0.5:
#                     print(f"Barge {k} travels from Terminal {i} to Terminal {j}")

#     # Capacity use on arcs involving node 0
#     print("\nCapacity usage on arcs involving node 0")
#     print("========================================")
#     for k in K_b:
#         for j in N:
#             if j == 0:
#                 continue

#             # j -> 0 (imports)
#             if y_ijk[j, 0, k].X > 0.5:
#                 assigned_quant = y_ijk[j, 0, k].X
#                 available_quant = Qk[k]
#                 print(
#                     f"Barge {k}, arc {j} -> 0: "
#                     f"{assigned_quant}/{available_quant} "
#                     f"({assigned_quant / available_quant * 100:.2f}%)"
#                 )

#             # 0 -> j (exports)
#             if z_ijk[0, j, k].X > 0.5:
#                 assigned_quant = z_ijk[0, j, k].X
#                 available_quant = Qk[k]
#                 print(
#                     f"Barge {k}, arc 0 -> {j}: "
#                     f"{assigned_quant}/{available_quant} "
#                     f"({assigned_quant / available_quant * 100:.2f}%)"
#                 )



