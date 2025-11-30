#!/usr/bin/env python3
"""
Mixed-Integer Linear Programming (MILP) model for container allocation optimization.
Implemented as the MILP_Algo class, structurally aligned with GreedyAlgo.
Requires Gurobi (gurobipy) with a valid license.

"""

import os
import toml
from datetime import datetime
import random
from gurobipy import Model, GRB, quicksum
import gurobipy as gp
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib.patches import FancyArrowPatch
import math
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



class MILP_Algo:
    def __init__(
            self,
            run_name="MILP_Run",
            qk=[  # Barge capacities in TEU
                20,         # Barge 0
                20,         # Barge 1
                20,         # Barge 2
                20,         # Barge 3
                10,         # Barge 4
                10,         # Barge 5
                10,         # Barge 6
            ],
            h_b=[  # Barge fixed costs in euros
                1100,      # Barge 0
                1700,      # Barge 1
                1800,      # Barge 2
                1900,      # Barge 3
                3300,      # Barge 4
                6300,      # Barge 5
                6300,      # Barge 6
            ],
            seed=0,
            reduced=False,
            h_t_40=200_000,                     # 40ft container trucking cost in euros
            h_t_20=140_000,                     # 20ft container trucking cost in euros
            handling_time=1/6,              # Container handling time in hours
            C_range=(150, 200),             # (min, max) number of containers when reduced=False
            N_range=(6, 6),                 # (min, max) number of terminals when reduced=False

            Oc_range=(24, 196),             # (min, max) opening time in hours
            Oc_offset_range=(50, 320),      # (min_offset, max_offset) such that
                                            # Dc is drawn in [Oc + min_offset, Oc + max_offset]

            travel_time_long_range=(3, 5),   # (min, max) travel time between dryport and sea terminals in hours
            travel_angle = math.pi, #* 1/6,          # angle sector for terminal placement
            # Travel_time_short_range=(1, 1),  # (min, max) travel time between sea terminals in hours

            P40_range=(0.2, 0.22),          # (min, max) probability of 40ft container
            PExport_range=(0.05, 0.75),      # (min, max) probability of export
            C_range_reduced=(65, 75),      # (min, max) containers when reduced=True
            N_range_reduced=(4, 4),         # (min, max) terminals when reduced=True
            gamma=100,                      # penalty per sea terminal visit [euros]
            big_m=1_000_000                 # big-M
    ):
        """
        Initialize the MILP optimi zer.

        Parameters mirror GreedyAlgo so that both can be constructed in the same way.
        Time-related ranges and all internal time variables are in hours.
        """
        self.time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.file_name = f"{run_name}_{self.time}"

        os.makedirs("Logs", exist_ok=True)
        os.makedirs("Solutions", exist_ok=True)
        os.makedirs("Settings", exist_ok=True)
        os.makedirs("Figures", exist_ok=True)

        dict_settings = {
            "run_name": run_name,
            "qk": qk,
            "h_b": h_b,
            "seed": seed,
            "reduced": reduced,
            "h_t_40": h_t_40,
            "h_t_20": h_t_20,
            "handling_time": handling_time,
            "C_range": C_range,
            "N_range": N_range,
            "Oc_range": Oc_range,
            "Oc_offset_range": Oc_offset_range,
            "travel_time_long_range": travel_time_long_range,
            "travel_angle": travel_angle,
            "P40_range": P40_range,
            "PExport_range": PExport_range,
            "C_range_reduced": C_range_reduced,
            "N_range_reduced": N_range_reduced,
            "gamma": gamma,
            "big_m": big_m
        }
        with open(f"Settings/settings_{self.file_name}.toml", "w") as f:
            toml.dump(dict_settings, f)

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

        self.travel_time_long_range = travel_time_long_range
        self.travel_angle = travel_angle
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
            theta = rng.uniform(math.pi - self.travel_angle, math.pi + self.travel_angle)

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

        self.model.Params.MIPFocus = 1   # focus on improving feasible solutions quickly
        # Options:
        # 0 = balanced (default)
        # 1 = feasibility
        # 2 = optimality
        # 3 = bound improvement

        # --------------- Gurobi configuration ---------------


        # Log file with timestamp to avoid overwriting
        self.model.Params.LogFile = f"Logs/log______{self.file_name}.log"

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
            limit_total_trucked_containers=False,
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
                        handling_term = quicksum(L * Z_cj[c][i] * f_ck[c, k] for c in C)

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
            
            
        ##############
        #### List ####
        ##############
            
        # 1. Each container is assigned to exactly one vehicle
#  print table. 
        # 2. Flow conservation for barges at each node
# checked
        # 3. Each barge leaves dryport (0) at most once
# checked
        # 4. No self-loops (i -> i)
# checked        
        # 5. Import quantity at terminal j for barge k

        # 6. Export quantity at terminal j for barge k

        # 7. Flow balance for import quantities at terminal j for barge k
# checked
        # 8. Flow balance for export quantities at terminal j for barge k
# checked
        # 9. Barge trip capacity constraint
# almost checked
        # 10. Export containers: departure time at dryport >= release time
# box plot looking thing. 
        # 11 & 12. Time propagation along arcs with handling time at arrival

        # 13. Export container service cannot start before opening time
# box plot looking thing
        # 14. All containers must be served before closing time
# box plot looking thing. 




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
        print("\nBarge Utilization Summary (considering all trips made)")

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
            if node == 0:
                # Dryport: count all imports (arrive to node 0) and all exports (depart from node 0)
                import_count = -len(I)
                export_count = -len(E)
            else:
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
    def plot_barge_solution_map_report_3(self):
        """
        Curved-edge barge route map with segment-order alpha encoding.
        Improvements over _2:
        - Each barge's path segments fade in as the barge progresses:
              early arcs → low alpha
              late arcs → alpha=1
        - Minimalistic aesthetic
        """



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
                    x, y, s=800, marker="s",
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
        multi = 2.0
        # curvature_values = [0.2*multi, 0.25*multi, 0.3*multi, 0.35*multi, 0.4*multi, 0.45*multi, -0.2*multi, -0.25*multi, -0.3*multi, -0.35*multi, -0.4*multi, -0.45*multi,]
        # curvature_values = [0.1*multi, 0.2*multi, 0.3*multi, 0.4*multi, 0.5*multi, 0.6*multi, -0.1*multi, -0.2*multi, -0.3*multi, -0.4*multi, -0.5*multi, -0.6*multi,]
        curvature_values = [0.1*multi, 0.2*multi, 0.3*multi, 0.4*multi, 0.5*multi, 0.6*multi] #, -0.2*multi, -0.3*multi, -0.4*multi, -0.5*multi, -0.6*multi,]

        legend_lines = []
        legend_labels = []

        for idx, k in enumerate(used_barges):
            linestyle = line_styles[idx]
            curvature = curvature_values[idx % len(curvature_values)]

            arcs = barge_arcs_ordered[k]
            S = max(1, len(arcs))  # number of segments

            alphas = np.linspace(1.0, 0.25, S)

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


                t_peak = 0.5
                peak = bezier_quad(P0, P1, P2, t_peak)

                # 2) tangent at t = 0.5
                T = 2 * (1 - t_peak) * (P1 - P0) + 2 * t_peak * (P2 - P1)
                T_norm = np.linalg.norm(T)
                if T_norm > 0:
                    # normal to the curve at the peak
                    n_curve = np.array([-T[1], T[0]]) / T_norm
                else:
                    # fallback: use the segment direction normal
                    n_curve = np.array([-d[1], d[0]]) / L

                # 3) choose how far away from the curve to place the stack
                offset_dist = 0.03  # tune this number to taste

                anchor_x = peak[0] + offset_dist * n_curve[0]
                anchor_y = peak[1] + offset_dist * n_curve[1]

                def compute_signs(P0, P2, anchor_x, anchor_y):
                    """
                    Compute sign_x and sign_y by comparing the anchor point (anchor_x, anchor_y)
                    to the midpoint of the straight line between P0 and P2.

                    Returns:
                        sign_x, sign_y  ∈ { -1, 1 }
                    """
                    # Midpoint of the chord
                    M = (P0 + P2) / 2     # array([Mx, My])
                    Mx, My = M

                    dx = anchor_x - Mx
                    dy = anchor_y - My

                    # Determine signs (never 0: 0 -> +1)
                    sign_x = 1 if dx >= 0 else -1
                    sign_y = 1 if dy >= 0 else -1

                    return sign_x, sign_y
                
                sign_x, sign_y = compute_signs(P0, P2, anchor_x, anchor_y)

                width_scaled = 0.09
                plotter = ContainerPlotter(width=width_scaled, x=anchor_x, y=anchor_y, sign_x=sign_x, sign_y=sign_y)
       
                total_capacity = self.Qk[k]
                self._draw_segment_stack(ax, plotter, i, j, k,
                                         total_width=5, max_rows=total_capacity / 5)



            # legend entry for this barge
            line, = ax.plot([], [], color="black", linewidth=2.5, linestyle=linestyle)
            legend_lines.append(line)
            legend_labels.append(f"Barge {k}")
            # ==========================================
            # Add Import / Export container legend icons
            # ==========================================


        # Small representative container size
        legend_w = width_scaled
        legend_h = legend_w * 8.6 / 20

        import_patch = Rectangle(
            (0, 0),
            legend_w,
            legend_h,
            facecolor="#00A63C",
            edgecolor="#006E28",
            linewidth=1.5
        )
        export_patch = Rectangle(
            (0, 0),
            legend_w,
            legend_h,
            facecolor="#FF6F00",
            edgecolor="#B23E00",
            linewidth=1.5
        )

        legend_lines.extend([export_patch, import_patch])
        legend_labels.extend(["Export", "Import"])

        # --------------------------
        # Styling
        # --------------------------
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Legend
        ax.legend(
        legend_lines, legend_labels,
        loc="upper right",
        frameon=True,
        fontsize=14,
        handlelength= 4.2,      # default is 2 — increase for longer patterns
        handletextpad=0.8,   # spacing between line and text
    )

        plt.tight_layout()
        plt.savefig(f"Figures/solution_map{self.file_name}.pdf")

    def _draw_segment_stack(self, ax, plotter, i, j, k,
                            total_width, max_rows):
        """
        # used in the plot_barge_solution_map_report_3 function
        Draw a container stack for arc (i, j, k) using actual TEU on that segment.

        Assumes:
            - y_ijk[i, j, k].X ≈ import TEU on this arc
            - z_ijk[i, j, k].X ≈ export TEU on this arc
        These are interpreted in units of 1 TEU (i.e. W_c = 1 for drawing).
        """
        total_width += 1
        # TEU on this arc (rounded to nearest int)
        import_teu = int(round(self.y_ijk[i, j, k].X))
        export_teu = int(round(self.z_ijk[i, j, k].X))

        total_teu = import_teu + export_teu
        if total_teu <= 0:
            return  # nothing to draw

        # Capacity of the drawing grid (in TEU slots)
        max_cols = max(1, total_width - 1)
        max_slots = max_rows * max_cols

        # # Do not draw more slots than the grid can show
        # total_teu = min(total_teu, max_slots)

        # Draw bounding box
        plotter.draw_capacity(ax, total_height=max_rows, total_width=total_width)

        # Fill slots in order: first imports (green), then exports (orange)
        idx = 1
        remaining_import = import_teu
        remaining_export = export_teu

        while idx <= total_teu:
            if remaining_import > 0:
                IorE = 1  # import
                remaining_import -= 1
            elif remaining_export > 0:
                IorE = 2  # export
                remaining_export -= 1
            else:
                break

            plotter.draw_container(
                ax,
                index=idx,
                total_height=max_rows,
                total_width=total_width,
                IorE=IorE,
                W_c=1    # treat each TEU as a 20ft for visualization
            )
            idx += 1

    def plot_time_windows(self, row_spacing: float = 0.1):
        """
        Plot container time windows and service times to visually verify time constraints.
        """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines

        m = self.model
        if m is None or m.status != GRB.OPTIMAL:
            print("No optimal solution available for plotting time windows.")
            return

        # --------------------------
        # Extract data
        # --------------------------
        C = self.C_list
        E = set(self.E)
        I = set(self.I)

        R_c = self.R_c
        O_c = self.O_c
        D_c = self.D_c
        Z_cj = self.Z_cj

        K = self.K_list
        K_b = set(self.K_b)
        K_t = self.K_t

        f_ck = self.f_ck
        t_jk = self.t_jk

        # --------------------------
        # Build ordered container list: exports first, then imports
        # --------------------------
        export_ids = sorted(E)
        import_ids = sorted(I)
        container_rows = export_ids + import_ids

        if not container_rows:
            print("No containers to plot.")
            return

        # Number of rows
        n_rows = len(container_rows)                                               # <<< changed

        # Adapt row spacing so rows fill vertical axis without wasted space
        if n_rows > 1:                                                             # <<< changed
            row_spacing = 1.0 / (n_rows - 1)                                       # <<< changed
        else:                                                                      # <<< changed
            row_spacing = 1.0                                                      # <<< changed

        # Map container -> row index (0 at top, but we'll invert axis later)
        row_index = {c: idx for idx, c in enumerate(container_rows)}

        # --------------------------
        # X-axis range: earliest opening to latest closing
        # --------------------------
        min_open = min(O_c[c] for c in C)
        max_close = max(D_c[c] for c in C)
        span = max_close - min_open
        margin = max(0.05 * span, 1.0)

        x_min = min_open - margin
        x_max = max_close + margin

        # --------------------------
        # Prepare figure
        # --------------------------
        # Fix width, adapt height to number of containers
        fig_width = 6                                                              # <<< changed
        base_height_per_row = 0.45                                                 # <<< changed
        fig_height = base_height_per_row * n_rows                                  # <<< changed
        fig_height = min(max(fig_height, 3.0), 12.0)  # clamp between 3" and 12"   # <<< changed

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))                    # <<< changed
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # --------------------------
        # Plot each container's time info
        # --------------------------
        for c in container_rows:
            y = row_index[c] * row_spacing

            if c in E:
                color      = "#FF6F00"
                color_dark = "#B23E00"
            if c in I:
                color      = "#00A63C"
                color_dark = "#006E28"

            R = R_c[c]
            O = O_c[c]
            D = D_c[c]

            # time window line [O, D]
            ax.hlines(
                y, O, D,
                colors=color_dark,
                linewidth=2,
            )

            # release time (square)
            ax.scatter(
                R, y,
                marker="s",
                s=30,
                facecolor="white",
                edgecolor="black",
                linewidth=1.0,
                zorder=3,
            )

            # opening time 
            ax.scatter(
                O, y,
                marker=">",
                s=50,
                facecolor=color,
                edgecolor=color_dark,
                linewidth=1.0,
                zorder=3,
            )

            # closing time 
            ax.scatter(
                D, y,
                marker="<",
                s=50,
                facecolor=color,
                edgecolor=color_dark,
                linewidth=1.0,
                zorder=3,
            )

            # Determine assigned vehicle
            assigned_k = None
            for k in K:
                if f_ck[c, k].X > 0.5:
                    assigned_k = k
                    break

            # Service time marker (only if barge)
            if assigned_k is not None and assigned_k in K_b:
                try:
                    j = Z_cj[c].index(1)
                except ValueError:
                    j = None

                if j is not None:
                    service_time = t_jk[j, assigned_k].X
                    ax.scatter(
                        service_time, y,
                        marker="2",
                        s=80,
                        facecolor="black",
                        linewidth=1.6,
                        zorder=4,
                    )

        # --------------------------
        # Axis formatting
        # --------------------------
        ax.set_xlim(x_min, x_max)

        yticks = [row_index[c] * row_spacing for c in container_rows]
        ylabels = [f"{c}" for c in container_rows]

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)

        ax.invert_yaxis()

        ax.set_xlabel("Time [hours]", fontsize=10)
        ax.set_ylabel("Containers", fontsize=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        # --------------------------
        # Legend below the plot
        # --------------------------
        export_window = mlines.Line2D([], [], color="#FF6F00", linewidth=2)
        import_window = mlines.Line2D([], [], color="#00A63C", linewidth=2)
        marker_release  = mlines.Line2D([], [], color="black", marker="s", linestyle="None", markerfacecolor="white", markeredgecolor="black")
        marker_service  = mlines.Line2D([], [], markeredgecolor="black", marker="2", linestyle="None")

        legend_handles = [
            export_window,
            import_window,
            marker_service,
            marker_release,
        ]
        legend_labels = [
            "Export time window [O, D]",
            "Import time window [O, D]",
            "Delivery/Pickup time t_jk",
            "Release time R",
        ]


        # after computing n_rows and row_spacing:


        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=2,
            frameon=True,
            fontsize=8,
        )


        plt.tight_layout()

        outfile = f"Figures/time_windows{self.file_name}.pdf"
        plt.savefig(outfile, dpi=300)
    def plot_time_windows(self, row_spacing: float = 0.1):
        """
        Plot container time windows and service times to visually verify time constraints.
        """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines

        m = self.model
        if m is None or m.status != GRB.OPTIMAL:
            print("No optimal solution available for plotting time windows.")
            return

        # --------------------------
        # Extract data
        # --------------------------
        C = self.C_list
        E = set(self.E)
        I = set(self.I)

        R_c = self.R_c
        O_c = self.O_c
        D_c = self.D_c
        Z_cj = self.Z_cj

        K = self.K_list
        K_b = set(self.K_b)
        K_t = self.K_t

        f_ck = self.f_ck
        t_jk = self.t_jk

        # --------------------------
        # Precompute assigned barge and service time per container
        # --------------------------
        assigned_barge = {}   # c -> k (if barge) or None
        service_time_c = {}   # c -> service time (if barge) or None

        for c in C:
            assigned_k = None
            for k in K:
                if f_ck[c, k].X > 0.5:
                    assigned_k = k
                    break

            if assigned_k is not None and assigned_k in K_b:
                # Find terminal j from Z_cj[c]
                try:
                    j = Z_cj[c].index(1)
                except ValueError:
                    j = None

                if j is not None:
                    st = t_jk[j, assigned_k].X
                else:
                    st = None

                assigned_barge[c] = assigned_k
                service_time_c[c] = st
            else:
                assigned_barge[c] = None
                service_time_c[c] = None

        # --------------------------
        # Build ordered container list:
        #   - exports first, then imports
        #   - within each type: group by barge index, order by service time
        # --------------------------
        export_ids = sorted(E)
        import_ids = sorted(I)

        export_rows = []
        import_rows = []

        # Helper: sorted containers of a given list, for a given barge k
        def containers_for_barge(container_list, k):
            conts = [c for c in container_list if assigned_barge.get(c) == k]
            # Order by service time; None goes last if it ever appears
            conts_sorted = sorted(
                conts,
                key=lambda c: (service_time_c[c] is None, service_time_c[c])
            )
            return conts_sorted

        # Exports, grouped by barge index
        for k in sorted(K_b):
            export_rows.extend(containers_for_barge(export_ids, k))
        # Exports without barge (e.g. truck)
        export_no_barge = [c for c in export_ids if assigned_barge.get(c) is None]
        export_rows.extend(sorted(export_no_barge))

        # Imports, grouped by barge index
        for k in sorted(K_b):
            import_rows.extend(containers_for_barge(import_ids, k))
        # Imports without barge (e.g. truck)
        import_no_barge = [c for c in import_ids if assigned_barge.get(c) is None]
        import_rows.extend(sorted(import_no_barge))

        # Final row order: exports, then imports
        container_rows = export_rows + import_rows

        if not container_rows:
            print("No containers to plot.")
            return

        # Number of rows
        n_rows = len(container_rows)

        # Adapt row spacing so rows fill vertical axis without wasted space
        if n_rows > 1:
            row_spacing = 1.0 / (n_rows - 1)
        else:
            row_spacing = 1.0

        # Map container -> row index (0 at top, but we'll invert axis later)
        row_index = {c: idx for idx, c in enumerate(container_rows)}

        # --------------------------
        # X-axis range: earliest opening to latest closing
        # --------------------------
        min_open = min(O_c[c] for c in C)
        max_close = max(D_c[c] for c in C)
        span = max_close - min_open
        margin = max(0.05 * span, 1.0)

        x_min = min_open - margin
        x_max = max_close + margin

        # --------------------------
        # Prepare figure
        # --------------------------
        # Fix width, adapt height to number of containers
        fig_width = 6
        base_height_per_row = 0.45
        fig_height = base_height_per_row * n_rows
        fig_height = min(max(fig_height, 3.0), 12.0)  # clamp between 3" and 12"

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # --------------------------
        # Plot each container's time info
        # --------------------------
        for c in container_rows:
            y = row_index[c] * row_spacing

            if c in E:
                color      = "#FF6F00"
                color_dark = "#B23E00"
            if c in I:
                color      = "#00A63C"
                color_dark = "#006E28"

            R = R_c[c]
            O = O_c[c]
            D = D_c[c]

            # time window line [O, D]
            ax.hlines(
                y, O, D,
                colors=color_dark,
                linewidth=2,
            )

            # release time (square)
            ax.scatter(
                R, y,
                marker="s",
                s=30,
                facecolor="white",
                edgecolor="black",
                linewidth=1.0,
                zorder=3,
            )

            # opening time
            ax.scatter(
                O, y,
                marker=">",
                s=50,
                facecolor=color,
                edgecolor=color_dark,
                linewidth=1.0,
                zorder=3,
            )

            # closing time
            ax.scatter(
                D, y,
                marker="<",
                s=50,
                facecolor=color,
                edgecolor=color_dark,
                linewidth=1.0,
                zorder=3,
            )

            # Service time marker (only if assigned to barge)
            assigned_k = assigned_barge.get(c)
            st = service_time_c.get(c)
            if assigned_k is not None and assigned_k in K_b and st is not None:
                ax.scatter(
                    st, y,
                    marker="2",
                    s=80,
                    facecolor="black",
                    linewidth=1.6,
                    zorder=4,
                )

        # --------------------------
        # Axis formatting
        # --------------------------
        ax.set_xlim(x_min, x_max)

        yticks = [row_index[c] * row_spacing for c in container_rows]

        # Widths for aligned labels
        k_width = max(1, max((len(str(k)) for k in K_b), default=1))
        c_width = max(1, max(len(str(c)) for c in C))

        ylabels = []
        for c in container_rows:
            k = assigned_barge.get(c)
            label = f"B{str(k).rjust(k_width)}".ljust(k_width+2) + " | " + str(c).rjust(c_width, '0')
            ylabels.append(label)

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)

        ax.invert_yaxis()

        ax.set_xlabel("Time [hours]", fontsize=10)
        ax.set_ylabel("Containers", fontsize=10)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        # --------------------------
        # Legend below the plot
        # --------------------------
        export_window = mlines.Line2D([], [], color="#FF6F00", linewidth=2)
        import_window = mlines.Line2D([], [], color="#00A63C", linewidth=2)
        marker_release = mlines.Line2D(
            [], [], color="black", marker="s", linestyle="None",
            markerfacecolor="white", markeredgecolor="black"
        )
        marker_service = mlines.Line2D(
            [], [], markeredgecolor="black", marker="2", linestyle="None"
        )

        legend_handles = [
            export_window,
            import_window,
            marker_service,
            marker_release,
        ]
        legend_labels = [
            "Export time window [O, D]",
            "Import time window [O, D]",
            "Delivery/Pickup time t_jk",
            "Release time R",
        ]

        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=2,
            frameon=True,
            fontsize=8,
        )

        plt.tight_layout()

        outfile = f"Figures/time_windows{self.file_name}.pdf"
        plt.savefig(outfile, dpi=300)


    def plot_time_windows(self, row_spacing: float = 0.1):
        """
        Plot container time windows and service times to visually verify time constraints.
        """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines

        m = self.model
        if m is None or m.status != GRB.OPTIMAL:
            print("No optimal solution available for plotting time windows.")
            return

        # --------------------------
        # Extract data
        # --------------------------
        C = self.C_list
        E = set(self.E)
        I = set(self.I)

        R_c = self.R_c
        O_c = self.O_c
        D_c = self.D_c
        Z_cj = self.Z_cj

        K = self.K_list
        K_b = set(self.K_b)
        K_t = self.K_t

        f_ck = self.f_ck
        t_jk = self.t_jk

        # --------------------------
        # Precompute assigned barge + service time
        # --------------------------
        assigned_barge = {}
        service_time_c = {}

        for c in C:
            assigned_k = None
            for k in K:
                if f_ck[c, k].X > 0.5:
                    assigned_k = k
                    break

            if assigned_k is not None and assigned_k in K_b:
                try:
                    j = Z_cj[c].index(1)
                except ValueError:
                    j = None

                st = t_jk[j, assigned_k].X if j is not None else None
                assigned_barge[c] = assigned_k
                service_time_c[c] = st
            else:
                assigned_barge[c] = None
                service_time_c[c] = None

        # --------------------------
        # Build ordered rows:
        #   Header row first
        #   Then exports grouped by barge + sorted by service time
        #   Then imports grouped by barge + sorted by service time
        # --------------------------
        export_ids = sorted(E)
        import_ids = sorted(I)

        def containers_for_barge(container_list, k):
            conts = [c for c in container_list if assigned_barge.get(c) == k]
            conts_sorted = sorted(conts, key=lambda c: (service_time_c[c] is None, service_time_c[c]))
            return conts_sorted

        export_rows = []
        import_rows = []

        # group exports
        for k in sorted(K_b):
            export_rows.extend(containers_for_barge(export_ids, k))
        export_rows.extend(sorted([c for c in export_ids if assigned_barge.get(c) is None]))

        # group imports
        for k in sorted(K_b):
            import_rows.extend(containers_for_barge(import_ids, k))
        import_rows.extend(sorted([c for c in import_ids if assigned_barge.get(c) is None]))

        # header entry:
        HEADER = "__HEADER__"

        # final row order
        container_rows = [HEADER] + export_rows + import_rows

        # --------------------------
        # Row spacing logic
        # --------------------------
        n_rows = len(container_rows)

        if n_rows > 1:
            row_spacing = 1.0 / (n_rows - 1)
        else:
            row_spacing = 1.0

        row_index = {c: idx for idx, c in enumerate(container_rows)}

        # --------------------------
        # X-axis range
        # --------------------------
        min_open = min(O_c[c] for c in C)
        max_close = max(D_c[c] for c in C)
        span = max_close - min_open
        margin = max(0.05 * span, 1.0)

        x_min = min_open - margin
        x_max = max_close + margin

        # --------------------------
        # Figure
        # --------------------------
        fig_width = 6
        base_height_per_row = 0.45
        fig_height = base_height_per_row * n_rows
        fig_height = min(max(fig_height, 3.0), 12.0)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # --------------------------
        # Plot rows (skip header row)
        # --------------------------
        for c in container_rows:
            if c == HEADER:
                continue

            y = row_index[c] * row_spacing

            if c in E:
                color, color_dark = "#FF6F00", "#B23E00"
            else:
                color, color_dark = "#00A63C", "#006E28"

            R, O, D = R_c[c], O_c[c], D_c[c]

            ax.hlines(y, O, D, colors=color_dark, linewidth=2)

            ax.scatter(R, y, marker="s", s=30, facecolor="white",
                    edgecolor="black", linewidth=1.0, zorder=3)

            ax.scatter(O, y, marker=">", s=50, facecolor=color,
                    edgecolor=color_dark, linewidth=1.0, zorder=3)

            ax.scatter(D, y, marker="<", s=50, facecolor=color,
                    edgecolor=color_dark, linewidth=1.0, zorder=3)

            k = assigned_barge.get(c)
            st = service_time_c.get(c)
            if k is not None and k in K_b and st is not None:
                ax.scatter(st, y, marker="2", s=80, facecolor="black",
                        linewidth=1.6, zorder=4)

        # --------------------------
        # Axis formatting
        # --------------------------
        ax.set_xlim(x_min, x_max)

        yticks = [row_index[c] * row_spacing for c in container_rows]

        # widths for alignment
        k_width = max(1, max((len(str(k)) for k in K_b), default=1))
        c_width = max(1, max(len(str(c)) for c in C))

        ylabels = []
        for c in container_rows:
            if c == HEADER:
                # ylabels.append(r"$\bf{Barge\;\;\;\;Container}$")
                ylabels.append(r"$\bf{Barge\;\;\;Cont}$")
                continue

            k = assigned_barge.get(c)
            k_str = str(k) if k is not None else "-"
            # label = f"[B{str(k_str).rjust(k_width)}]   C={str(c).rjust(c_width,'0')}"
            label = f"B{str(k_str).rjust(k_width)}".ljust(k_width+2) + "   |   " + str(c).rjust(c_width, '0') + "  "
            ylabels.append(label)

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=9)

        ax.invert_yaxis()

        ax.set_xlabel("Time [hours]", fontsize=10)
        ax.set_ylabel("")  # REMOVE VERTICAL Y-AXIS LABEL

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        # --------------------------
        # Legend
        # --------------------------
        export_window = mlines.Line2D([], [], color="#FF6F00", linewidth=2)
        import_window = mlines.Line2D([], [], color="#00A63C", linewidth=2)
        marker_release = mlines.Line2D([], [], color="black", marker="s", linestyle="None",
                                    markerfacecolor="white", markeredgecolor="black")
        marker_service = mlines.Line2D([], [], markeredgecolor="black", marker="2", linestyle="None")

        legend_handles = [
            export_window,
            import_window,
            marker_service,
            marker_release,
        ]
        legend_labels = [
            "Export time window [O, D]",
            "Import time window [O, D]",
            "Delivery/Pickup time t_jk",
            "Release time R",
        ]

        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=2,
            frameon=True,
            fontsize=8,
        )

        plt.tight_layout()

        outfile = f"Figures/time_windows{self.file_name}.pdf"
        plt.savefig(outfile, dpi=300)



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

        # -----------------------------------------------------------
        # Save and reload optimized model for later use
        # -----------------------------------------------------------
        if self.model.status == GRB.OPTIMAL:

            # Ensure directory exists
            import os
                    # self.name_run = f"Solutions/solved_{run_name}_{self.time}"


            # Model save path (.sol is Gurobi’s recommended solution format)
            save_path = f"Solutions/solved_{self.file_name}.sol"

            # Save optimized model
            self.model.write(save_path)
            print(f"\nSaved optimized model to: {save_path}")

            # Load model back into memory
            # Note: This loads solution attributes but not constraints/vars by name.
            try:
                self.loaded_model = gp.read(save_path)
                print(f"Reloaded optimized model from: {save_path}")
            except Exception as e:
                print(f"Could not reload saved model: {e}")
        

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
                self.plot_time_windows()


class ContainerPlotter:
    """
    Minimal wrapper class for draw_container() and draw_capacity().
    Logic, colours, and geometry remain EXACTLY as in the original functions.
    """

    def __init__(self, width=20, x=22, y=22, sign_x=1, sign_y=1):
        # Colors (exact same values)
        self.color_green = "#00A63C"
        self.color_green_dark = "#006E28"

        self.color_orange = "#FF6F00"
        self.color_orange_dark = "#B23E00"

        self.color_gray = "#888888"

        self.width = width
        self.height = width * 8.6 / 20

        self.starting_x = x
        self.starting_y = y

        self.sign_x = sign_x
        self.sign_y = sign_y

    # ------------------------------------------------------------------
    def draw_container(self, ax, index, total_height, total_width, IorE=1, W_c=1 ):
        """
        Draw a single container in a grid layout.
        IDENTICAL to your original function.
        """

        # Effective number of columns per row
        max_cols = max(1, total_width - 1)

        idx0 = index - 1
        row = idx0 // max_cols
        col = idx0 % max_cols

        x = self.starting_x + col * self.width
        y = self.starting_y + row * self.height

        if self.sign_x == -1:
            x = self.starting_x - (col + 1) * self.width
        
        if self.sign_y == -1:
            y = self.starting_y - total_height * self.height + (row - 1) * self.height

        # Colours
        if IorE == 1:
            face = self.color_green
            edge = self.color_green_dark
        elif IorE == 2:
            face = self.color_orange
            edge = self.color_orange_dark
        else:
            face = "#CCCCCC"
            edge = "#888888"

        rect = Rectangle(
            (x, y),
            self.width * W_c,
            self.height,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.8
        )
        ax.add_patch(rect)

    # ------------------------------------------------------------------
    def draw_capacity(self, ax, total_height, total_width):
        """
        Draw the grey bounding box around the full capacity.
        IDENTICAL to your original function.
        """
        max_cols = max(1, total_width - 1)

        margin = self.width / 10

        total_w = max_cols * self.width
        total_h = total_height * self.height

        color_gray = self.color_gray
        lw = 3

        if self.sign_y == 1:
            ax.plot(
                [self.starting_x - self.sign_x * margin * 1.1, self.starting_x - self.sign_x * margin * 1.1],
                [self.starting_y - margin, self.starting_y + total_h],
                color=color_gray, linewidth=lw
            )

            ax.plot(
                [self.starting_x - self.sign_x * margin, self.starting_x + self.sign_x * total_w + self.sign_x * margin],
                [self.starting_y - margin, self.starting_y - margin],
                color=color_gray, linewidth=lw
            )

            ax.plot(
                [self.starting_x + self.sign_x * total_w + self.sign_x * margin, self.starting_x + self.sign_x * total_w + self.sign_x * margin],
                [self.starting_y - margin, self.starting_y + total_h],
                color=color_gray, linewidth=lw
            )



        elif self.sign_y == -1:
            # Left vertical line (flipped vertically)
            ax.plot(
                [self.starting_x - self.sign_x * margin * 1.1, 
                self.starting_x - self.sign_x * margin * 1.1],
                [self.starting_y - total_h - self.height - margin, 
                self.starting_y - self.height - margin],
                color=color_gray, linewidth=lw
            )


            # Bottom horizontal line (flipped vertically)
            ax.plot(
                [self.starting_x - self.sign_x * margin,
                self.starting_x + self.sign_x * total_w + self.sign_x * margin],
                [self.starting_y - total_h - self.height - margin, self.starting_y - total_h - self.height- margin],
                color=color_gray, linewidth=lw
            )

            # Right vertical line (flipped vertically)
            ax.plot(
                [self.starting_x + self.sign_x * total_w + self.sign_x * margin, 
                self.starting_x + self.sign_x * total_w + self.sign_x * margin],
                [self.starting_y - total_h - self.height - margin, 
                self.starting_y - self.height - margin],
                color=color_gray, linewidth=lw
            )



# Optional quick test if you run MILP.py directly:
if __name__ == "__main__":
    print("\n\n\n\n\n\n\n\n\n\n\n")
    milp = MILP_Algo(reduced=True)   # e.g. smaller instances
    milp.run(with_plots=True)

