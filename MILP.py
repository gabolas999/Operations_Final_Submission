#!/usr/bin/env python3
"""
Mixed-Integer Linear Programming (MILP) model for container allocation optimization.
Implemented as the MILP_Algo class, structurally aligned with GreedyAlgo.
Requires Gurobi (gurobipy) with a valid license.
"""

import random
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np
import pandas as pd
from tabulate import tabulate


class MILP_Algo:
    def __init__(
            self,
            qk=[  # Barge capacities in TEU
                104,        # Barge 0
                99,         # Barge 1
                81,         # Barge 2
                # 52,         # Barge 3
                # 28,         # Barge 4
            ],
            h_b=[  # Barge fixed costs in euros
                3700,      # Barge 0
                3600,      # Barge 1
                3400,      # Barge 2
                # 2800,      # Barge 3
                # 1800,      # Barge 4
            ],
            seed=100,
            reduced=False,
            h_t_40=200,                     # 40ft container trucking cost in euros
            h_t_20=140,                     # 20ft container trucking cost in euros
            handling_time=1/6,              # Container handling time in hours
            C_range=(100, 600),             # (min, max) number of containers when reduced=False
            N_range=(10, 20),               # (min, max) number of terminals when reduced=False
            Dc_range=(24, 196),             # (min, max) closing time in hours
            Oc_offset_range=(24, 120),      # (min_offset, max_offset) such that
                                            # Oc is drawn in [Dc - max_offset, Dc - min_offset]
            P40_range=(0.75, 0.9),          # (min, max) probability of 40ft container
            PExport_range=(0.05, 0.7),      # (min, max) probability of export
            C_range_reduced=(33, 200),      # (min, max) containers when reduced=True
            N_range_reduced=(2, 3),         # (min, max) terminals when reduced=True
            gamma=100,                      # penalty per sea terminal visit [euros]
            big_m=1_000_000                 # big-M
    ):
        """
        Initialize the MILP optimizer.

        Parameters mirror GreedyAlgo so that both can be constructed in the same way.
        Time-related ranges are in hours (like GreedyAlgo) and converted to minutes
        internally for the MILP model.
        """
        # Parameters
        self.seed = seed
        self.reduced = reduced
        self.Qk = qk          # barge capacities in TEU
        self.H_b = h_b        # barge fixed costs in euros
        self.H_t_40 = h_t_40  # euros per 40ft container
        self.H_t_20 = h_t_20  # euros per 20ft container
        self.Handling_time_hours = handling_time  # hours per container
        self.Handling_time_minutes = handling_time * 60.0

        # Ranges (hours / probabilities)
        self.C_range = C_range
        self.N_range = N_range
        self.Dc_range = Dc_range
        self.Oc_offset_range = Oc_offset_range
        self.P40_range = P40_range
        self.PExport_range = PExport_range

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
        self.R_c = []               # release times [minutes]
        self.O_c = []               # opening times [minutes]
        self.D_c = []               # closing times [minutes]
        self.H_T = []               # trucking cost per container [euros]
        self.Z_cj = []              # assignment of container c to terminal j
        self.C_dict = {}            # same structure as in GreedyAlgo for compatibility

        self.T_ij_matrix = []       # travel time matrix [minutes]

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
        as GreedyAlgo.generate_container_info, but converted to minutes
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
        Dc_min_hr, Dc_max_hr = self.Dc_range
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
            # Closing time in hours
            Dc_hr = rng.randint(Dc_min_hr, Dc_max_hr)
            # Opening time in hours (before closing, within offsets)
            Oc_hr = rng.randint(Dc_hr - Oc_off_max_hr, Dc_hr - Oc_off_min_hr)

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

            # Convert times to minutes for MILP model
            Rc_min = Rc_hr * 60
            Oc_min = Oc_hr * 60
            Dc_min = Dc_hr * 60

            # Trucking cost per container
            if W_teu == 1:
                truck_cost = self.H_t_20
            else:
                truck_cost = self.H_t_40

            # Store
            self.W_c.append(W_teu)
            self.R_c.append(Rc_min)
            self.O_c.append(Oc_min)
            self.D_c.append(Dc_min)
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

    def generate_travel_times(self, long=5, short=1, aleatorio=False):
        """
        Build travel time matrix T_ij [minutes].

        i = origin terminal
        j = destination terminal
        - long: time between dryport (0) and any other node
        - short: time between non-0 nodes
        - if aleatorio=True, generate random symmetric times (except diagonal)
        """
        num_nodes = self.N
        rng = random.Random(self.seed + 1)

        T_ij_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    T_ij = 0
                elif i == 0 or j == 0:
                    # From/to dryport (0)
                    T_ij = long
                else:
                    T_ij = short
                T_ij_matrix[i][j] = T_ij

        if aleatorio:
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    dist = rng.randint(1, 10)
                    T_ij_matrix[i][j] = dist
                    T_ij_matrix[j][i] = dist
            for i in range(num_nodes):
                T_ij_matrix[i][i] = 0

        self.T_ij_matrix = T_ij_matrix

    # -----------------------
    # Model setup
    # -----------------------

    def setup_model(self):
        """Create Gurobi model and decision variables."""
        self.model = Model("BargeScheduling")

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
        L = self.Handling_time_minutes
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
        """Run the optimization."""
        if self.model is None:
            raise RuntimeError("Model not set up. Call setup_model(), set_objective(), add_constraints() first.")
        self.model.optimize()

    # -----------------------
    # Result printing helpers
    # -----------------------

    def print_results(self):
        """Print basic optimization results."""
        m = self.model
        if m is None or m.status != GRB.OPTIMAL:
            print("No optimal solution found. Status:", m.status if m is not None else "No model")
            return

        C = self.C_list
        N = self.N_list
        K = self.K_list
        K_b = self.K_b
        K_t = self.K_t
        Qk = self.Qk

        f_ck = self.f_ck
        x_ijk = self.x_ijk
        y_ijk = self.y_ijk
        z_ijk = self.z_ijk

        print("\n\nOptimal solution found!")
        print(f"Optimal objective value: {m.objVal:.2f}")

        # Container -> vehicle assignment
        print("\nContainer to Vehicle Assignments (f_ck)")
        print("========================================")
        print(f"Number of containers: {len(C)}")

        for c in C:
            for k in K:
                if f_ck[c, k].X > 0.5:
                    if k == K_t:
                        print(f"Container {c} is assigned to TRUCK (k={k})")
                    else:
                        print(f"Container {c} is assigned to BARGE {k}")

        # Barge routes
        print("\nBarge Travel Decisions (x_ijk)")
        print("================================")
        for k in K_b:
            for i in N:
                for j in N:
                    if i != j and x_ijk[i, j, k].X > 0.5:
                        print(f"Barge {k} travels from Terminal {i} to Terminal {j}")

        # Capacity use on arcs involving node 0
        print("\nCapacity usage on arcs involving node 0")
        print("========================================")
        for k in K_b:
            for j in N:
                if j == 0:
                    continue

                # j -> 0 (imports)
                if y_ijk[j, 0, k].X > 0.5:
                    assigned_quant = y_ijk[j, 0, k].X
                    available_quant = Qk[k]
                    print(
                        f"Barge {k}, arc {j} -> 0: "
                        f"{assigned_quant}/{available_quant} "
                        f"({assigned_quant / available_quant * 100:.2f}%)"
                    )

                # 0 -> j (exports)
                if z_ijk[0, j, k].X > 0.5:
                    assigned_quant = z_ijk[0, j, k].X
                    available_quant = Qk[k]
                    print(
                        f"Barge {k}, arc 0 -> {j}: "
                        f"{assigned_quant}/{available_quant} "
                        f"({assigned_quant / available_quant * 100:.2f}%)"
                    )

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
                "Release Time [min]": R_c[c],
                "Opening Time [min]": O_c[c],
                "Closing Time [min]": D_c[c],
                "Assigned Vehicle": assigned_label,
                "Sort Node": sort_node,
                "Sort Type": 0 if container_type == "Export" else 1
            })

        df = pd.DataFrame(container_data)
        df = df.sort_values(by=["Sort Node", "Sort Type"]).drop(columns=["Sort Node", "Sort Type"])

        print("\nContainer Table (Grouped by Node and Type)")
        print("==========================================")
        print(tabulate(df, headers="keys", tablefmt="grid"))

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
        print("\nNode Table")
        print("==========")
        print(tabulate(df, headers="keys", tablefmt="grid"))

    def print_barge_table(self):
        """
        Prints a table summarizing barge routes and capacity utilization per arc.
        """
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
                            "Utilization (%)": f"{utilization_percent:.2f}",
                        })

            if routes:
                df = pd.DataFrame(routes)
                print(f"\nBarge {k} Route & Capacity Usage")
                print("================================")
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
        self.setup_model()
        self.set_objective()
        self.add_constraints()
        self.solve()

        self.print_node_table()

        if self.model.status == GRB.OPTIMAL:
            self.print_results()
            self.print_container_table()
            self.print_barge_table()
            print(f"\nTotal containers: {self.C}")
            print(f"K_b (barges): {self.K_b}")
            print(f"K_t (truck): {self.K_t}")
            if with_plots:
                self.plot_barge_displacements()


# Optional quick test if you run MILP.py directly:
if __name__ == "__main__":
    milp = MILP_Algo(reduced=True)   # e.g. smaller instances
    milp.run(with_plots=True)
