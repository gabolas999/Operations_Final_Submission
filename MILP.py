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
            # run_name="MILP_Run",
            run_name="______",
            qk=[  # Barge capacities in TEU
                40,         # Barge 0
                30,         # Barge 1
                20,         # Barge 2
                15,         # Barge 3
            ],
            h_b=[  # Barge fixed costs in euros
                1600,      # Barge 0
                1500,      # Barge 1
                1100,      # Barge 2
                900,      # Barge 3
            ],
            seed=0,               # Random seed for reproducibility
            reduced=False,
            h_t_40=200,                 # 40ft container trucking cost in euros
            h_t_20=140,                 # 20ft container trucking cost in euros
            handling_time=1/6,              # Container handling time in hours
            C_range=(150, 175),              # (min, max) number of containers when reduced=False
            N_range=(20, 20),                 # (min, max) number of terminals when reduced=False

            Oc_range=(0, 24),             # (min, max) opening time in hours
            Oc_offset_range=(20, 70),      # (min_offset, max_offset) such that
                                            # Dc is drawn in [Oc + min_offset, Oc + max_offset]

            travel_time_long_range=(9, 13),   # (min, max) travel time between dryport and sea terminals in hours
            travel_angle = math.pi/8,             # angle sector for terminal placement
            travel_time_scale = 10,             # scale down travel times for better layout

            P40_range=(0.2, 0.22),              # (min, max) probability of 40ft container
            PExport_range=(0.05, 0.75),         # (min, max) probability of export
            C_range_reduced=(120, 120),           # (min, max) containers when reduced=True
            N_range_reduced=(6, 6),             # (min, max) terminals when reduced=True
            gamma=100,                          # penalty per sea terminal visit [euros]
            big_m=1000,                          # big-M
            enable_arc_elimination=True
    ):
        """
        Initialize the MILP optimizer.

        Parameters mirror GreedyAlgo so that both can be constructed in the same way.
        Time-related ranges and all internal time variables are in hours.
        """
        self.time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.file_name = f"{run_name}_{self.time}"

        os.makedirs("Storage_orig/Logs", exist_ok=True)
        os.makedirs("Storage_orig/Solutions", exist_ok=True)
        os.makedirs("Storage_orig/Settings", exist_ok=True)
        os.makedirs("Storage_orig/Figures", exist_ok=True)

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
        with open(f"Storage_orig/Settings/settings_{self.file_name}.toml", "w") as f:
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
        self.travel_time_scale = travel_time_scale
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
        # self.generate_travel_times()
        self.generate_travel_times_fazi_case_study()

        self.enable_arc_elimination = enable_arc_elimination
        self.valid_arcs = []  # To store filtered arcs

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
                Rc_hr = rng.randint(0, 12)  # release window for exports
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
            r = rng.randint(long_min, long_max) / self.travel_time_scale  # scaled down for better layout

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
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2) * self.travel_time_scale  # scale back to time
                T[i][j] = int(dist)

        self.T_ij_matrix = T


    def generate_travel_times_fazi_case_study(self):
        """
        Generates T_ij matrix and node coordinates based strictly on 
        Fazi et al. (2015) Case Study (Table 2 & Section 3.3).
        
        Logic:
        - Node 0 is the Inland Terminal (Veghel).
        - Remaining nodes are distributed among 3 Sea Clusters:
          1. Maasvlakte (Rotterdam West)
          2. City Terminal (Rotterdam East)
          3. Antwerp
        
        Travel Times (Hours):
        - Within same cluster (different quays): 1 h
        - Veghel <-> Maasvlakte/City: 11 h
        - Veghel <-> Antwerp: 13 h
        - Maasvlakte <-> City: 4 h
        - Antwerp <-> Rotterdam (Maas/City): 16 h
        """
        import math
        import random

        num_nodes = self.N
        
        # ---------------------------------------------------------
        # 1. Assign Nodes to Clusters
        # ---------------------------------------------------------
        # Cluster IDs:
        # 0: Dry Port (Veghel)
        # 1: Maasvlakte
        # 2: City Terminal
        # 3: Antwerp
        
        node_cluster = {0: 0}
        
        # Distribute remaining nodes (quays) among the 3 sea clusters
        # We assume a balanced distribution for verification
        sea_clusters = [1, 2, 3]
        for i in range(1, num_nodes):
            # Round robin assignment: 1, 2, 3, 1, 2...
            node_cluster[i] = sea_clusters[(i - 1) % len(sea_clusters)]

        # ---------------------------------------------------------
        # 2. Define Inter-Cluster Travel Times (Table 2 of Paper)
        # ---------------------------------------------------------
        # (Cluster A, Cluster B) -> Hours
        cluster_dist = {
            # Dry Port Connections
            (0, 1): 11.0, # Veghel - Maasvlakte
            (0, 2): 11.0, # Veghel - City
            (0, 3): 13.0, # Veghel - Antwerp
            
            # Inter-Sea-Terminal Connections
            (1, 2): 4.0,  # Maasvlakte - City
            (1, 3): 16.0, # Maasvlakte - Antwerp
            (2, 3): 16.0, # City - Antwerp
        }

        # ---------------------------------------------------------
        # 3. Build T_ij Matrix
        # ---------------------------------------------------------
        T = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    T[i][j] = 0.0
                    continue
                
                c_i = node_cluster[i]
                c_j = node_cluster[j]
                
                if c_i == c_j:
                    # Assumption (i): "Within a cluster all quays are equidistant (1 h)"
                    T[i][j] = 1.0
                else:
                    # Look up inter-cluster distance (symmetric)
                    dist = cluster_dist.get((c_i, c_j))
                    if dist is None:
                        dist = cluster_dist.get((c_j, c_i))
                    T[i][j] = dist

        self.T_ij_matrix = T

        # ---------------------------------------------------------
        # 4. Generate Coordinates for Plotting (Schematic Map)
        # ---------------------------------------------------------
        # We manually define center points for clusters to match geography
        # (Relative positions: Veghel East, Maasvlakte West, Antwerp South)
        cluster_centers = {
            0: (12.0, 0.0),   # Veghel (Right/East)
            1: (-8.0, 5.0),   # Maasvlakte (Top-Left/North-West)
            2: (-2.0, 3.0),   # City (Mid-Left)
            3: (-6.0, -6.0)   # Antwerp (Bottom-Left/South)
        }
        
        node_xy = []
        rng = random.Random(self.seed)
        
        for i in range(num_nodes):
            c_id = node_cluster[i]
            cx, cy = cluster_centers[c_id]
            
            if i == 0:
                # Dry port is fixed
                node_xy.append((cx, cy))
            else:
                # Add small random jitter so quays in same cluster don't overlap
                # Radius = 1.0 to represent the "1 hour" proximity visually
                angle = rng.uniform(0, 2 * math.pi)
                r = rng.uniform(0.5, 1.5)
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                node_xy.append((x, y))

        self.node_xy = node_xy

    def plot_topography_preview(self):
        """
        Plots a professional, non-overlapping topography for a LaTeX report.
        - Uses structured circular placement for quays to prevent overlap.
        - Completely removes all axes, ticks, and titles.
        - Highlights clusters with soft colored halos.
        - Places node IDs clearly inside the markers.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not hasattr(self, 'node_xy') or not self.node_xy:
            print("No coordinates found. Please run the generator first.")
            return

        # 1. Professional Color Palette & Naming
        colors = {0: '#d63031', 1: '#0984e3', 2: '#00b894', 3: '#e17055'}
        cluster_names = {0: "VEGHEL (INLAND)", 1: "MAASVLAKTE", 2: "CITY TERMINAL", 3: "ANTWERP"}

        # Geographic centers
        cluster_centers = {
            0: (12.0, 0.0),
            1: (-8.0, 5.0),
            2: (-2.0, 3.0),
            3: (-6.0, -6.0)
        }

        # RE-CALCULATE COORDINATES FOR VISUAL CLARITY
        # This ensures no overlap regardless of what the generator did
        visual_coords = {0: cluster_centers[0]}
        nodes_per_cluster = {1: [], 2: [], 3: []}
        for i in range(1, self.N):
            c_id = ((i - 1) % 3) + 1
            nodes_per_cluster[c_id].append(i)

        # Distribute quays in a circle around their center
        radius = 1.8  # Sufficient distance to prevent overlap of large markers
        for c_id, node_list in nodes_per_cluster.items():
            cx, cy = cluster_centers[c_id]
            n = len(node_list)
            for idx, node_idx in enumerate(node_list):
                angle = 2 * np.pi * idx / n
                visual_coords[node_idx] = (cx + radius * np.cos(angle), cy + radius * np.sin(angle))

        # Setup Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor("white")

        # 2. Draw Cluster Halos (Visual grouping)
        for c_id, (cx, cy) in cluster_centers.items():
            if c_id == 0: continue
            halo = plt.Circle((cx, cy), radius * 1.8, color=colors[c_id], alpha=0.08, lw=0, zorder=1)
            ax.add_patch(halo)

        # 3. Draw Intra-cluster connections (Lines between quays)
        for c_id, node_list in nodes_per_cluster.items():
            for i in range(len(node_list)):
                for j in range(i + 1, len(node_list)):
                    p1 = visual_coords[node_list[i]]
                    p2 = visual_coords[node_list[j]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                            color='#b2bec3', linestyle='-', linewidth=1, alpha=0.4, zorder=2)

        # 4. Plot Nodes and Labels
        for i in range(self.N):
            c_id = 0 if i == 0 else ((i - 1) % 3) + 1
            x, y = visual_coords[i]

            marker = 's' if i == 0 else 'o'
            size = 600 if i == 0 else 450  # Large nodes for visibility

            # Draw the node
            ax.scatter(x, y, s=size, marker=marker, color=colors[c_id],
                       edgecolor='black', linewidth=1.5, zorder=10)

            # Node ID Label (Bold white text inside the node)
            ax.text(x, y, str(i), ha='center', va='center',
                    fontsize=10, color='white', fontweight='bold', zorder=11)

        # 5. Cluster Region Names (Positioned to avoid nodes)
        for c_id, (cx, cy) in cluster_centers.items():
            # Place name above or below the cluster 'halo'
            v_offset = 3.8 if cy >= 0 else -4.2
            ax.text(cx, cy + v_offset, cluster_names[c_id],
                    color=colors[c_id], ha='center', va='center',
                    fontsize=12, fontweight='black', zorder=12)

        # Final Cleaning
        ax.axis('off')  # Remove axes, ticks, and frame
        ax.set_aspect('equal', 'datalim')

        # Padding to ensure cluster names stay inside the frame
        plt.margins(0.2)
        plt.tight_layout()

        # Save for LaTeX
        outfile = f"Storage_orig/Figures/topography_preview_{self.file_name}.pdf"
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Topography preview saved to: {outfile}")

    # -----------------------
    # Model setup
    # -----------------------

    def setup_model(self):
        """Create Gurobi model and decision variables."""
        self.model = Model("BargeScheduling")

        self.model.Params.MIPFocus = 2   # focus on improving feasible solutions quickly
        # Options:
        # 0 = balanced (default)
        # 1 = feasibility
        # 2 = optimality
        # 3 = bound improvement

        # --------------- Gurobi configuration ---------------


        # Log file with timestamp to avoid overwriting
        self.model.Params.LogFile = f"Storage_orig/Logs/log______{self.file_name}.log"

        # Stopping criterion: 1% relative MIP gap
        self.model.Params.MIPGap = 0.01

        # Re-enable Gurobi's own console log
        self.model.Params.OutputFlag = 1

        self.model.Params.TimeLimit = 1000

        # (Optional) If you want fewer log lines, uncomment this:
        # self.model.Params.DisplayInterval = 5  # print progress every 5 seconds
        # ----------------------------------------------------

        C = self.C_list
        N = self.N_list
        K = self.K_list
        K_b = self.K_b

        # Arc elimination logic
        if self.enable_arc_elimination:
            # Tighten M to the maximum possible time horizon
            self.M = max(self.D_c) + max(max(row) for row in self.T_ij_matrix) + 10

            # Pre-calculate Earliest Start and Latest Deadline per Terminal
            node_deadlines = {j: max([self.D_c[c] for c in C if self.Z_cj[c][j] == 1] or [0]) for j in N}
            node_starts = {i: min([self.O_c[c] for c in C if self.Z_cj[c][i] == 1] or [0]) for i in N}

            self.valid_arcs = []
            for k in K_b:
                for i in N:
                    for j in N:
                        if i == j: continue

                        # Pruning Rule: Can we get from i to j and process
                        # at least 1 container before the terminal closes?
                        # (We only prune for sea terminals j > 0)
                        if j > 0:
                            travel = self.T_ij_matrix[i][j]
                            if node_starts[i] + travel + self.Handling_time > node_deadlines[j]:
                                continue  # SKIP THIS ARC

                        self.valid_arcs.append((i, j, k))
            print(f"Arc Elimination Active: Created {len(self.valid_arcs)} arcs.")
        else:
            # Standard "Full" arc set
            self.valid_arcs = [(i, j, k) for i in N for j in N for k in K_b if i != j]
            self.M = 1000  # Use default Big-M

        # Define variables using valid_arcs
        self.f_ck = self.model.addVars(C, self.K_list, vtype=GRB.BINARY, name="f_ck")
        self.x_ijk = self.model.addVars(self.valid_arcs, vtype=GRB.BINARY, name="x_ijk")
        self.y_ijk = self.model.addVars(self.valid_arcs, vtype=GRB.INTEGER, lb=0, name="y_ijk")
        self.z_ijk = self.model.addVars(self.valid_arcs, vtype=GRB.INTEGER, lb=0, name="z_ijk")

        # Remaining variables (p, d, t) are node-based, not arc-based
        self.p_jk = self.model.addVars(N, K_b, vtype=GRB.INTEGER, lb=0, name="p_jk")
        self.d_jk = self.model.addVars(N, K_b, vtype=GRB.INTEGER, lb=0, name="d_jk")
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
                     for k in K_b for i in N for j in N if j != 0 if i != j)
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
                    quicksum(self.x_ijk[i, j, k] for j in N if (i, j, k) in self.x_ijk) -
                    quicksum(self.x_ijk[j, i, k] for j in N if (j, i, k) in self.x_ijk) == 0,
                    name=f"Flow_Cons_{i}_{k}"
                )

        # 3. Each barge leaves dryport (0) at most once
        for k in K_b:
            m.addConstr(
                quicksum(self.x_ijk[0, j, k] for j in N if (0, j, k) in self.x_ijk) <= 1,
                name=f"Depart_{k}"
            )

        # # 4. No self-loops (i -> i)
        # for i in N:
        #     for k in K_b:
        #         m.addConstr(x_ijk[i, i, k] == 0, name=f"No_Self_Loop_{i}_{k}")

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
                    quicksum(self.y_ijk[j, i, k] for i in N if (j, i, k) in self.y_ijk) -
                    quicksum(self.y_ijk[i, j, k] for i in N if (i, j, k) in self.y_ijk) == self.p_jk[j, k]
                )
                m.addConstr(
                    quicksum(self.z_ijk[i, j, k] for i in N if (i, j, k) in self.z_ijk) -
                    quicksum(self.z_ijk[j, i, k] for i in N if (j, i, k) in self.z_ijk) == self.d_jk[j, k]
                )

        # 9. Barge trip capacity constraint
        for (i, j, k) in self.valid_arcs:
            m.addConstr(self.y_ijk[i, j, k] + self.z_ijk[i, j, k] <= self.Qk[k] * self.x_ijk[i, j, k])

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
            if include_time_constraints:
                for (i, j, k) in self.valid_arcs:
                    if j == 0: continue  # Propagate to sea terminals only
                    handling_term = quicksum(self.Handling_time * self.Z_cj[c][i] * self.f_ck[c, k] for c in C)
                    m.addConstr(self.t_jk[j, k] >= self.t_jk[i, k] + handling_term + self.T_ij_matrix[i][j] - (
                                1 - self.x_ijk[i, j, k]) * self.M)
                    m.addConstr(self.t_jk[j, k] <= self.t_jk[i, k] + handling_term + self.T_ij_matrix[i][j] + (
                                1 - self.x_ijk[i, j, k]) * self.M)

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
            # 2. Flow conservation for barges at each node              #
    # checked
            # 3. Each barge leaves dryport (0) at most once             #
    # checked
            # 4. No self-loops (i -> i)                                 #
    # checked           
            # 5. Import quantity at terminal j for barge k              #
    # checked           
            # 6. Export quantity at terminal j for barge k              #
    # checked           
            # 7. Flow balance for import quantities at terminal j for barge k           #
    # checked
            # 8. Flow balance for export quantities at terminal j for barge k           #
    # checked
            # 9. Barge trip capacity constraint                                         #                                     
    # almost checked
            # 10. Export containers: departure time at dryport >= release time
    # box plot looking thing. 
            # 11 & 12. Time propagation along arcs with handling time at arrival
    # boc plot looking thing.
            # 13. Export container service cannot start before opening time
    # box plot looking thing
            # 14. All containers must be served before closing time
    # box plot looking thing. 

    def get_solution_dict(self):
        """
        Extracts the current solution (variable names and values) 
        to pass to the next iteration.
        Only stores non-zero values to save memory (Sparse approach).
        """
        if self.model is None or self.model.status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            return None
        
        sol_dict = {}
        for v in self.model.getVars():
            # Only save binary/integer vars that are non-zero (approx > 0.5)
            # For continuous variables, you might want to save them too, 
            # but usually integers are the most important for MIP start.
            if v.x > 0.0001: 
                sol_dict[v.VarName] = v.x
        return sol_dict

    def apply_warm_start(self, sol_input):
        """
        Injects a previous solution into the current model as a MIP Start.
        Can accept:
        1. A dictionary (from get_solution_dict)
        2. A file path string (path to a .sol or .mst file)
        """
        if sol_input is None:
            return

        # CASE A: Input is a Dictionary (In-memory transfer)
        if isinstance(sol_input, dict):
            print(f"   -> Applying Warm Start from Dictionary ({len(sol_input)} vars)...")
            count = 0
            for v in self.model.getVars():
                if v.VarName in sol_input:
                    v.Start = sol_input[v.VarName]
                    count += 1
            print(f"   -> Warm Start set for {count} variables.")

        # CASE B: Input is a File Path (Loading from .sol file)
        elif isinstance(sol_input, str):
            if os.path.exists(sol_input):
                print(f"   -> Applying Warm Start from File: {sol_input}")
                # Gurobi automatically reads .sol files as MIP Starts
                self.model.read(sol_input)
            else:
                print(f"   Warning: Warm start file not found: {sol_input}")

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
            for k in K_b for i in N for j in N if j != 0 if i != j
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

    def print_economic_validation(self):
        """
        Validates economic rationality.
        Exposes the comparison between Barge costs and the Trucking alternative.
        """
        import pandas as pd
        from tabulate import tabulate

        m = self.model
        if m is None or m.status != 2:  # 2 is GRB.OPTIMAL
            print("No optimal solution available for economic validation.")
            return

        print("\n\n############################################################")
        print("      Economic Rationality Analysis (Barge vs. Truck)")
        print("############################################################")

        # --- [A] System Parameters (Trucking Costs) ---
        print(f"\n[A] Road Transport Cost Parameters:")
        print(f"    - Cost for 20ft Truck (1 TEU): {self.H_t_20:>8.2f} €")
        print(f"    - Cost for 40ft Truck (2 TEU): {self.H_t_40:>8.2f} €")

        # --- [B] Analyze Optimization Results ---
        print(f"\n[B] Performance Comparison per Barge Voyage:")

        K_b = self.K_b
        H_b = self.H_b
        H_T = self.H_T
        W_c = self.W_c
        T = self.T_ij_matrix
        Gamma = self.Gamma

        x_ijk = self.x_ijk
        f_ck = self.f_ck
        C_list = self.C_list
        N_list = self.N_list

        economic_data = []

        for k in K_b:
            # Identify if barge k is active
            active_barge = sum(x_ijk[0, j, k].X for j in N_list if j != 0) > 0.5
            if not active_barge:
                continue

            # --- 1. Barge Costs (H_b^k + C_var_k) ---
            fixed_cost = H_b[k] if k in H_b else (H_b[int(k)] if isinstance(H_b, list) else 0)
            travel_cost = sum(T[i][j] * x_ijk[i, j, k].X for i in N_list for j in N_list if i != j)
            port_cost = sum(Gamma * x_ijk[i, j, k].X for i in N_list for j in N_list if j != 0 and i != j)

            total_barge_cost = fixed_cost + travel_cost + port_cost

            # --- 2. Cargo Mix ---
            assigned_containers = [c for c in C_list if f_ck[c, k].X > 0.5]
            if not assigned_containers:
                continue

            n_20 = sum(1 for c in assigned_containers if W_c[c] == 1)
            n_40 = sum(1 for c in assigned_containers if W_c[c] == 2)
            q_k = sum(W_c[c] for c in assigned_containers)
            alpha = (n_20 * 1) / q_k if q_k > 0 else 0

            # --- 3. Truck Alternative Details ---
            # Every container in the barge would require 1 truck trip if moved by road
            truck_trips_avoided = n_20 + n_40
            total_avoided_truck_cost = sum(H_T[c] for c in assigned_containers)
            avg_truck_cost_per_teu = total_avoided_truck_cost / q_k

            # --- 4. Comparative Metrics ---
            avg_barge_cost_per_teu = total_barge_cost / q_k
            net_savings = total_avoided_truck_cost - total_barge_cost

            # Break-even volume: q_k >= Barge_Total / Avg_Truck_Unit_Cost
            min_q_k = total_barge_cost / avg_truck_cost_per_teu if avg_truck_cost_per_teu > 0 else 0

            is_rational = q_k >= (min_q_k - 0.001)

            economic_data.append({
                "Barge": k,
                "TEU (q_k)": f"{q_k}",
                "Mix (20/40)": f"{n_20}/{n_40}",
                "Trucks Removed": f"{truck_trips_avoided}",
                "Truck Cost (€)": f"{total_avoided_truck_cost:,.0f}",
                "Barge Cost (€)": f"{total_barge_cost:,.0f}",
                "Net Savings (€)": f"{net_savings:,.0f}",
                "Truck €/TEU": f"{avg_truck_cost_per_teu:.2f}",
                "Barge €/TEU": f"{avg_barge_cost_per_teu:.2f}",
                "Min q_k": f"{min_q_k:.1f}",
                "Rational?": "YES" if is_rational else "NO"
            })

        if not economic_data:
            print("  No barges were used. All cargo was handled by road transport.")
        else:
            df = pd.DataFrame(economic_data)
            print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))

            # Aggregate stats
            total_saved = sum(float(row["Net Savings (€)"].replace(',', '')) for row in economic_data)
            total_trucks = sum(int(row["Trucks Removed"]) for row in economic_data)

            print(f"\n  SUMMARY STATISTICS:")
            print(f"  - Total Road Trips Avoided: {total_trucks} trucks")
            print(f"  - Total Network Cost Savings: {total_saved:,.2f} €")
            print(f"  - Break-even Note: 'Min q_k' is the TEU volume needed for the barge to beat")
            print(f"    the road alternative, given the specific 20ft/40ft mix (\u03b1) of that voyage.")


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
    def plot_barge_solution_map_report_Without_containers(self):
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
        multi = 3.0
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
                
                # sign_x, sign_y = compute_signs(P0, P2, anchor_x, anchor_y)

                # width_scaled = 0.09
                # plotter = ContainerPlotter(width=width_scaled, x=anchor_x, y=anchor_y, sign_x=sign_x, sign_y=sign_y)
       
                # total_capacity = self.Qk[k]
                # self._draw_segment_stack(ax, plotter, i, j, k,
                                        #  total_width=5, max_rows=total_capacity / 5)



            # legend entry for this barge
            line, = ax.plot([], [], color="black", linewidth=2.5, linestyle=linestyle)
            legend_lines.append(line)
            legend_labels.append(f"Barge {k}")
            # ==========================================
            # Add Import / Export container legend icons
            # ==========================================


        # # Small representative container size
        # legend_w = width_scaled
        # legend_h = legend_w * 8.6 / 20

        # import_patch = Rectangle(
        #     (0, 0),
        #     legend_w,
        #     legend_h,
        #     facecolor="#00A63C",
        #     edgecolor="#006E28",
        #     linewidth=1.5
        # )
        # export_patch = Rectangle(
        #     (0, 0),
        #     legend_w,
        #     legend_h,
        #     facecolor="#FF6F00",
        #     edgecolor="#B23E00",
        #     linewidth=1.5
        # )

        # legend_lines.extend([export_patch, import_patch])
        # legend_labels.extend(["Export", "Import"])

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
        fontsize=15,
        handlelength= 4.2,      # default is 2 — increase for longer patterns
        handletextpad=0.8,   # spacing between line and text
    )

        plt.tight_layout()
        plt.savefig(f"Storage_orig/Figures/solution_map{self.file_name}_no_cont.pdf")

    def plot_barge_solution_map_report_ONLY_NODES(self):
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
        # Prepare figure
        # --------------------------
        # fig, ax = plt.subplots(figsize=(12, 8))
        fig, ax = plt.subplots(figsize=((12/1.4), (8/1.4)))
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
        # Draw dotted connections between all node pairs
        # --------------------------
        for i in N:
            xi, yi = node_xy[i]

            for j in N:
                if j <= i:
                    continue  # avoid self-loops and duplicate lines

                xj, yj = node_xy[j]

                curve = np.array([
                    [xi, yi],
                    [xj, yj],
                ])

                ax.plot(
                    curve[:, 0], curve[:, 1],
                    color="black",
                    linewidth=1.2,
                    linestyle=":",
                    alpha=1.0,
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

        plt.tight_layout()
        plt.savefig(f"Storage_orig/Figures/solution_map{self.file_name}_simple.pdf")

    def plot_barge_solution_map_report_3(self, curvature_multiplier=3.0, size_scale=1.0,
                                         node_offsets=None, coordinate_scaling=1.0,
                                         sol_file_path=None):
        import re
        import collections
        import numpy as np
        import matplotlib.pyplot as plt

        def bezier_quad(P0, P1, P2, t):
            return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2

        def boxes_intersect(b1, b2, padding=0.0):
            """Returns True if two bounding boxes overlap."""
            return not (b1['xmax'] + padding <= b2['xmin'] or
                        b1['xmin'] - padding >= b2['xmax'] or
                        b1['ymax'] + padding <= b2['ymin'] or
                        b1['ymin'] - padding >= b2['ymax'])

        # Helper class to spoof variable attributes when loading from .sol
        class MockVar:
            def __init__(self, val):
                self.X = float(val)

        # --------------------------
        # 1. Extract Values (x, y, z)
        # --------------------------
        x_vals, y_vals, z_vals = {}, {}, {}

        if sol_file_path:
            try:
                with open(sol_file_path, 'r') as f:
                    for line in f:
                        if line.startswith('#') or not line.strip(): continue
                        parts = line.split()
                        if len(parts) >= 2:
                            name, val = parts[0], float(parts[1])
                            mx = re.search(r"x_ijk\[(\d+),(\d+),(\d+)\]", name)
                            my = re.search(r"y_ijk\[(\d+),(\d+),(\d+)\]", name)
                            mz = re.search(r"z_ijk\[(\d+),(\d+),(\d+)\]", name)

                            if mx: x_vals[tuple(map(int, mx.groups()))] = MockVar(val)
                            if my: y_vals[tuple(map(int, my.groups()))] = MockVar(val)
                            if mz: z_vals[tuple(map(int, mz.groups()))] = MockVar(val)

                if not x_vals:
                    print(f"Warning: No variables matching 'x_ijk' found in {sol_file_path}")

                old_x, old_y, old_z = self.x_ijk, self.y_ijk, self.z_ijk
                self.x_ijk, self.y_ijk, self.z_ijk = x_vals, y_vals, z_vals
                print(f"Loaded solution data from {sol_file_path}")
            except Exception as e:
                print(f"Error reading .sol file: {e}")
                return
        else:
            if self.model is None:
                print("No active model found. Please provide a sol_file_path.")
                return

        # --------------------------
        # 2. Node Geometry & Scaling
        # --------------------------
        node_xy = {i: np.array(pos, dtype=float) for i, pos in enumerate(self.node_xy)}
        N = self.N_list
        K_b = self.K_b

        if coordinate_scaling != 1.0:
            center = np.array(list(node_xy.values())).mean(axis=0)
            for k in node_xy:
                node_xy[k] = center + (node_xy[k] - center) * coordinate_scaling

        if node_offsets:
            for n_id, offset in node_offsets.items():
                if n_id in node_xy: node_xy[n_id] += np.array(offset)

        xs = [pos[0] for pos in node_xy.values()]
        ys = [pos[1] for pos in node_xy.values()]
        graph_size = max(max(xs) - min(xs), max(ys) - min(ys))
        if graph_size == 0: graph_size = 1.0

        # --------------------------
        # 3. Directed Parallel Lanes Engine
        # --------------------------
        active_arcs = []
        used_barges = set()

        for k in K_b:
            for i in N:
                for j in N:
                    if i != j and (i, j, k) in self.x_ijk and self.x_ijk[i, j, k].X > 0.5:
                        active_arcs.append((i, j, k))
                        used_barges.add(k)

        if not active_arcs:
            print("No barge movements found.")
            if sol_file_path: self.x_ijk, self.y_ijk, self.z_ijk = old_x, old_y, old_z
            return

        # Group by EXACT directed route (A -> B)
        directed_groups = collections.defaultdict(list)
        for (i, j, k) in active_arcs:
            directed_groups[(i, j)].append(k)

        arc_curves = {}
        for (i, j), barges in directed_groups.items():
            P0, P2 = node_xy[i], node_xy[j]
            d = P2 - P0
            L = np.linalg.norm(d)
            if L == 0: continue

            n_vec = np.array([-d[1], d[0]]) / L  # Left-hand normal

            for lane_idx, k in enumerate(barges):
                lane_offset = (0.15 + 0.12 * lane_idx) * min(curvature_multiplier, 2.5)
                P1 = (P0 + P2) / 2 + lane_offset * L * n_vec

                # Generate exact points for the curve
                curve_pts = np.array([bezier_quad(P0, P1, P2, t) for t in np.linspace(0, 1, 60)])
                arc_curves[(i, j, k)] = {
                    'P0': P0, 'P1': P1, 'P2': P2, 'n_vec': n_vec, 'curve_pts': curve_pts
                }

        # --------------------------
        # 4. Plotting Setup
        # --------------------------
        line_styles = ["-", "--", ":", "-.", (0, (1, 1, 2, 1)), (0, (8, 4))] * 10
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        occupied_boxes = []

        # --- PHASE 1: DRAW ALL NODES AND LINES FIRST ---
        node_radius = graph_size * 0.08
        for j in N:
            x, y = node_xy[j]
            ax.scatter(x, y, s=(800 if j == 0 else 400), marker=("s" if j == 0 else "o"),
                       facecolor="white", edgecolor="black", zorder=4)
            ax.text(x, y, f"{j}", ha="center", va="center", fontsize=9, zorder=5)
            # Register Node Obstacle
            occupied_boxes.append({
                'xmin': x - node_radius, 'xmax': x + node_radius,
                'ymin': y - node_radius, 'ymax': y + node_radius
            })

        legend_lines, legend_labels = [], []
        barges_in_legend = set()

        line_thickness = graph_size * 0.015
        for k in sorted(list(used_barges)):
            linestyle = line_styles[k % len(line_styles)]
            barge_arcs = [(i, j, k) for (i, j, kk) in active_arcs if kk == k]
            alphas = np.linspace(1.0, 0.35, len(barge_arcs)) if len(barge_arcs) > 1 else [1.0]

            for s, (i, j, k) in enumerate(barge_arcs):
                curve = arc_curves[(i, j, k)]['curve_pts']
                ax.plot(curve[:, 0], curve[:, 1], color="black", linewidth=2.5,
                        linestyle=linestyle, alpha=float(alphas[s]), zorder=2)

                # Register Line as Obstacle (Sample points to create a collision boundary)
                for pt in curve[::3]:  # Sample every 3rd point for speed
                    occupied_boxes.append({
                        'xmin': pt[0] - line_thickness, 'xmax': pt[0] + line_thickness,
                        'ymin': pt[1] - line_thickness, 'ymax': pt[1] + line_thickness
                    })

            if k not in barges_in_legend:
                line, = ax.plot([], [], color="black", linewidth=2.5, linestyle=linestyle)
                legend_lines.append(line)
                legend_labels.append(f"Barge {k}")
                barges_in_legend.add(k)

        # --- PHASE 2: CALCULATE STACK POSITIONS ---
        cw = 0.16 * size_scale  # Container slot width
        ch = cw * 8.6 / 20  # Container slot height
        base_standoff = graph_size * 0.04 * size_scale  # Minimum distance from line

        stacks_to_draw = []

        for k in sorted(list(used_barges)):
            barge_arcs = [(i, j, k) for (i, j, kk) in active_arcs if kk == k]

            for (i, j, k) in barge_arcs:
                data = arc_curves[(i, j, k)]
                P0, P1, P2 = data['P0'], data['P1'], data['P2']

                # Calculate required stack size
                cap = self.Qk[k] if hasattr(self, 'Qk') else 40
                max_rows = max(1, cap / 5)
                max_cols = 4  # total_width=5 implies 4 data columns
                stack_w = max_cols * cw + (cw * 0.5)
                stack_h = max_rows * ch + (ch * 0.5)

                best_anchor = None
                best_sx, best_sy = 1, 1
                best_bbox = None
                best_score = -float('inf')

                # Sweep outward from the line to find clear space
                for push_mult in [1.0, 1.5, 2.5, 4.0, 6.0]:
                    spot_found_at_this_level = False

                    # Sweep along the curve
                    for t_cand in np.linspace(0.15, 0.85, 30):
                        peak = bezier_quad(P0, P1, P2, t_cand)

                        # Calculate outward normal at this specific t
                        T = 2 * (1 - t_cand) * (P1 - P0) + 2 * t_cand * (P2 - P1)
                        T_norm = np.linalg.norm(T)
                        n_curve = np.array([-T[1], T[0]]) / T_norm if T_norm > 0 else data['n_vec']

                        # Proposed anchor point (pushed away from line)
                        anchor = peak + (base_standoff * push_mult) * n_curve

                        # Determine growth direction (away from center of arc)
                        sx = 1 if anchor[0] >= (P0[0] + P2[0]) / 2 else -1
                        sy = 1 if anchor[1] >= (P0[1] + P2[1]) / 2 else -1

                        # Compute exact Bounding Box depending on drawing direction
                        bbox = {
                            'xmin': anchor[0] if sx == 1 else anchor[0] - stack_w,
                            'xmax': anchor[0] + stack_w if sx == 1 else anchor[0],
                            'ymin': anchor[1] if sy == 1 else anchor[1] - stack_h,
                            'ymax': anchor[1] + stack_h if sy == 1 else anchor[1]
                        }

                        # Check collision against ALL nodes, lines, and previous stacks
                        padding = 0.02 * size_scale
                        collision = any(boxes_intersect(bbox, ob, padding) for ob in occupied_boxes)

                        if not collision:
                            # Score favors being closer to the line (lower push_mult) and closer to middle (t=0.5)
                            score = - (push_mult * 10) - (abs(t_cand - 0.5) * 5)
                            if score > best_score:
                                best_score = score
                                best_anchor = anchor
                                best_sx, best_sy = sx, sy
                                best_bbox = bbox
                                spot_found_at_this_level = True

                    if spot_found_at_this_level:
                        break  # Found the closest clear orbit, stop pushing out

                # Fallback if extremely crowded
                if best_anchor is None:
                    best_anchor = bezier_quad(P0, P1, P2, 0.5) + (base_standoff * 2) * data['n_vec']
                    best_sx, best_sy = 1, 1
                    best_bbox = {'xmin': best_anchor[0], 'xmax': best_anchor[0] + stack_w,
                                 'ymin': best_anchor[1], 'ymax': best_anchor[1] + stack_h}

                # Register the claimed spot so next stacks avoid it
                occupied_boxes.append(best_bbox)
                stacks_to_draw.append({
                    'anchor': best_anchor, 'sx': best_sx, 'sy': best_sy,
                    'i': i, 'j': j, 'k': k, 'cap': cap
                })

        # --- PHASE 3: DRAW ALL STACKS ---
        for st in stacks_to_draw:
            plotter = ContainerPlotter(width=cw, x=st['anchor'][0], y=st['anchor'][1],
                                       sign_x=st['sx'], sign_y=st['sy'])
            self._draw_segment_stack(ax, plotter, st['i'], st['j'], st['k'],
                                     total_width=5, max_rows=st['cap'] / 5)

        # Restore original variables if loaded from file
        if sol_file_path:
            self.x_ijk, self.y_ijk, self.z_ijk = old_x, old_y, old_z

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([]);
        ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.legend(legend_lines, legend_labels, loc="upper right", frameon=True, fontsize=12)

        plt.tight_layout()
        plt.savefig(f"Storage_orig/Figures/solution_map{self.file_name}.pdf")


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

    def plot_time_windows(self, row_spacing: float = 1.0):
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
        E, I = set(self.E), set(self.I)
        R_c, O_c, D_c, Z_cj = self.R_c, self.O_c, self.D_c, self.Z_cj
        K, K_b = self.K_list, set(self.K_b)
        f_ck, t_jk = self.f_ck, self.t_jk

        assigned_barge = {}
        service_time_c = {}
        for c in C:
            assigned_k = next((k for k in K if f_ck[c, k].X > 0.5), None)
            if assigned_k is not None and assigned_k in K_b:
                try:
                    j = Z_cj[c].index(1)
                except:
                    j = None
                st = t_jk[j, assigned_k].X if j is not None else None
                assigned_barge[c], service_time_c[c] = assigned_k, st
            else:
                assigned_barge[c], service_time_c[c] = None, None

        # --------------------------
        # Build ordered rows
        # --------------------------
        def type_order(c):
            return 0 if c in E else 1

        barge_rows = []
        for k in sorted(K_b):
            conts = [c for c in C if assigned_barge.get(c) == k]
            barge_rows.extend(
                sorted(conts, key=lambda c: (service_time_c[c] is None, service_time_c[c], type_order(c), c)))

        no_barge = [c for c in C if assigned_barge.get(c) is None]
        no_barge_sorted = sorted(no_barge,
                                 key=lambda c: (type_order(c), service_time_c[c] is None, service_time_c[c], c))

        HEADER = "__HEADER__"
        container_rows = [HEADER] + barge_rows + no_barge_sorted
        n_rows = len(container_rows)

        # --------------------------
        # Figure Dimensions
        # --------------------------
        fig_width = 5.8  # Slightly reduced width
        base_height_per_row = 0.5  # Reduced to fit better on a page
        fig_height = max(3.5, min(base_height_per_row * n_rows, 15.0))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # --------------------------
        # Plot rows
        # --------------------------
        for idx, c in enumerate(container_rows):
            if c == HEADER: continue

            y = idx  # Integer-based Y makes controlling vertical space easier
            color, color_dark = ("#FF6F00", "#B23E00") if c in E else ("#00A63C", "#006E28")
            R, O, D = R_c[c], O_c[c], D_c[c]

            ax.hlines(y, O, D, colors=color_dark, linewidth=2, zorder=2)
            ax.scatter(R, y, marker="s", s=25, facecolor="white", edgecolor="black", linewidth=0.8, zorder=3)
            ax.scatter(O, y, marker=">", s=40, facecolor=color, edgecolor=color_dark, zorder=3)
            ax.scatter(D, y, marker="<", s=40, facecolor=color, edgecolor=color_dark, zorder=3)

            st = service_time_c.get(c)
            if st is not None:
                ax.scatter(st, y, marker="2", s=75, facecolor="black", linewidth=1.5, zorder=4)

        # --------------------------
        # Axis formatting
        # --------------------------
        # Fixed the span calculation error
        min_o = min(O_c[c] for c in C)
        max_d = max(D_c[c] for c in C)
        span = max_d - min_o
        ax.set_xlim(min_o - 0.05 * span, max_d + 0.05 * span)

        # Labels formatting
        k_width = max(1, max((len(str(k)) for k in K_b), default=1))
        c_width = max(1, max(len(str(c)) for c in C))

        ylabels = []
        for idx, c in enumerate(container_rows):
            if c == HEADER:
                ylabels.append(r"$\bf{Barge \;\;\;\; Cont}$")
                continue
            k = assigned_barge.get(c)
            k_str = f"B{str(k).rjust(k_width)}" if k is not None else " Truck "
            label = f"{k_str.ljust(k_width + 2)} | {str(c).rjust(c_width, '0')} "
            ylabels.append(label)

        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(ylabels, fontsize=10, family='monospace')

        ax.invert_yaxis()

        # REDUCE SPACE BETWEEN LAST ROW AND X-AXIS:
        # Since axis is inverted, the "bottom" (x-axis) is the highest Y index.
        # Setting the limit to (last_index + 0.6) makes the gap very small.
        ax.set_ylim(n_rows - 0.4, -0.6)

        ax.set_xlabel("Time [hours]", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # --------------------------
        # Legend
        # --------------------------
        legend_handles = [
            mlines.Line2D([], [], color="#FF6F00", linewidth=2),
            mlines.Line2D([], [], color="#00A63C", linewidth=2),
            mlines.Line2D([], [], color="black", marker="2", markersize=10, linestyle="None"),
            mlines.Line2D([], [], color="black", marker="s", mfc="white", markeredgecolor="black", linestyle="None")
        ]
        legend_labels = ["Export window", "Import window", "Service time", "Release time"]

        ax.legend(
            legend_handles, legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.03),  # Positioned closer to the x-axis label
            ncol=2, frameon=True, fontsize=9
        )

        plt.tight_layout()

        outfile = f"Storage_orig/Figures/time_windows{self.file_name}.pdf"
        # bbox_inches='tight' removes extra white space around the PDF
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_barge_specific_split_timelines(self, margin_hours=2.0):
        """
        Generates one figure per barge.
        The figure is split horizontally into subplots (chunks) for each terminal visit.
        
        UPDATES:
        - Black Dot = Exact Arrival Time (t_jk).
        - Grey Band = Extends from Black Dot to (Black Dot + Handling Duration).
        """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches

        m = self.model
        if m is None or m.status != GRB.OPTIMAL:
            print("No optimal solution available.")
            return

        # --- Data Unpacking ---
        N = self.N_list
        K_b = self.K_b
        f_ck = self.f_ck
        x_ijk = self.x_ijk
        t_jk = self.t_jk
        Z_cj = self.Z_cj
        C = self.C_list
        E = set(self.E)
        I = set(self.I)
        R_c = self.R_c
        O_c = self.O_c
        D_c = self.D_c
        L_hours = self.Handling_time

        used_barges = []
        for k in K_b:
            if sum(x_ijk[0, j, k].X for j in N if j != 0) > 0.5:
                used_barges.append(k)

        if not used_barges:
            print("No barges used.")
            return

        # --- Loop per Barge ---
        for k in used_barges:
            
            # 1. Reconstruct Route
            route = [0]
            curr = 0
            visited = {0}
            while True:
                next_node = None
                for j in N:
                    if j != curr and x_ijk[curr, j, k].X > 0.5:
                        next_node = j
                        break
                if next_node is None or next_node in visited:
                    break 
                route.append(next_node)
                visited.add(next_node)
                curr = next_node

            num_stops = len(route)
            
            # 2. Identify containers and sort for Y-axis
            barge_containers = [c for c in C if f_ck[c, k].X > 0.5]
            
            # Sort: Exports first, then Imports
            def sort_key(c):
                is_import = 1 if c in I else 0
                time_val = O_c[c] if is_import else R_c[c]
                return (is_import, time_val)
            
            barge_containers.sort(key=sort_key)
            y_map = {c: i for i, c in enumerate(barge_containers)}
            total_rows = len(barge_containers)

            # 3. Setup Figure
            fig, axes = plt.subplots(1, num_stops, figsize=(3.5 * num_stops, max(4, total_rows * 0.3)), 
                                     sharey=True)
            if num_stops == 1: axes = [axes]
            
            fig.suptitle(f"Barge {k} Operations (Capacity: {self.Qk[k]})", fontsize=14, fontweight='bold', y=0.98)

            # --- Loop per Stop (Chunk) ---
            for ax_idx, (node, ax) in enumerate(zip(route, axes)):
                
                # A. Timing & Duration
                # t_jk represents the time the barge is ready at the terminal
                start_time = t_jk[node, k].X 
                
                # Identify handled containers
                if node == 0:
                    # At Dry Port: Loading Exports
                    handled_here = [c for c in barge_containers if c in E]
                else:
                    # At Sea Terminal: Deliver Exports OR Pickup Imports
                    handled_here = [c for c in barge_containers if Z_cj[c][node] == 1]
                
                # Duration starts FROM the arrival/start time
                duration = len(handled_here) * L_hours
                end_time = start_time + duration

                # B. Plot The "Active" Window (Grey Band)
                # Spans from Arrival -> Arrival + Handling
                ax.axvspan(start_time, end_time, color='lightgrey', alpha=0.5, zorder=0)
                
                # Vertical edges
                ax.axvline(start_time, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
                ax.axvline(end_time, color='black', linestyle=':', linewidth=0.8, alpha=0.5)

                # C. Plot Containers
                for c in handled_here:
                    y = y_map[c]
                    
                    if c in E:
                        color, color_dark = "#FF6F00", "#B23E00" # Orange
                    else:
                        color, color_dark = "#00A63C", "#006E28" # Green
                        
                    R, O, D = R_c[c], O_c[c], D_c[c]
                    
                    # 1. The Valid Time Window Line
                    ax.hlines(y, O, D, colors=color_dark, linewidth=1.5, zorder=2)
                    
                    # 2. Markers
                    if c in E:
                        ax.scatter(R, y, marker="s", s=30, facecolor="white", edgecolor="black", zorder=3)
                    ax.scatter(O, y, marker=">", s=40, facecolor=color, edgecolor=color_dark, zorder=3)
                    ax.scatter(D, y, marker="<", s=40, facecolor=color, edgecolor=color_dark, zorder=3)
                    
                    # 3. The Black Dot (Arrival / Start of Service)
                    # Plotted exactly at start_time
                    ax.scatter(start_time, y, marker="o", color="black", s=30, zorder=5)

                # D. Formatting
                # Zoom in: Start - Margin TO End + Margin
                eff_duration = max(duration, 0.5)
                ax.set_xlim(start_time - margin_hours, end_time + margin_hours)
                
                if node == 0:
                    ax.set_title("Dry Port\n(Start)", fontsize=10, fontweight='bold')
                else:
                    ax.set_title(f"Term {node}\n(Visit)", fontsize=10, fontweight='bold')
                
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.grid(True, axis='x', linestyle=':', alpha=0.5)

            # --- Global Labels ---
            inv_map = {v: k for k, v in y_map.items()}
            y_ticks = range(total_rows)
            y_labels = [f"C{inv_map[i]}" for i in y_ticks]
            
            axes[0].set_yticks(y_ticks)
            axes[0].set_yticklabels(y_labels, fontsize=8)
            axes[0].set_ylabel("Container ID")
            
            # Legend
            legend_handles = [
                mpatches.Patch(facecolor='lightgrey', edgecolor='gray', label='Handling Duration'),
                mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Arrival (Start Ops)'),
                mlines.Line2D([], [], color='#FF6F00', marker='<', label='Export Due'),
                mlines.Line2D([], [], color='#00A63C', marker='>', label='Import Open'),
            ]
            fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=9)
            
            plt.subplots_adjust(bottom=0.15, wspace=0.1)
            
            filename = f"Storage_orig/Figures/timeline_split_{self.file_name}_barge_{k}.pdf"
            plt.savefig(filename)
            plt.close()
            print(f"Generated split timeline for Barge {k}: {filename}")

    def print_time_schedule(self):
        """
        Prints a detailed chronological schedule for each barge in the console.
        FIXED: Prevents infinite loop when barge returns to Node 0.
        """
        import pandas as pd
        from tabulate import tabulate

        m = self.model
        if m is None or m.status != GRB.OPTIMAL:
            print("No optimal solution available to print schedule.")
            return

        # Unpack Data
        N = self.N_list
        K_b = self.K_b
        x_ijk = self.x_ijk
        t_jk = self.t_jk
        f_ck = self.f_ck
        Z_cj = self.Z_cj
        C = self.C_list
        E = set(self.E)
        I = set(self.I)
        L_hours = self.Handling_time

        print("\n\n      Barge Schedules       ")
        print("============================")

        barge_found = False

        for k in K_b:
            # 1. Check if barge is used (leaves Dry Port)
            is_used = sum(x_ijk[0, j, k].X for j in N if j != 0) > 0.5
            if not is_used:
                continue
            
            barge_found = True
            schedule_data = []
            
            # 2. Reconstruct Route: Start at 0
            curr = 0
            visit_order = 1
            
            # Safety: Track visited arcs to prevent cycles
            visited_nodes = set()
            
            while True:
                # --- A. Gather Timing Info ---
                # For Node 0, t_jk is departure time usually, but let's just grab the value
                arrival_val = t_jk[curr, k].X
                
                # --- B. Calculate Handling at this Node ---
                handled_containers = []
                activity_desc = ""
                
                if curr == 0 and visit_order == 1:
                    # START of trip (Dry Port)
                    handled_containers = [c for c in C if f_ck[c, k].X > 0.5 and c in E]
                    activity_desc = "Start / Load Exports"
                elif curr == 0 and visit_order > 1:
                    # END of trip (Return to Dry Port)
                    # No new loading, just arrival
                    handled_containers = []
                    activity_desc = "Return / End Trip"
                else:
                    # Sea Terminal
                    handled_containers = [c for c in C if f_ck[c, k].X > 0.5 and Z_cj[c][curr] == 1]
                    activity_desc = "Unload Exp / Load Imp"

                qty = len(handled_containers)
                duration = qty * L_hours
                departure_val = arrival_val + duration

                # --- C. Add to Table ---
                schedule_data.append({
                    "Order": visit_order,
                    "Terminal": "Dry Port (0)" if curr == 0 else f"Terminal {curr}",
                    "Arrival": f"{arrival_val:.2f}",
                    "Cont.": qty,
                    "Dur.": f"{duration:.2f}",
                    "Depart": f"{departure_val:.2f}",
                    "Activity": activity_desc
                })
                
                # If we just processed the return to 0, stop.
                if curr == 0 and visit_order > 1:
                    break

                # --- D. Find Next Node ---
                next_node = None
                for j in N:
                    # Look for active arc
                    if j != curr and x_ijk[curr, j, k].X > 0.5:
                        next_node = j
                        break
                
                if next_node is None:
                    # No outgoing arc (shouldn't happen if flow is conserved, unless end of route)
                    break
                
                # Update for next iteration
                curr = next_node
                visit_order += 1
                
                # Safety break for huge loops
                if visit_order > len(N) + 2:
                    print(f"Warning: Cycle detected for Barge {k}")
                    break

            # 3. Print Table for this Barge
            df = pd.DataFrame(schedule_data)
            print(f"\n--- Barge {k} (Capacity: {self.Qk[k]} TEU) ---")
            print(tabulate(df, headers="keys", tablefmt="simple", showindex=False))

        if not barge_found:
            print("No barges were utilized in this solution (All containers trucked).")
        print("\n")



    # -----------------------
    # Convenience pipeline
    # -----------------------

    def run(self, with_plots=True, warm_start_sol=None):
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

        if warm_start_sol:
                    self.apply_warm_start(warm_start_sol)

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
            save_path = f"Storage_orig/Solutions/solved_{self.file_name}.sol"

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
            self.print_time_schedule()
            self.print_economic_validation()
            if with_plots:
                # self.plot_barge_displacements()
                # self.plot_barge_solution_map()
                # self.plot_barge_solution_map_report()
                # self.plot_barge_solution_map_report_2()
                self.plot_barge_solution_map_report_3(curvature_multiplier=4.0, size_scale=2.0)
                # self.plot_barge_solution_map_report_Without_containers()
                # self.plot_barge_solution_map_report_ONLY_NODES()
                self.plot_time_windows()
                # self.plot_barge_specific_split_timelines(margin_hours=3.0)
                

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
    # milp.generate_travel_times_fazi_case_study()
    #milp.plot_topography_preview()
    milp.run(with_plots=True)

    # Separation offsets for nodes 1&4 and 2&5
    # offsets = {
    #    1: (-3, -1),  # Move node 1 up
    #    4: (-3, 1),  # Move node 4 down
    #    2: (1+2, -1),  # Move node 2 left
    #    5: (-1+2, 1)  # Move node 5 right
    #}

    # Plotting call with options for offsets, scaling, and .sol file loading
    #milp.plot_barge_solution_map_report_3(
    #    node_offsets=offsets,
    #    curvature_multiplier=5.0,
    #    size_scale=2.0,
    #    sol_file_path="Storage_orig/Solutions/solved________2026_02_25_22_18_28.sol"  # Change to None to use model in memory
    #)
