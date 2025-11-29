#!/usr/bin/env python3
"""
Implementation of the Mixed-Integer Linear Programming (MILP) model for Container Allocation Optimization.
This module defines the MILP class which sets up and solves the optimization problem using Gurobi. (Requires an installation and License.)

"""

import random
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np
import pandas as pd
from tabulate import tabulate

random.seed(100)
print("\n\n\n\n\n\n\n\n\n\n\n")

"""
MILP model for barge-truck scheduling.

Notes:
- Qk is scaled by 20 to match container sizes (20/40) as in your original code.
- Last vehicle in K is the truck (index K_t); others are barges (K_b).
"""

# -----------------------
# Global parameters
# -----------------------

Qk = [104, 99, 81]  # [TEU] (Twenty-foot Equivalent Unit)
print("Qk (TEU):", Qk)
Qk = [q * 20 for q in Qk]  # keep your original scaling to match 20/40 sizes
print("Qk (capacity units):", Qk)

HkB = [3700, 3600, 3400]  # [euros] per barge round trip
Ht_40 = 200               # [euros] per 40-foot container
Ht_20 = 140               # [euros] per 20-foot container
L = 10                    # [minutes] handling time per container
Gamma = 100               # [euros] penalty per sea terminal visit
M = 1_000_000             # big-M


# -----------------------
# Instance generation
# -----------------------

def generate_instance():
    """
    Generate random instance data:
    - C: containers
    - N: terminals
    - E / I: export / import sets
    - K: vehicles (barges + truck)
    - Z_cj: assignment of container c to terminal j
    - W_c: container sizes (20 or 40)
    - R_c, O_c, D_c: release, opening, closing times [minutes]
    - H_T: trucking cost per container
    """
    num_C = random.randint(50, 70)   # number of containers                    #+#+# 100, 200
    num_N = random.randint(2, 3)       # number of terminals (fixed to 6 here)   #+#+# 6, 6

    # Number of vehicles:
    # - len(Qk) barges
    # - 1 truck (last index)
    K_barge = len(Qk) + 1
    K = list(range(K_barge))

    C = list(range(num_C))  # container indices
    N = list(range(num_N))  # terminal indices

    E = []        # export containers
    I = []        # import containers
    W_c = []      # container size (20 or 40)
    R_c = []      # release time
    O_c = []      # opening time
    D_c = []      # closing time
    H_T = []      # trucking cost per container

    # Z_cj[c][j] = 1 if container c associated with terminal j
    Z_cj = [[0 for _ in N] for _ in C]

    for c in C:
        # Export vs import
        p_export = random.uniform(0.05, 0.7)
        is_export = (random.random() < p_export)

        if is_export:
            E.append(c)
        else:
            I.append(c)

        # 40-foot vs 20-foot
        p_40 = random.uniform(0.75, 0.9)
        size = 40 if random.random() > p_40 else 20
        W_c.append(size)

        # Trucking cost based on size
        H_T.append(Ht_40 if size == 40 else Ht_20)

        # Times [minutes]
        release = random.randint(0, 1440)   # 0–1 day
        closing = random.randint(1440, 11760)  # 1–8 days
        opening = random.randint(max(0, closing - 7200), closing - 1440)

        R_c.append(release)
        O_c.append(opening)
        D_c.append(closing)

        # Assign container to a non-dryport terminal (1..N-1)
        j = random.randint(1, len(N) - 1)
        Z_cj[c][j] = 1

    return C, N, E, I, K, Z_cj, W_c, R_c, O_c, D_c, H_T


C, N, E, I, K, Z_cj, W_c, R_c, O_c, D_c, H_T = generate_instance()

# Barge indices and truck index
K_b = K[:-1]      # all but last are barges
K_t = K[-1]       # last one is truck

# -----------------------
# Travel time matrix
# -----------------------

def Tij(num_nodes, long, short, aleatorio=False):
    """
    Build travel time matrix T_ij [minutes].

    i = origin terminal
    j = destination terminal
    - long: time between dryport (0) and any other node
    - short: time between non-0 nodes
    - if aleatorio=True, generate random symmetric times (except diagonal)
    """
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
                dist = random.randint(1, 10)
                T_ij_matrix[i][j] = dist
                T_ij_matrix[j][i] = dist
        for i in range(num_nodes):
            T_ij_matrix[i][i] = 0

    return T_ij_matrix


T_ij_matrix = Tij(len(N), 660, 60)  # times in minutes


# -----------------------
# Model setup
# -----------------------

m = Model("BargeScheduling")

# Decision variables
# f_ck = 1 if container c is allocated to vehicle k (barges or truck)
f_ck = m.addVars(C, K, vtype=GRB.BINARY, name="f_ck")

# x_ijk = 1 if barge k sails from terminal i to j
x_ijk = m.addVars(N, N, K_b, vtype=GRB.BINARY, name="x_ijk")

# p_jk: import quantity loaded by barge k at terminal j
# d_jk: export quantity unloaded by barge k at terminal j
p_jk = m.addVars(N, K_b, vtype=GRB.INTEGER, lb=0, name="p_jk")
d_jk = m.addVars(N, K_b, vtype=GRB.INTEGER, lb=0, name="d_jk")

# y_ijk: import quantity carried by barge k from i to j
# z_ijk: export quantity carried by barge k from i to j
y_ijk = m.addVars(N, N, K_b, vtype=GRB.INTEGER, lb=0, name="y_ijk")
z_ijk = m.addVars(N, N, K_b, vtype=GRB.INTEGER, lb=0, name="z_ijk")

# t_jk: time barge k is at terminal j
t_jk = m.addVars(N, K_b, vtype=GRB.CONTINUOUS, name="t_jk")

m.update()

# -----------------------
# Objective function
# -----------------------

objective = (
    # 1. Trucking cost
    quicksum(f_ck[c, K_t] * H_T[c] for c in C)
    +
    # 2. Barge fixed cost (if barge leaves dryport)
    quicksum(x_ijk[0, j, k] * HkB[k] for k in K_b for j in N if j != 0)
    +
    # 3. Travel cost between terminals
    quicksum(T_ij_matrix[i][j] * x_ijk[i, j, k]
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

# Optional: limit total trucked containers (kept from your original code)
limit_total_trucked_containers = True
if limit_total_trucked_containers:
    m.addConstr(quicksum(f_ck[c, K_t] for c in C) <= len(C) - 10,
                name="Trucked_Limit")

# 2. Each container is assigned to exactly one vehicle
for c in C:
    m.addConstr(quicksum(f_ck[c, k] for k in K) == 1,
                name=f"Container_Assignment_{c}")

# 3. Flow conservation for barges at each node
for i in N:
    for k in K_b:
        m.addConstr(
            quicksum(x_ijk[i, j, k] for j in N if j != i)
            - quicksum(x_ijk[j, i, k] for j in N if j != i) == 0,
            name=f"Flow_Conservation_{i}_{k}"
        )

# 4. Each barge leaves dryport (0) at most once
for k in K_b:
    m.addConstr(
        quicksum(x_ijk[0, j, k] for j in N if j != 0) <= 1,
        name=f"Departures_{k}"
    )

# 5. No self-loops (i -> i)
for i in N:
    for k in K_b:
        m.addConstr(x_ijk[i, i, k] == 0, name=f"No_Self_Loop_{i}_{k}")

# 6. Import quantity at terminal j for barge k
for k in K_b:
    for j in N[1:]:
        m.addConstr(
            p_jk[j, k] == quicksum(W_c[c] * Z_cj[c][j] * f_ck[c, k]
                                   for c in I),
            name=f"Import_Quantity_{j}_{k}"
        )

# 7. Export quantity at terminal j for barge k
for k in K_b:
    for j in N[1:]:
        m.addConstr(
            d_jk[j, k] == quicksum(W_c[c] * Z_cj[c][j] * f_ck[c, k]
                                   for c in E),
            name=f"Export_Quantity_{j}_{k}"
        )

# 8. Flow balance for import quantities at terminal j for barge k
for j in N[1:]:
    for k in K_b:
        m.addConstr(
            quicksum(y_ijk[j, i, k] for i in N if i != j)
            - quicksum(y_ijk[i, j, k] for i in N if i != j)
            == p_jk[j, k],
            name=f"Import_Balance_{j}_{k}"
        )

# 9. Flow balance for export quantities at terminal j for barge k
for j in N[1:]:
    for k in K_b:
        m.addConstr(
            quicksum(z_ijk[i, j, k] for i in N if i != j)
            - quicksum(z_ijk[j, i, k] for i in N if i != j)
            == d_jk[j, k],
            name=f"Export_Balance_{j}_{k}"
        )

# 10. Barge trip capacity constraint
for i in N:
    for j in N:
        if i == j:
            continue
        for k in K_b:
            m.addConstr(
                y_ijk[i, j, k] + z_ijk[i, j, k] <= Qk[k] * x_ijk[i, j, k],
                name=f"Flow_Capacity_{i}_{j}_{k}"
            )

# 11. Export containers: departure time at dryport >= release time
for c in E:
    for k in K_b:
        m.addConstr(
            t_jk[0, k] >= R_c[c] * f_ck[c, k],
            name=f"Vehicle_Departure_{c}_{k}"
        )

# Time constraints
include_time_constraints = True
if include_time_constraints:
    # 12 & 13. Time propagation along arcs with handling time at arrival
    for i in N:
        for j in N[1:]:
            if i == j:
                continue
            for k in K_b:
                handling_term = quicksum(L * Z_cj[c][j] * f_ck[c, k] for c in C)

                m.addConstr(
                    t_jk[j, k] >= t_jk[i, k] + handling_term + T_ij_matrix[i][j] - (1 - x_ijk[i, j, k]) * M,
                    name=f"Time_LB_{i}_{j}_{k}"
                )
                m.addConstr(
                    t_jk[j, k] <= t_jk[i, k] + handling_term + T_ij_matrix[i][j] + (1 - x_ijk[i, j, k]) * M,
                    name=f"Time_UB_{i}_{j}_{k}"
                )

    # 14. Export container service cannot start before opening time
    for c in C:
        for j in N[1:]:
            for k in K_b:
                m.addConstr(
                    t_jk[j, k] >= O_c[c] * Z_cj[c][j] - (1 - f_ck[c, k]) * M,
                    name=f"Export_Time_{c}_{j}_{k}"
                )

    # 15. All containers must be served before closing time
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

m.optimize()

print("==== Instance data ====")
print("C (containers):", C)
print("N (nodes):", N)
print("E (exports):", E)
print("I (imports):", I)
print("K (vehicles):", K)
print("Truck index (K_t):", K_t)


# -----------------------
# Result printing helpers
# -----------------------

def print_results(model):
    if model.status != GRB.OPTIMAL:
        print("No optimal solution found. Status:", model.status)
        return

    print("\n\nOptimal solution found!")
    print(f"Optimal objective value: {model.objVal:.2f}")

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


def print_container_table(C, W_c, R_c, O_c, D_c, E, I, Z_cj, K, f_ck):
    """
    Prints a table summarizing container properties and assigned barge/truck.
    Exports: Node 0 -> Node j
    Imports: Node j -> Node 0
    """
    container_data = []
    for c in C:
        container_type = "Export" if c in E else "Import"
        node = Z_cj[c].index(1)  # terminal associated with c

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

        container_data.append({
            "Container ID": c,
            "Size (ft)": W_c[c],
            "Type": container_type,
            "Origin": origin,
            "Destination": destination,
            "Release Time": R_c[c],
            "Opening Time": O_c[c],
            "Closing Time": D_c[c],
            "Assigned Vehicle": assigned_label,
            "Sort Node": sort_node,
            "Sort Type": 0 if container_type == "Export" else 1
        })

    df = pd.DataFrame(container_data)
    df = df.sort_values(by=["Sort Node", "Sort Type"]).drop(columns=["Sort Node", "Sort Type"])

    print("\nContainer Table (Grouped by Node and Type)")
    print("==========================================")
    print(tabulate(df, headers="keys", tablefmt="grid"))


def print_node_table(N, Z_cj, E, I):
    """
    Prints a table summarizing how many imports/exports are associated to each node.
    """
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


def print_barge_table(K_b, Qk, x_ijk, N, y_ijk, z_ijk):
    """
    Prints a table summarizing barge routes and capacity utilization per arc.
    """
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
                        "Capacity Used": capacity_used,
                        "Capacity": total_capacity,
                        "Utilization (%)": f"{utilization_percent:.2f}",
                    })

        if routes:
            df = pd.DataFrame(routes)
            print(f"\nBarge {k} Route & Capacity Usage")
            print("================================")
            print(tabulate(df, headers="keys", tablefmt="grid"))


def plot_barge_displacements_off(T_ij_matrix, x_ijk, t_jk, K_b, N):
    """
    Plots displacements of barges based on x_ijk using MDS layout.
    (Trucked containers are not plotted to keep this simple and correct.)
    """
    T_ij_matrix = np.array(T_ij_matrix)

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
    plt.show()


# -----------------------
# Use helpers if optimal
# -----------------------

print_node_table(N, Z_cj, E, I)

if m.status == GRB.OPTIMAL:
    print_results(m)
    print_container_table(C, W_c, R_c, O_c, D_c, E, I, Z_cj, K, f_ck)
    print_barge_table(K_b, Qk, x_ijk, N, y_ijk, z_ijk)

    print(f"\nTotal containers: {len(C)}")
    print(f"K_b (barges): {K_b}")
    print(f"K_t (truck): {K_t}")

    plot_barge_displacements_off(T_ij_matrix, x_ijk, t_jk, K_b, N)
