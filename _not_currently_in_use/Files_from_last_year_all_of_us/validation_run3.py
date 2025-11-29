import random
random.seed(100)
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np

"""
the time constraints are not working as expected, 
if you look at the map and at the t_jks, time does not match since it moves backward in time. 
it does the trip  0 - 3 - 1 - 2 - 0, and the times it reaches it are t_0 = 6, t_3 = 93, t_1 = 82, t_2 = 39, ____

Thus it is clearly incorrect


"""


# Qk = [20, 15, 20, 20] #[TEU] (Twenty-foot Equivalent Unit)
# HkB = [50, 50, 50, 50] #[euros] (per round trip)
Qk = [2, 5, 4]              #[TEU] (Twenty-foot Equivalent Unit)
HkB = [50, 50, 50]           #[euros] (per round trip)
Qk = [5]              #[TEU] (Twenty-foot Equivalent Unit)
HkB = [50]           #[euros] (per round trip)
Ht_40 = 200 #[euros] (40-foot container)
Ht_20 = 140 #[euros] (20-foot container)
L = 0.5 #[min] (handling time per container)
Gamma = 1 #[euros] penalty for sea terminal visit (TUNEEEEEEE!!!!!!!!!!!!!!!!!!!!!!!!)
M = 1000 #big M



def generate_instance():

    C = random.randint(7, 7) #number of containers      11
    N = random.randint(4, 4) #number of terminals

    # K_truck = 1 #number of trucks
    K_barge = len(Qk) #number of barges = defined by defining the number of capacities


    # K = [i for i in range(K_truck + K_barge)]
    K = [i for i in range(K_barge)]

    C = [i for i in range(C)] #container index  # length = 20
    N = [i for i in range(N)] #terminal index   #length = 4
    E = [] # Containers that are exports
    I = [] # Containers that are imports
    W_c = []    # Container sizes
    R_c = []    # Container Release time    
    O_c = []    # Container Opening time
    D_c = []    # Container Closing time

    H_T = []    # Cost of trucking each container
    Z_cj = [[0 for _ in range(len(N))] for _ in range(len(C))]

    for container_idx in C:
        #probability for each container to be export

        p_export = 0.5 #random.uniform(0, 1)
        is_export = "E" if random.random() < p_export else "I"

        if is_export == "E":
            E.append(container_idx)
        else:
            I.append(container_idx)



        #probability to be a 40-foot container
        p_40 = 0#random.uniform(0.75, 0.9)  # it will always be a 20-foot container
        Size = 1 if random.random() > p_40 else 1

        W_c.append(Size)

        # H_T definition based on the probabilities of container types
        if Size == 1:
            H_T.append(Ht_20)
        else:
            H_T.append(Ht_40)

        
        #Release, closing and opening dates in hours, starting from hour 0 on Monday
        Release = random.randint(0, 10) #release date container
        Closing = random.randint(100, 400) #closing date container
        Opening = random.randint(10, 100) #open date container

        R_c.append(Release)
        O_c.append(Opening)
        D_c.append(Closing)

        j = random.randint(1,len(N)-1)
        # j = 3
        Z_cj[container_idx][j] = 1

    return C, N, E, I, K, Z_cj, W_c, R_c, O_c, D_c, H_T


C, N, E, I, K, Z_cj, W_c, R_c, O_c, D_c, H_T  = generate_instance()

# O_c[6] = 800
# D_c[6] = 900



def Tij(N, long, short, aleatorio = False): # i is origin terminal and j is destination terminal, Tij is the travel time between terminals i and j in [minutes] (from Fazi et al., 2015)
    
    T_ij_matrix = [[0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        for j in range(N):
            if i == 0 and i != j: #from dryport to seaport (Some exports)
                T_ij = long
            elif j == 0 and i != j: #from seaport to dryport (Basically imports)
                T_ij = long
            elif i == j : #intra terminal travel time
                T_ij = M
            # elif i == j and i == 0: #intra dryport travel time
            #     T_ij = M
            else:
                T_ij = short
            T_ij_matrix[i][j] = T_ij
        
    if aleatorio:
            for i in range(N):
                for j in range(N):
                    if i == j : #intra terminal travel time
                        T_ij_matrix[i][j] = M
                    else:
                        local_random = random.Random()

                        dist = local_random.randint(1, 10)
                        T_ij_matrix[i][j] = dist
                        T_ij_matrix[j][i] = dist

    return T_ij_matrix


T_ij_matrix = Tij(len(N), 5, 3)







def plot_nodes(T_ij_matrix):
    
    # Step 1: Convert T_ij_matrix to a numpy array
    T_ij_matrix = np.array(T_ij_matrix)

    # Step 2: Apply Multidimensional Scaling (MDS)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    node_positions = mds.fit_transform(T_ij_matrix)

    # Step 3: Plot the nodes
    plt.figure(figsize=(8, 6))

    for idx, (x, y) in enumerate(node_positions):
        plt.scatter(x, y, label=f"Node {idx}", s=100, marker='^', facecolors='none', edgecolors='k')  # Scatter point for each node
        plt.text(x + 0.3, y + 0.2, f"{idx}", fontsize=9)  # Label each node

    plt.title("2D Map of Nodes Based on Travel Times")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    # plt.legend()
    plt.show()

# plot_nodes(T_ij_matrix)




###################
### MODEL SETUP ###
###################

# Initialize empty model

m = Model("Barge Scheduling")

#Create decision variables

#Bin - equals 1 if container c is allocated to means k
f_ck = m.addVars(C, K, vtype = GRB.BINARY, name = "f_ck") 

#Bin - equals 1 if barge k sails from terminal i to j
x_ijk = m.addVars(N, N, K, vtype = GRB.BINARY, name = "x_ijk") 

#Int - Import quantity loaded by barge k at sea terminal j 
p_jk = m.addVars(N, K, vtype = GRB.INTEGER, lb=0, name = "p_jk") 
#Int - Export quantity unloaded by barge k at sea terminal j 
d_jk = m.addVars(N, K, vtype = GRB.INTEGER, lb=0, name = "d_jk") 

#Int - Import quantity carried by barge k from terminal i to j
y_ijk = m.addVars(N, N, K, vtype = GRB.INTEGER, lb=0, name = "y_ijk") 
#Int - Export quantity carried by barge k from terminal i to j
z_ijk = m.addVars(N, N, K, vtype = GRB.INTEGER, lb=0, name = "z_ijk") 

#Cont - Time barge k is at terminal j 
t_jk = m.addVars(N, K, vtype = GRB.CONTINUOUS, name = "t_jk") 

m.update()



# objective = quicksum(x_ijk[0, j, k] * HkB[k] for k in K for j in N if j != 0)
objective = quicksum(x_ijk[0, j, k] * HkB[k] for k in K for j in N for i in N if j != 0) + \
            quicksum(x_ijk[i, j, k] * Gamma for k in K for j in N for i in N if j != 0)  #+ quicksum(T_ij_matrix[i][j] * x_ijk[i, j, k] for k in K for i in N for j in N if i != j) 

m.setObjective(objective, GRB.MINIMIZE)


for c in C:         # 2
    m.addConstr(quicksum(f_ck[c, k] for k in K) == 1, name=f"Container_Assignment_{c}")


for i in N:         # 3 
    for k in K:
        m.addConstr(
            quicksum(x_ijk[i, j, k] for j in N if j != i) - quicksum(x_ijk[j, i, k] for j in N if j != i) == 0,
            name=f"Flow_Conservation_{i}_{k}"
        )



for k in K:         # 4     constrains each barge to only leave once  (maximum leave is one)
    m.addConstr(
        quicksum(x_ijk[0, j, k] for j in N if j != 0) <= 1,
        name=f"Departures fixed_{k}"
    )



## ---------------------------------------------------------------------------
for i in N:         # my own  trips between the same node (non_existent trips) #~# added by me 
    for j in N:
        if i == j:
            m.addConstr(
                quicksum(x_ijk[i, j, k] for k in K)  == 0,
                name=f"Avoid trips to the same node_{i}_{j}"
            )




for k in K:         # 5 Import quantity 
    for j in N[1:]:  
        m.addConstr(
            p_jk[j, k] == quicksum(W_c[c] * Z_cj[c][j] * f_ck[c, k] for c in I),
            name=f"Import_Quantity_{j}_{k}"
        )



for k in K:         # 6 export quantity
    for j in N[1:]:  
        m.addConstr(
            d_jk[j, k] == quicksum(W_c[c] * Z_cj[c][j] * f_ck[c, k] for c in E),
            name=f"Export_Quantity_{j}_{k}"
        )


for j in N[1:]:     # 7 flow equations for import quantities at terminal j handled by harge k 
    for k in K:
        m.addConstr(
            quicksum(y_ijk[j, i, k] for i in N if i != j) - quicksum(y_ijk[i, j, k] for i in N if i != j) == p_jk[j, k],
            name=f"Import_Balance_{j}_{k}"
        )


for j in N[1:]:     # 8  flow equations for export quantities at terminal j handled by harge k 
    for k in K:
        m.addConstr(
            quicksum(z_ijk[i, j, k] for i in N if i != j) - quicksum(z_ijk[j, i, k] for i in N if i != j) == d_jk[j, k],
            name=f"Export_Balance_{j}_{k}"
        )


for i in N:     # 9 capacity of each barge is not overdone at each trip
    for j in N:
        for k in K:
            m.addConstr(
                y_ijk[i, j, k] + z_ijk[i, j, k] <= Qk[k] * x_ijk[i, j, k],
                name=f"Flow_Capacity_{i}_{j}_{k}"
            )



for c in E:
    for k in K:         # 10
        m.addConstr(
            t_jk[0, k] >= R_c[c] * f_ck[c, k],
            name=f"Vehicle_Departure_{c}_{k}"
        )


#================================================================================================
# Time constraints
#================================================================================================
#================================================================================================
include_time_constraints = True
if include_time_constraints:
#     for i in N:
#         for j in N:
#             for k in K:
#                 m.addConstr(        # 11
#                     t_jk[j, k] >= t_jk[i, k] + quicksum(L*Z_cj[c][i] * f_ck[c, k] for c in C) + T_ij_matrix[i][j] - (1 - x_ijk[i, j, k]) * M,
#                     name=f"Time_LowerBound_{i}_{j}_{k}"
#                 )
#                 m.addConstr(        # 12
#                     t_jk[j, k] <= t_jk[i, k] + quicksum(L*Z_cj[c][i] * f_ck[c, k] for c in C) + T_ij_matrix[i][j] + (1 - x_ijk[i, j, k]) * M,
#                     name=f"Time_UpperBound_{i}_{j}_{k}"
#                 )



    for c in C:             # 13
        for j in N[1:]:  # Exclude dry port (j=0)
            for k in K:
                m.addConstr(
                    t_jk[j, k] >= O_c[c] * Z_cj[c][j] - (1 - f_ck[c, k]) * M,
                    name=f"Export_Time_{c}_{j}_{k}"
                )

    for c in C:            # 14 
        for j in N[1:]:  # Exclude dry port (j=0)
            for k in K:
                m.addConstr(
                    t_jk[j, k] * Z_cj[c][j] <= D_c[c] + (1 - f_ck[c, k]) * M,
                    name=f"Demand_Fulfillment_{c}_{j}_{k}"
                )


# Run optimization
m.optimize()


print("below are our own print statements")



#================================================================================================
# Reasults from the instance generation
#================================================================================================
print("C: set of containers\n", C)
print("N: set of nodes\n", N)
print("E: export containers\n", E)
print("I: import containers\n", I)
print("K: set of vehicles\n", K)
print("Z_cj:\n", Z_cj)
print("W_c: container size\n", W_c)
print("R_c: release times \n", R_c)
print("O_c: opening times\n", O_c)
print("D_c: closing times\n", D_c)
print("H_T: Cost of trucking container\n", H_T)

print_sources_n_origins_of_containers = True

if print_sources_n_origins_of_containers:
    print(f"\nAmount of export containers departing from 0:\n", len(E))
    print(f"\nAmount of import containers ending at 0:\n", len(I))
    for idx, elem in enumerate(Z_cj):
        if idx in E:
            print(f"Export Container {idx} is going to node {elem.index(1)}")

    for idx, elem in enumerate(Z_cj):
        if idx in I:
            print(f"Import Container {idx} is coming from node {elem.index(1)}")

print("\nT_ij_matrix (distances matrix):")
for row in T_ij_matrix:
    print(row)





#================================================================================================
# results from optimization
#================================================================================================
def print_results(m):
    # Check if an optimal solution was found
    if m.status == GRB.OPTIMAL:
        print("\n\n\nOptimal solution found!")
        print(f"Optimal objective value: {m.objVal}")

        # Extracting the values of decision variables:
        # 1. f_ck: Assignment of containers to barges
        print("\nContainer to Barge Assignments (f_ck):\n====================================================")
        print(f"Length of set of containers {len(K)}")
        for c in C:
            for k in K:
                if f_ck[c, k].X > 0.5:  # If the variable is 1 (assigned)
                    if k==5:
                        print(f"Container {c} is assigned to Truck {k}")
                    else:
                        print(f"Container {c} is assigned to Barge {k}")

        # 2. x_ijk: Travel decisions of barges between terminals
        print("\nBarge Travel Decisions (x_ijk):\n====================================================")
        for i in N:
            for j in N:
                if i != j:
                    for k in K:
                        if x_ijk[i, j, k].X > 0.5:  # If the variable is 1 (traveling)
                            print(f"Barge {k} travels from Terminal {i} to Terminal {j}")
                        # if x_ijk[i, j, k].X > 0.5:  # If the variable is 1 (traveling)
                        #     print(f"Barge {k} travels from Terminal {i} to Terminal {j}")


        #total quantity transported in the lines 0-j
        assigned_quant = 0
        for k in K:             # used to print out all the capacities of the made trips between station 0 and all other stations
            assigned_quant = 0          # can also be used to identify the assignments in between seaports
            for j in N:
                if y_ijk[j, 0, k].X > 0.5:
                    assigned_quant += y_ijk[j, 0, k].X
                    available_quant = Qk[k]
                    print(F"Capacity use of barge {k} at trip {j}-0 is at {assigned_quant/available_quant*100:.2f}%, {assigned_quant}/{available_quant}")


                if z_ijk[0, j, k].X > 0.5:
                    assigned_quant += z_ijk[0, j, k].X
                    available_quant = Qk[k]
                    print(F"Capacity use of barge {k} at trip 0-{j} is at {assigned_quant/available_quant*100:.2f}%, {assigned_quant}/{available_quant}")
 
                    
        print("\n\n\n\n\nVariable values\n====================================================")
        m.printAttr("X")
    else:
        print("No optimal solution found. Status:", m.status)



print_results(m)


def plot_barge_displacements(T_ij_matrix, x_ijk, K, N):
    """
    Plots the displacements of barges based on the travel decisions (x_ijk).
    """
    # Step 1: Convert T_ij_matrix to a numpy array
    T_ij_matrix = np.array(T_ij_matrix)

    # Step 2: Apply Multidimensional Scaling (MDS) to position nodes
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    node_positions = mds.fit_transform(T_ij_matrix)

    # Step 3: Plot the nodes
    plt.figure(figsize=(10, 8))

    # Plot nodes
    for idx, (x, y) in enumerate(node_positions):
        plt.scatter(x, y, label=f"Node {idx}", s=100, marker='o', color='skyblue')
        plt.text(x + 0.02, y + 0.02, f"{idx}", fontsize=9)

    # Step 4: Plot the displacements (edges) based on x_ijk
    for k in K:  # Iterate through each barge
        for i in N:
            for j in N:
                if i != j and x_ijk[i, j, k].X > 0.5:  # If barge k travels from i to j
                    # Get positions of terminals i and j
                    x1, y1 = node_positions[i]
                    x2, y2 = node_positions[j]
                    
                    # Plot the displacement as an arrow
                    plt.arrow(
                        x1, y1, x2 - x1, y2 - y1,
                        color=f'C{k}',  # Different color for each barge
                        width=0.005,
                        head_width=0.05,
                        length_includes_head=True,
                        alpha=0.7,
                        label=f"Barge {k} path"
                    )

    # Add title, legend, and grid
    plt.title("Barge Displacements Between Terminals")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()


def plot_barge_displacements_off(T_ij_matrix, x_ijk, K, N):
    """
    Plots the displacements of barges based on the travel decisions (x_ijk).
    """
    # Step 1: Convert T_ij_matrix to a numpy array
    T_ij_matrix = np.array(T_ij_matrix)

    # Step 2: Apply Multidimensional Scaling (MDS) to position nodes
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    node_positions = mds.fit_transform(T_ij_matrix)

    # Step 3: Plot the nodes
    plt.figure(figsize=(8, 6))

    # Plot nodes
    for idx, (x, y) in enumerate(node_positions):
        plt.scatter(x, y, s=100, marker='o', color='skyblue')       # label=f"Node {idx}",
        plt.text(x + 0.02, y + 0.02, f"{idx}", fontsize=9)

    # Step 4: Plot the displacements (edges) based on x_ijk
    for k in K:  # Iterate through each barge
        for i in N:
            for j in N:
                if i != j and x_ijk[i, j, k].X > 0.5:  # If barge k travels from i to j
                    # Get positions of terminals i and j
                    x1, y1 = node_positions[i]
                    x2, y2 = node_positions[j]

                    # Apply an offset based on the barge index to separate overlapping lines
                    offset = 0.01 * k  # Adjust offset based on barge index
                    x1 += offset
                    y1 += offset
                    x2 += offset
                    y2 += offset
                    
                    # Plot the displacement as an arrow
                    plt.arrow(
                        x1, y1, x2 - x1, y2 - y1,
                        color=f'C{k}',  # Different color for each barge
                        width=0.005,
                        head_width=0.05,
                        length_includes_head=True,
                        alpha=0.7,
                        label=f"Barge {k} path between {i}, {j}"
                    )
                    # Add a label for the barge on the arrow
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    plt.text(mid_x, mid_y, f"B{k}", fontsize=8, color=f'C{k}')

    # Add title, legend, and grid
    plt.title("Barge Displacements Between Terminals")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()

def plot_barge_displacements_with_node_info(T_ij_matrix, x_ijk, K, N, Z_cj, E, I):
    """
    Plots the displacements of barges based on the travel decisions (x_ijk),
    and includes compact, clear information boxes next to each node showing container statistics.
    """
    # Step 1: Convert T_ij_matrix to a numpy array
    T_ij_matrix = np.array(T_ij_matrix)

    # Step 2: Apply Multidimensional Scaling (MDS) to position nodes
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    node_positions = mds.fit_transform(T_ij_matrix)

    # Step 3: Calculate import/export statistics for each node
    node_info = []
    for node in N:
        import_count = sum(1 for c in I if Z_cj[c][node] == 1)
        export_count = sum(1 for c in E if Z_cj[c][node] == 1)
        node_info.append((import_count, export_count))

    # Step 4: Plot the nodes and information boxes
    plt.figure(figsize=(12, 8))

    for idx, (x, y) in enumerate(node_positions):
        # Plot the node
        plt.scatter(x, y, s=150, marker='o', color='skyblue', edgecolor='black')
        # Add a compact box with node statistics
        import_count, export_count = node_info[idx]
        text_box = f"| In{import_count}, Ex{export_count}"  # Compact representation
        # plt.text(
        #     x + 0.15, y + 0.05, text_box,
        #     fontsize=8, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2')
        # )
        plt.text(x + 0.02, y + 0.02, f"Node {idx}  {text_box}", fontsize=8, weight='bold')


    # Step 5: Plot the displacements (edges) based on x_ijk
    for k in K:  # Iterate through each barge
        for i in N:
            for j in N:
                if i != j and x_ijk[i, j, k].X > 0.5:  # If barge k travels from i to j
                    # Get positions of terminals i and j
                    x1, y1 = node_positions[i]
                    x2, y2 = node_positions[j]

                    # Apply an offset based on the barge index to separate overlapping lines
                    offset = 0.01 * k  # Adjust offset based on barge index
                    x1 += offset
                    y1 += offset
                    x2 += offset
                    y2 += offset

                    # Plot the displacement as an arrow
                    plt.arrow(
                        x1, y1, x2 - x1, y2 - y1,
                        color=f'C{k}',  # Different color for each barge
                        width=0.005,
                        head_width=0.05,
                        length_includes_head=True,
                        alpha=0.7,
                        label=f"Barge {k} path between {i}, {j}"
                    )
                    # Add a label for the barge on the arrow
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    plt.text(mid_x, mid_y, f"B{k}", fontsize=8, color=f'C{k}')

    # Add title, legend, and grid
    plt.title("Barge Displacements Between Terminals with Node Information")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS

def plot_barge_displacements_aesthetic(T_ij_matrix, x_ijk, K, N):
    """
    Plots the displacements of barges based on the travel decisions (x_ijk).
    """
    # Step 1: Convert T_ij_matrix to a numpy array
    T_ij_matrix = np.array(T_ij_matrix)

    # Step 2: Apply Multidimensional Scaling (MDS) to position nodes
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    node_positions = mds.fit_transform(T_ij_matrix)

    # Step 3: Create the plot
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-whitegrid')

    # Plot nodes with consistent blueish coloring
    for idx, (x, y) in enumerate(node_positions):
        plt.scatter(
            x, y, s=150, color="#4A90E2", edgecolor="black", zorder=5 #5
        )
        plt.text(
            x + 0.04, y + 0.04, f"{idx}", fontsize=10, color="#1B1B1B", zorder=6
        )

    # Step 4: Plot the displacements (edges) based on x_ijk
    cmap = plt.cm.Blues  # Blueish color map
    max_barges = max(K)

    for k in K:  # Iterate through each barge
        for i in N:
            for j in N:
                if i != j and x_ijk[i, j, k].X > 0.5:  # If barge k travels from i to j
                    # Get positions of terminals i and j
                    x1, y1 = node_positions[i]
                    x2, y2 = node_positions[j]

                    # Apply an offset based on the barge index to separate overlapping lines
                    offset = 0.01 * k  # Adjust offset based on barge index
                    x1 += offset
                    y1 += offset
                    x2 += offset
                    y2 += offset

                    # Color intensity depends on barge index
                    arrow_color = cmap(0.5 + 0.5 * k / max_barges)

                    # Plot the displacement as an arrow
                    plt.arrow(
                        x1, y1, x2 - x1, y2 - y1,
                        color=arrow_color,
                        width=0.004,
                        head_width=0.05,
                        length_includes_head=True,
                        alpha=0.8,
                        zorder=4
                    )

                    # Add a label for the barge on the arrow
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    plt.text(
                        mid_x, mid_y, f"B{k}", fontsize=9, color=arrow_color
                    )

    # Add title, legend, and fine-tune appearance
    plt.title(
        "Barge Displacements Between Terminals",
        fontsize=14, color="#1B1B1B", weight="bold"
    )
    plt.xlabel("X Coordinate", fontsize=12, color="#1B1B1B")
    plt.ylabel("Y Coordinate", fontsize=12, color="#1B1B1B")
    plt.tight_layout()
    plt.show()



import pandas as pd
from tabulate import tabulate

def print_container_table(C, W_c, R_c, O_c, D_c, E, I, Z_cj):
    """
    Prints a professional table summarizing container properties.
    """
    container_data = []
    for c in C:
        container_type = "Export" if c in E else "Import"
        origin = "Node 0" if container_type == "Export" else f"Node {Z_cj[c].index(1)}"
        destination = "Node 0" if container_type == "Import" else f"Node {Z_cj[c].index(1)}"
        # destination_idx = Z_cj[c].index(1)-1
        # destination = f"Node {destination_idx}"

        container_data.append({
            "Container ID": c,
            "Size (TEU)": W_c[c],
            "Type": container_type,
            "Origin": origin,
            "Destination": destination,
            "Release Time": R_c[c],
            "Opening Time": O_c[c],
            "Closing Time": D_c[c]
        })

    df = pd.DataFrame(container_data)
    print("\nContainer Table:")
    print(tabulate(df, headers='keys', tablefmt='grid'))

def print_node_table(N, Z_cj, E, I):
    """
    Prints a professional table summarizing node properties.
    """
    node_data = []
    for node in N:
        # Count import containers originating at this node
        import_count = sum(1 for c in I if Z_cj[c][node] == 1)
        
        # Count export containers ending at this node
        export_count = sum(1 for c in E if Z_cj[c][node] == 1)
        
        node_data.append({
            "Node ID": f"Node {node}",
            "Import Containers Originating": import_count,
            "Export Containers Ending": export_count,
        })

    # Create a DataFrame for cleaner presentation
    df = pd.DataFrame(node_data)
    print("\nNode Table:")
    print(tabulate(df, headers='keys', tablefmt='grid'))


def print_barge_table(K, Qk, x_ijk, N, y_ijk, z_ijk):
    """
    Prints a professional table summarizing barge properties and the utilization for each ordered route.
    """
    barge_data = []
    for k in K:
        total_capacity = Qk[k]
        utilized_capacity = 0
        routes = []
        capacity_details = []

        # Collect routes and capacity usage
        for i in N:
            for j in N:
                if i != j and x_ijk[i, j, k].X > 0.5:  # If the barge travels from i to j
                    capacity_used = y_ijk[i, j, k].X + z_ijk[i, j, k].X
                    utilized_capacity += capacity_used
                    utilization_percent = (capacity_used / total_capacity) * 100 if total_capacity > 0 else 0
                    routes.append((f"Node {i} → Node {j}", capacity_used, utilization_percent))

        # Sort routes to ensure continuity
        sorted_routes = []
        if routes:
            sorted_routes.append(routes.pop(0))  # Start with the first route
            while routes:
                last_node = int(sorted_routes[-1][0].split("→")[-1].strip().split()[-1])
                for route in routes:
                    start_node = int(route[0].split("→")[0].strip().split()[-1])
                    if start_node == last_node:
                        sorted_routes.append(route)
                        routes.remove(route)
                        break

        # Prepare capacity details for printing
        for route, capacity_used, utilization_percent in sorted_routes:
            capacity_details.append({
                "Route": route,
                "Capacity Used": capacity_used,
                "Available Capacity": total_capacity,
                "Utilization (%)": f"{utilization_percent:.2f}"
            })


        # Print detailed capacity usage for each route
        if capacity_details:
            print(f"\nCapacity Details for Barge {k}:")
            capacity_df = pd.DataFrame(capacity_details)
            print(tabulate(capacity_df, headers="keys", tablefmt="grid"))

# Call the functions with appropriate data
print_container_table(C, W_c, R_c, O_c, D_c, E, I, Z_cj)
print_node_table(N, Z_cj, E, I)
if m.status == GRB.OPTIMAL:
    print_barge_table(K, Qk, x_ijk, N, y_ijk, z_ijk)

    # Call the function after the optimization
    plot_barge_displacements_off(T_ij_matrix, x_ijk, K, N)
    # plot_barge_displacements_aesthetic(T_ij_matrix, x_ijk, K, N)
    # function to plot the number of containers in each node    
    # plot_barge_displacements_with_node_info(T_ij_matrix, x_ijk, K, N, Z_cj, E, I)
    
