'''Due to the practical relevance of the proposed problem, the instances are based on the case of a Dutch barge operator active in the
Port of Rotterdam region, who provided data (2016–2018) on handled container. The dry port is located in the North Brabant, a
province in the south of the Netherlands, about 120 km away from the Port of Rotterdam, with an average travel time for barges of
11 h.
The Port of Rotterdam seaport consists of several terminals. The data set considers 36 terminals within the port. Fig. 1 shows two
main agglomerations of terminals where the majority of the containers is handled, and in particular: the closest area to the North Sea
(Maasvlakte terminals), and the City terminal located further to the east. For the distances, we refer to (Fazi et al., 2015).
The barge operator (BO) manages a fleet of barges consisting of 5 privately owned small-medium size vessels. Capacities in TEU
are: 104, 99, 81, 52, 28. The BO has contracts with the barge owners to hold regular services between the sea port and the dry port. '''

'''The BO estimated that the rental cost for each round trip is respectively: €3700, €3600, €3400, €2800, €1800. These costs reflect an
economy of scale. The BO also owns a fleet of truck. Fixed average cost for trucking a 40-foot container is set to €200, and for a 20-
foot container to €140. Note that these costs are based on the guidelines of the barge operator and are purely indicative of the real
cost. However, they are in line with the costs used in previous research (Behdani et al., 2014).
In the first experiment, we test the performance of the meta-heuristic with 20 randomly generated instances, considering solely
transportation costs. The parameters related to the containers follow the trends in the available data set. In particular, for each
instance, we set the number of containers to be random between 100 and 600, and the number of terminals between 10 and 20. The
terminal locations are assigned randomly.
In the data set, the number of export containers may vary significantly week by week, whereas import containers have a more
steady presence. To consider this, for each instance we randomly generated a number within
[0.05, 0.7]
. Then, this number is the
probability for each container to be export in that instance. The same approach is used to assign the size, either 20 or 40 foot, to the
containers. The probability to be a 40-foot container is in the range
[0.75, 0.9]
. For each instance, time features in hours are randomly
chosen from selected weeks from the data set and normalized, considering time 0 to be the start of planning period. Advance
information on the availability of export containers at the inland terminal typically does not exceed 24 h. Due dates are in the range
[24, 196]
. The opening dates, when not available in the data set, were randomly generated between 2 and 5 days before the due date.
Finally, handling time per container is set to 10 min.
With concern to the second experiment, we draw 5 instances from the data set. The data matches handled containers in selected
weeks where complete data is available, including the final implemented decisions of the planners. In this regard, alphanumerical
codes are assigned to barged containers, indicating import and export voyages, the used barge, and the sequential voyage number. In
this way, we are able to generate insights on the planning procedures. We summarize the data used for both experiments in Table 2'''

#Distances from Faxi et al. 2015: Dryport is in Veghel, Netherlands
#Travelling distances considering barge travel times: 
'''0 h for same quay, 1 h within same sea terminal
4 h between Maasvlakte and Rotterdam City Terminal
11 h from Veghel to Maasvlakte and Rotterdam City Terminal'''

import random
random.seed(100)
from gurobipy import Model, GRB, quicksum

Qk = [104, 99, 81, 52, 28] #[TEU] (Twenty-foot Equivalent Unit)
HkB = [3700, 3600, 3400, 2800, 1800] #[euros] (per round trip)
Ht_40 = 20000000000 #[euros] (40-foot container)
Ht_20 = 14000000000 #[euros] (20-foot container)
L = 10 #[min] (handling time per container)
Gamma = 1000 #[euros] penalty for sea terminal visit (TUNEEEEEEE!!!!!!!!!!!!!!!!!!!!!!!!)
M = 100000 #big M

#make list of time from and to all terminals
def Tij(N): # i is origin terminal and j is destination terminal, Tij is the travel time between terminals i and j in [minutes] (from Fazi et al., 2015)
    
    T_ij_matrix = [[0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        for j in range(N):
            if i == 0 and i != j: #from dryport to seaport (exports)
                T_ij = 11*60
                # T_ij = 6*80
            elif j == 0 and i != j: #from seaport to dryport (Basically imports)
                T_ij = 11*60
                #T_ij = 6*80
            elif i == j and i != 0: #intra terminal travel time
                T_ij = 0
            elif i == j and i == 0: #intra dryport travel time
                T_ij = 0
            else: #between city and maasvlakt terminals travel time
                T_ij = 4*60
            T_ij_matrix[i][j] = T_ij

    return T_ij_matrix

def generate_instance():
    
    C = random.randint(100, 400) #number of containers
    N = random.randint(10, 20) #number of terminals

    C = random.randint(100, 150) #number of containers
    N = random.randint(3, 6) #number of terminals

    K_truck = 1 #number of trucks
    K_barge = 5 #number of barges
    #Trucks will always be the last X indices in the "K" set eg. 5 barges and 2 trucks barges indices = 0,1,2,3,4 and trucks indicies = 5,6
    K = [i for i in range(K_truck + K_barge)] #total number of vehicles

    C = [i for i in range(C)] #container index
    N = [i for i in range(N)] #terminal index
    E = [] #container index that are export
    I = [] #container index that are import
    W_c = [] #container size in order of container index
    Z_cj = [[0 for _ in range(len(N))] for _ in range(len(C))] #1 if container c is destined to at dock j, 0 otherwise
    # Z_ci = [[0 for _ in range(len(N))] for _ in range(len(C))] #1 if container c originates from dock i, 0 otherwise
    R_c = []
    D_c = []
    O_c = []
    H_T = []

    for container_idx in C:
        #probability for each container to be export
        p_export = random.uniform(0.05, 0.7)
        is_export = "E" if random.random() < p_export else "I"

        if is_export == "E":
            E.append(container_idx)
        else:
            I.append(container_idx)
        
        #probability to be a 40-foot container
        p_40 = random.uniform(0.75, 0.9)
        Size = 2 if random.random() < p_40 else 1

        W_c.append(Size)

        # H_T definition based on the probabilities of container types
        if Size == 1:
            H_T.append(Ht_20)
        else:
            H_T.append(Ht_40)

        #Release, closing and opening dates in hours, starting from hour 0 on Monday
        Release = random.randint(0, 24*60) #release date container
        Closing = random.randint(24*60, 196*60) #closing date container
        Opening = random.randint(max(0, Closing-120*60), Closing-24*60) #open date container

        R_c.append(Release)
        D_c.append(Closing)
        O_c.append(Opening)


        j = random.randint(1, len(N)-1)
        Z_cj[container_idx][j] = 1

        

    instance = [C, N, E, I, K, Z_cj, W_c, R_c, D_c, O_c, H_T]

    return instance

instance_list = []

Number_of_instances = 1

for i in range(0, Number_of_instances):
    x = generate_instance()
    instance_list.append(x)

choice_of_instance = random.choice(instance_list)

C = choice_of_instance[0]
N = choice_of_instance[1]
E = choice_of_instance[2]
I = choice_of_instance[3]
K = choice_of_instance[4]
Z_cj = choice_of_instance[5]
W_c = choice_of_instance[6]
R_c = choice_of_instance[7]
D_c = choice_of_instance[8]
O_c = choice_of_instance[9]
H_T = choice_of_instance[10]
K_b = [i for i in K if i < len(K) - 1]
K_t = K[len(K_b):]



print("C:\n", C)
print("N:\n", N)
print("E:\n", E)
print("I:\n", I)
print("K:\n", K)
print("K_b:\n", K_b)
print("K_t:\n", K_t)
print("Z_cj:\n", Z_cj)
print("W_c:\n", W_c)
print("R_c:\n", R_c)
print("D_c:\n", D_c)
print("O_c:\n", O_c)
print("H_T:\n", H_T)



#make matrix of time from and to all terminals
T_ij_matrix = Tij(len(N))

# print(T_ij_matrix)

# VISUALIZE NODE LOCATIONS IN 2D SPACE

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS




# Assuming the model has been solved and the values of `x_ijk`, `y_ijk`, and `z_ijk` are available
# plot_nodes_with_cargo_info(T_ij_matrix, Z_cj, x_ijk, y_ijk, z_ijk, node_names, E, I)


###################
### MODEL SETUP ###
###################

# Initialize empty model

m = Model("Barge Scheduling")

#Create decision variables

f_ck = m.addVars(C, K, vtype = GRB.BINARY, name = "f_ck") #Binary variable, equals 1 if container c is allocated to means k

x_ijk = m.addVars(N, N, K_b, vtype = GRB.BINARY, name = "x_ijk") #Binary variable, equals 1 if barge k sails from terminal i to j

p_jk = m.addVars(N, K_b, vtype = GRB.INTEGER, name = "p_jk") #Import quantity loaded by barge k at sea terminal j 

d_jk = m.addVars(N, K_b, vtype = GRB.INTEGER, name = "d_jk") #Export quantity unloaded by barge k at sea terminal j

y_ijk = m.addVars(N, N, K_b, lb = 0, vtype = GRB.INTEGER, name = "y_ijk") #Import quantity carried by barge k from terminal i to j

z_ijk = m.addVars(N, N, K_b, lb = 0, vtype = GRB.INTEGER, name = "z_ijk") #Export quantity carried by barge k from terminal i to j

t_jk = m.addVars(N, K_b, lb = 0, vtype = GRB.INTEGER, name = "t_jk") #Time barge k is at terminal j 

m.update()


#Objective function

objective = quicksum(f_ck[c,K_t[0]] * H_T[c] for c in C) + \
            quicksum(x_ijk[0, j, k] * HkB[k] for k in K_b for j in N if j != 0) + \
            quicksum(T_ij_matrix[i][j] * x_ijk[i, j, k] for k in K_b for i in N for j in N if i != j) + \
            quicksum(Gamma * x_ijk[i, j, k] for k in K_b for i in N[1:] for j in N if i != j)

m.setObjective(objective, GRB.MINIMIZE)

for c in C:
    m.addConstr(quicksum(f_ck[c, k] for k in K) == 1, name=f"Container_Assignment_{c}")

for i in N:
    for k in K_b:
        m.addConstr(
            quicksum(x_ijk[i, j, k] for j in N if j != i) - quicksum(x_ijk[j, i, k] for j in N if j != i) == 0,
            name=f"Flow_Conservation_{i}_{k}"
        )

for k in K_b:
    m.addConstr(quicksum(x_ijk[0, j, k] for j in N if j != 0) <= 1, name=f"Barge_Visit_Limit_{k}")


for k in K_b:
    for j in N[1:]:  # Exclude dry port (j=0)
        m.addConstr(
            p_jk[j, k] == quicksum(W_c[c] * Z_cj[c][j] * f_ck[c, k] for c in I),
            name=f"Import_Quantity_{j}_{k}"
        )

for k in K_b:
    for j in N[1:]:  # Exclude dry port (j=0)
        m.addConstr(
            d_jk[j, k] == quicksum(W_c[c] * Z_cj[c][j] * f_ck[c, k] for c in E),
            name=f"Export_Quantity_{j}_{k}"
        )

for j in N[1:]: # Exclude dry port (j=0)
    for k in K_b:
        m.addConstr(
            quicksum(y_ijk[j, i, k] for i in N if i != j) - quicksum(y_ijk[i, j, k] for i in N if i != j) == p_jk[j, k],
            name=f"Import_Balance_{j}_{k}"
        )

for j in N[1:]: # Exclude dry port (j=0)
    for k in K_b:
        m.addConstr(
            quicksum(z_ijk[i, j, k] for i in N if i != j) - quicksum(z_ijk[j, i, k] for i in N if i != j) == d_jk[j, k],
            name=f"Export_Balance_{j}_{k}"
        )

for i in N:
    for j in N:
        for k in K_b:
            m.addConstr(
                y_ijk[i, j, k] + z_ijk[i, j, k] <= Qk[k] * x_ijk[i, j, k],
                name=f"Flow_Capacity_{i}_{j}_{k}"
            )

for c in E:
    for k in K_b:
        m.addConstr(
            t_jk[0, k] >= R_c[c] * f_ck[c, k],
            name=f"Vehicle_Departure_{c}_{k}"
        )

for i in N:
    for j in N:
        for k in K_b:
            m.addConstr(
                t_jk[j, k] >= t_jk[i, k] + quicksum(L*Z_cj[c][i] * f_ck[c, k] for c in C) + T_ij_matrix[i][j] - (1 - x_ijk[i, j, k]) * M,
                name=f"Time_LowerBound_{i}_{j}_{k}"
            )
            m.addConstr(
                t_jk[j, k] <= t_jk[i, k] + quicksum(L*Z_cj[c][i] * f_ck[c, k] for c in C) + T_ij_matrix[i][j] + (1 - x_ijk[i, j, k]) * M,
                name=f"Time_UpperBound_{i}_{j}_{k}"
            )

for c in C:
    for j in N[1:]:  # Exclude dry port (j=0)
        for k in K_b:
            m.addConstr(
                t_jk[j, k] >= O_c[c] * Z_cj[c][j] - (1 - f_ck[c, k]) * M,
                name=f"Export_Time_{c}_{j}_{k}"
            )

for c in C:
    for j in N[1:]:  # Exclude dry port (j=0)
        for k in K_b:
            m.addConstr(
                t_jk[j, k] * Z_cj[c][j] <= D_c[c] + (1 - f_ck[c, k]) * M,
                name=f"Demand_Fulfillment_{c}_{j}_{k}"
            )

m.update()

# .write('model_formulation.lp')

# m.printStats()


m.optimize()




# Check if an optimal solution was found
if m.status == GRB.OPTIMAL:
    print("\n\n\n\nOptimal solution found!")
    print(f"Optimal objective value: {m.objVal}")

    # Extracting the values of decision variables:
    # 1. f_ck: Assignment of containers to barges
    print("\nContainer to Barge Assignments (f_ck):")
    print(f"Length of set of containers {len(K)}")
    for c in C:
        for k in K:
            if f_ck[c, k].X > 0.5:  # If the variable is 1 (assigned)
                if k==5:
                    print(f"Container {c} is assigned to Truck {k}")
                else:
                    print(f"Container {c} is assigned to Barge {k}")

    # 2. x_ijk: Travel decisions of barges between terminals
    print("\nBarge Travel Decisions (x_ijk):")
    for i in N:
        for j in N:
            if i != j:
                for k in K_b:
                    if x_ijk[i, j, k].X > 0.5:  # If the variable is 1 (traveling)
                        print(f"\nBarge {k} travels from Terminal {i} to Terminal {j}")
                    # if x_ijk[i, j, k].X > 0.5:  # If the variable is 1 (traveling)
                    #     print(f"Barge {k} travels from Terminal {i} to Terminal {j}")
    m.printAttr("X")
else:
    print("No optimal solution found. Status:", m.status)



print(f"Optimal objective function value: {m.objVal}")



for row in T_ij_matrix:
    print(row)



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


plot_nodes(T_ij_matrix)


