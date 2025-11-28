# build dataset for each instance
import random
import math
import networkx as nx
import numpy as np

Qk = [104, 99, 81, 52, 28] # TEU
H_b = [3700, 3600, 3400, 2800, 1800] # euros
H_t_40 = 200 # euros
H_t_20 = 140 # euros
Handling_time = 1/6 # hours (10 minutes)

def container_info(seed):
    random.seed(seed)

    C = random.randint(100,600) # number of containers
    N = random.randint(10,20) # number of terminals

    C_dict = {}

    for i in range(C):

        Rc = random.randint(0,24) # availability hours
        Dc = random.randint(24,196) # closing hours
        Oc = random.randint(Dc-120,Dc-24) # opening hours
        P_40 = random.uniform(0.75,0.9) # probability of 40ft container
        P_Export = random.uniform(0.05,0.7) # probability of export

        if random.random() < P_40:
            Wc = 2 # 40ft container
        else:
            Wc = 1 # 20ft container
        
        if random.random() < P_Export:
            In_or_Out = 2 # Export
            Terminal = random.randint(1,N) # assigned delivery terminal location
        else:
            In_or_Out = 1 # Import
            Terminal = random.randint(1,N) # assigned pickup terminal location
        
        C_dict[i] = {
            "Rc": Rc,
            "Dc": Dc,
            "Oc": Oc,
            "Wc": Wc,
            "In_or_Out": In_or_Out,
            "Terminal": Terminal
        }

    return C_dict, C, N

[C_dict, C, N] = container_info(100)

# Example output
#print(C_dict[56]["In_or_Out"]) # 1 for Import, 2 for Export

def T_ij(N):
    T_ij_list = np.zeros((N, N), dtype=int)

    number_of_sub_terminals = math.floor((N - 1) / 3)

    index_antwerp = list(range(1,number_of_sub_terminals+1)) # Antwerp sub-terminals
    index_rotterdam = list(range(number_of_sub_terminals+1, 2*number_of_sub_terminals+1)) # Rotterdam sub-terminals
    index_maasvlakte = list(range(2*number_of_sub_terminals+1,N)) # Maasvlakte sub-terminals

    if len(index_maasvlakte) == number_of_sub_terminals+2: # making sure the length of Maasvlakte is not too different to other two
        index_rotterdam.append(index_maasvlakte[0])
        index_maasvlakte = index_maasvlakte[1:]

    # print(index_antwerp, index_rotterdam, index_maasvlakte)

    for i in range(N):
        for j in range(N):
            if i == j:
                T_ij_list[i][j] = 0

            elif i in index_antwerp and j in index_antwerp:
                T_ij_list[i][j] = 1
            elif i in index_rotterdam and j in index_rotterdam:
                T_ij_list[i][j] = 1
            elif i in index_maasvlakte and j in index_maasvlakte:
                T_ij_list[i][j] = 1
                
            elif (i == 0 or j == 0) and (j in index_antwerp or i in index_antwerp):
                T_ij_list[i][j] = 13

            elif (i == 0 or j == 0) and ((j in index_maasvlakte or j in index_rotterdam) or (i in index_maasvlakte or i in index_rotterdam)):
                T_ij_list[i][j] = 11

            elif (i in index_antwerp or j in index_antwerp) and ((j in index_maasvlakte or j in index_rotterdam) or (i in index_maasvlakte or i in index_rotterdam)):
                T_ij_list[i][j] = 16

            elif (i in index_maasvlakte or j in index_maasvlakte) and (i in index_rotterdam or j in index_rotterdam):
                T_ij_list[i][j] = 4
    
            else:
                T_ij_list[i][j] = 666            

    return T_ij_list

T_ij_list = T_ij(N)
n = len(T_ij_list)

# Example output
# print(T_ij_list[1][5]) # 1 for Antwerp sub-terminal, 5 for Rotterdam sub-terminal

# Master route

G = nx.complete_graph(n)
for i in range(n):
    for j in range(n):
        if i != j:
            G[i][j]['weight'] = T_ij_list[i][j]

# Find approximate TSP cycle (returns to start)
master_route = nx.approximation.traveling_salesman_problem(G, cycle=True, weight='weight')

print("Master route:", master_route) # used to order containers for the greedy algorithm for the inital solution

C_ordered = []

for i in master_route[1:-1]:
    for j in range(C):
        if C_dict[j]["Terminal"] == i:
            C_ordered.append(j)

print("Ordered containers:", C_ordered) # ordered list of containers based on the master route

Barges = [104, 99, 81, 52, 28] # TEU

def get_route(L_current):

    route = [0]  # start at terminal 0 (main terminal)
    current_terminal = 0 # start at terminal 0 (main terminal)

    for c in L_current:
        if c["Terminal"] == current_terminal:
            continue
        else:
            route.append(c["Terminal"])
            current_terminal = c["Terminal"]
    
    return route

def get_timing(route, T_ij_list, handling_time, L_current):

    departure_time = 0

    current_max = 0
    dry_port_handling_time = 0

    for c in L_current:
        if c["In_or_Out"] == 2:
            dry_port_handling_time += handling_time

            if c["Rc"] > current_max:
                current_max = c["Rc"]
                

    departure_time = current_max + dry_port_handling_time

    D_terminal = [departure_time]  # time of departure from each terminal
    O_terminal = [0]  # time of arrival at each terminal

    current_terminal = 0  # start at terminal 0 (dry port)

    term_departure_time = departure_time  # start time at departure time
    term_arrival_time = 0  # start time at 0

    for terminal in route[1:]:

        travel_time = T_ij_list[current_terminal][terminal]
        handling_time_total = handling_time * sum(1 for c in L_current if c["Terminal"] == terminal)

        term_departure_time += travel_time # add travel time to next terminal
        term_departure_time += handling_time_total # add handling time at terminal

        term_arrival_time += travel_time + D_terminal[-1]  # arrival time is the same as departure time after handling

        D_terminal.append(term_departure_time)  # append the time of arrival at the terminal
        O_terminal.append(term_arrival_time)  # append the time of arrival at the terminal

        current_terminal = terminal
    
    return D_terminal, O_terminal

def check_for_cap(L_current, barges, idx):

    total_teu = sum(c["Wc"] for c in L_current)

    if total_teu <= barges[idx]:
        return True
    else:
        return False

f_ck_init = np.zeros((C, len(Barges)))  # feasibility check for each container and barge
print("Initial feasibility check matrix shape:", f_ck_init)

# for c in C_ordered:
#     L_current(c)

#     if check_for_cap(L_current, Barges, barge_idx):
#         continue
#     else:
#         barge_idx += 1