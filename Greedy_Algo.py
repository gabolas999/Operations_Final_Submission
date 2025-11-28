#!/usr/bin/env python3
"""
Greedy Algorithm for Container Allocation Optimization

This module provides a unified class-based implementation of the greedy algorithm
for container-to-barge allocation optimization.

This file does work #+#+# Gabo
"""

import random
import math
import networkx as nx
import numpy as np


class GreedyOptimizer:
    """Unified greedy algorithm class for container allocation optimization"""
    
    def __init__(self, qk, h_b, seed=0, reduced=False, h_t_40=200, h_t_20=140, handling_time=1/6):
        """
        Initialize the greedy optimizer
        
        Parameters:
        -----------
        qk : list
            Barge capacities in TEU
        h_b : list
            Barge fixed costs in euros
        seed : int
            Random seed for instance generation
        reduced : bool
            If True, generate smaller instances for testing
        h_t_40 : float
            40ft container trucking cost in euros
        h_t_20 : float
            20ft container trucking cost in euros
        handling_time : float
            Container handling time in hours
        """
        # Parameters
        self.seed = seed
        self.reduced = reduced
        self.Qk = qk #if qk is not None else [104, 99, 81, 52, 28]  # TEU
        self.H_b = h_b #if h_b is not None else [3700, 3600, 3400, 2800, 1800]  # euros
        self.H_t_40 = h_t_40  # euros
        self.H_t_20 = h_t_20  # euros
        self.Handling_time = handling_time  # hours
        
        # Instance data - will be populated by generate_instance()
        self.C_dict = {}
        self.C = 0
        self.N = 0
        self.T_matrix = []  # Travel time matrix (N x N)
        self.Barges = []
        self.C_ordered = []
        self.master_route = []
        
        # Results storage
        self.f_ck_init = None
        self.route_list = []
        self.barge_departure_delay = []
        self.trucked_containers = {}
        self.total_cost = 0
        self.truck_cost = 0
        self.barge_cost = 0
        self.xijk = None
        
        # Generate instance automatically
        self.generate_instance()
    
    def generate_container_info(self):
        """Generate container information based on seed and reduced flag"""
        random.seed(self.seed)

        if self.reduced:
            self.C = random.randint(33, 200)  # number of containers
            self.N = random.randint(4, 8)     # number of terminals
        else:
            self.C = random.randint(100, 600)  # number of containers
            self.N = random.randint(10, 20)    # number of terminals

        self.C_dict = {}

        for i in range(self.C):
            Dc = random.randint(24, 196)  # closing hours
            Oc = random.randint(Dc-120, Dc-24)  # opening hours
            P_40 = random.uniform(0.75, 0.9)  # probability of 40ft container
            P_Export = random.uniform(0.05, 0.7)  # probability of export

            if random.random() < P_40:
                Wc = 2  # 40ft container
            else:
                Wc = 1  # 20ft container
            
            if random.random() < P_Export:
                In_or_Out = 2  # Export
                Rc = random.randint(0, 24)
                Terminal = random.randint(1, self.N-1)  # assigned delivery terminal location
            else:
                In_or_Out = 1  # Import
                Rc = 0
                Terminal = random.randint(1, self.N-1)  # assigned pickup terminal location
            
            self.C_dict[i] = {
                "Rc": Rc,
                "Dc": Dc,
                "Oc": Oc,
                "Wc": Wc,
                "In_or_Out": In_or_Out,
                "Terminal": Terminal
            }
    
    def generate_travel_times(self):
        """Generate travel time matrix T_matrix"""
        self.T_matrix = np.zeros((self.N, self.N), dtype=int)

        number_of_sub_terminals = math.floor((self.N - 1) / 3)

        index_antwerp = list(range(1, number_of_sub_terminals+1))  # Antwerp sub-terminals
        index_rotterdam = list(range(number_of_sub_terminals+1, 2*number_of_sub_terminals+1))  # Rotterdam sub-terminals
        index_maasvlakte = list(range(2*number_of_sub_terminals+1, self.N))  # Maasvlakte sub-terminals

        if len(index_maasvlakte) == number_of_sub_terminals+2:  # making sure the length of Maasvlakte is not too different to other two
            index_rotterdam.append(index_maasvlakte[0])
            index_maasvlakte = index_maasvlakte[1:]

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    self.T_matrix[i][j] = 0
                elif i in index_antwerp and j in index_antwerp:
                    self.T_matrix[i][j] = 1
                elif i in index_rotterdam and j in index_rotterdam:
                    self.T_matrix[i][j] = 1
                elif i in index_maasvlakte and j in index_maasvlakte:
                    self.T_matrix[i][j] = 1
                elif (i == 0 or j == 0) and (j in index_antwerp or i in index_antwerp):
                    self.T_matrix[i][j] = 13
                elif (i == 0 or j == 0) and ((j in index_maasvlakte or j in index_rotterdam) or (i in index_maasvlakte or i in index_rotterdam)):
                    self.T_matrix[i][j] = 11
                elif (i in index_antwerp or j in index_antwerp) and ((j in index_maasvlakte or j in index_rotterdam) or (i in index_maasvlakte or i in index_rotterdam)):
                    self.T_matrix[i][j] = 16
                elif (i in index_maasvlakte or j in index_maasvlakte) and (i in index_rotterdam or j in index_rotterdam):
                    self.T_matrix[i][j] = 4
                else:
                    self.T_matrix[i][j] = 666
    
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
    
    def generate_instance(self):
        """Generate complete problem instance"""
        self.generate_container_info()
        self.generate_travel_times()
        self.generate_master_route()
        self.generate_ordered_containers()
        self.Barges = self.Qk.copy()  # Set barge capacities
    
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
        print(f"Total cost: {self.total_cost:.0f} Euros")
        print(f"Barge cost: {self.barge_cost:.0f} Euros           ({self.barge_cost / self.total_cost * 100:.1f}%)")
        print(f"Truck cost: {self.truck_cost:.0f} Euros           ({self.truck_cost / self.total_cost * 100:.1f}%)")
        print(f"Containers: {self.C}")
        print(f"Terminals: {self.N}")
        print(f"Trucked containers: {len(self.trucked_containers)}           ({len(self.trucked_containers) / self.C * 100:.1f}%)")
        
        # Print barge utilization
        for k, route in enumerate(self.route_list):
            if len(route) > 1:  # Only print if barge is used
                containers_on_barge = sum(1 for c in range(self.C) if self.f_ck_init[c][k] == 1)
                teu_on_barge = sum(self.C_dict[c]["Wc"] for c in range(self.C) if self.f_ck_init[c][k] == 1)
                print(f"Barge {k}: {containers_on_barge} containers, {teu_on_barge}/{self.Barges[k]} TEU")

    @property
    def T_ij_list(self):
        """Alias for T_matrix for backward compatibility"""
        return self.T_matrix


qk = [     # Barge capacities in TEU
        104,        # Barge 0
        99,         # Barge 1
        81,         # Barge 2
        52,         # Barge 3
        28,         # Barge 4
    ]


h_b = [     # Barge fixed costs in euros
    3700,      # Barge 0
    3600,      # Barge 1
    3400,      # Barge 2
    2800,      # Barge 3
    1800,      # Barge 4
]



# Create global instance for backward compatibility
_global_optimizer = GreedyOptimizer(qk, h_b)

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
    optimizer = GreedyOptimizer(qk, h_b)
    results = optimizer.solve_greedy()
    optimizer.print_results()