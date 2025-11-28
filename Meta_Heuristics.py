"""
This module does work  #+#+# Gabo
"""



import random
import numpy as np
import matplotlib.pyplot as plt
import pulp
from Greedy_Algo import (
    get_route, get_timing, check_for_cap, delay_window,
    C_dict, Barges, T_ij_list, Handling_time, Qk, C_ordered, H_b, H_t_20, H_t_40
)


def repair_route(assigned, C_dict, Qk, Tij, Handling_time):
    """
    assigned: list of container‐ids assigned to this barge
    C_dict:   full container info, as in your dataset
    Qk:       capacity of this barge (TEU)
    Tij:      NxN travel‐time matrix
    Handling_time: hours per container
    returns:   a feasible route (list of nodes), or None if infeasible
    """

    # 1) build the set of nodes we must visit: depot=0 plus each unique terminal
    terminals = {0}
    for c in assigned:
        terminals.add(C_dict[c]['Terminal'])
    N = list(terminals)

    # pre‐compute pickup/drop sizes per node
    # at depot (0) we only unload exports, at sea terminals both
    p = {j:0 for j in N}   # pickups
    d = {j:0 for j in N}   # deliveries
    for c in assigned:
        j = C_dict[c]['Terminal']
        if C_dict[c]['In_or_Out']==1:
            p[j] += C_dict[c]['Wc']
        else:
            d[j] += C_dict[c]['Wc']

    # 2) create PuLP model
    prob = pulp.LpProblem("repair_route", pulp.LpMinimize)

    # 3) variables
    x = pulp.LpVariable.dicts(
        'x',
        [(i,j) for i in N for j in N if i!=j],
        cat='Binary'
    )
    t = pulp.LpVariable.dicts(
        't',
        N,
        lowBound=0,
        cat='Continuous'
    )

    # 4) objective: minimize total travel time
    prob += pulp.lpSum(Tij[i][j] * x[(i,j)] for (i,j) in x)

    # 5) flow‐balance constraints (Eq 3 & 20–21)
    # leave depot exactly once, return once
    prob += pulp.lpSum(x[(0,j)] for j in N if j!=0) == 1
    prob += pulp.lpSum(x[(i,0)] for i in N if i!=0) == 1
    # intermediate nodes: in=out
    for h in N:
        if h==0: continue
        prob += (
            pulp.lpSum(x[(i,h)] for i in N if i!=h) ==
            pulp.lpSum(x[(h,j)] for j in N if j!=h)
        )

    # 6) subtour‐elimination via MTZ (Miller–Tucker–Zemlin)
    # u[j] ordinal variable
    u = pulp.LpVariable.dicts('u', N, lowBound=0, upBound=len(N), cat='Integer')
    for i in N:
        for j in N:
            if i!=j and (i,j) in x:
                prob += u[i] + 1 <= u[j] + len(N)*(1 - x[(i,j)])

    # 7) capacity constraints (Eq 9)
    # we linearize by ensuring the maximum load at each node ≤ Qk
    # track load uload[j] at each node j:
    uload = pulp.LpVariable.dicts('load', N, lowBound=0, upBound=Qk, cat='Continuous')
    # at depot: load = total exports
    prob += uload[0] == pulp.lpSum(d[j] for j in N if j!=0)
    # flow conservation of load on each arc
    for i in N:
        for j in N:
            if i!=j and (i,j) in x:
                # uload[j] ≥ uload[i] - drop[j] + pick[j] - M*(1-x[i,j])
                M = Qk
                prob += (
                    uload[j] >= uload[i]
                                 - d[j]
                                 + p[j]
                                 - M*(1 - x[(i,j)])
                )

    # 8) time‐window constraints (Eq 11–14)
    bigM = 1e5
    # release date at depot = max export Rc
    R0 = max(C_dict[c]['Rc'] for c in assigned if C_dict[c]['In_or_Out']==2) if any(C_dict[c]['In_or_Out']==2 for c in assigned) else 0
    prob += t[0] >= R0
    for i in N:
        for j in N:
            if i!=j and (i,j) in x:
                # t[j] ≥ t[i] + handling_time*(#boxes at i) + Tij[i][j] - M(1 - x[i,j])
                service_i = Handling_time * (p[i] + d[i])
                prob += (
                    t[j] >= t[i] + service_i + Tij[i][j]
                             - bigM*(1 - x[(i,j)])
                )
    # and container TWs at each node
    for j in N:
        O_j = min(C_dict[c]['Oc'] for c in assigned if C_dict[c]['Terminal']==j) if any(C_dict[c]['Terminal']==j for c in assigned) else 0
        D_j = max(C_dict[c]['Dc'] for c in assigned if C_dict[c]['Terminal']==j) if any(C_dict[c]['Terminal']==j for c in assigned) else bigM
        prob += t[j] >= O_j
        prob += t[j] <= D_j

    # 9) solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))

    if pulp.LpStatus[prob.status] != 'Optimal':
        return None

    # 10) extract route
    # start from 0, follow arcs x[i,j]=1
    route = [0]
    cur = 0
    visited = {0}
    while True:
        for j in N:
            if j!=cur and pulp.value(x[(cur,j)]) > 0.5:
                route.append(j)
                cur = j
                break
        if cur == 0:
            break
    return route


class MetaHeuristic:
    def __init__(self,
                 C_ordered, C_dict,
                 Barges, H_b, H_t_20, H_t_40,
                 T_ij_list, Handling_time):
        # problem data
        self.C_ordered = list(C_ordered)
        self.C_dict    = C_dict

        # 1) compute slacks and pick top‐10% as critical
        slacks = {c: self.C_dict[c]['Dc'] - self.C_dict[c]['Oc'] 
                  for c in self.C_ordered}
        ncrit  = max(1, int(0.1 * len(self.C_ordered)))
        crit_sorted = sorted(slacks, key=slacks.get)
        self.critical = set(crit_sorted[:ncrit])

        # parameters
        self.truck_move_prob    = 0.6
        self.critical_move_prob = 0.6
        self.ten_crit           = 30

        self.Barges    = Barges
        self.H_b       = H_b
        self.H_t       = {1: H_t_20, 2: H_t_40}
        self.T_ij      = T_ij_list
        self.handling  = Handling_time

        # sizes
        self.C = len(C_dict)
        self.K = len(Barges)

        # solution representation
        self.f_ck = np.zeros((self.C, self.K), dtype=int)

        # tabu structures (move_key -> tenure)
        self.T1 = {}
        self.T2 = {}
        self.T3 = {}

        # parameters (tune these!)
        self.critical            = {}       # set of your “tightest” containers
        self.ten_move   = 20
        self.ten_crit   = 20
        self.ten_barban = 10
        self.shake_thr  = 100

    def initial_solution(self):
        """Greedy fill each barge in master‐order with up to one shift."""
        barge = 0
        to_ignore = []
        dep_delay = 0
        self.f_ck[:] = 0

        while barge < self.K:
            for c in self.C_ordered:
                if c in to_ignore:
                    continue

                # 1) assign
                self.f_ck[c,barge] = 1

                # 2) build load & cap‐check
                assigned = [i for i in range(self.C) if self.f_ck[i,barge]==1]
                Lcur = {i:self.C_dict[i] for i in assigned}
                route= get_route(Lcur)
                if not check_for_cap(route, Lcur, self.Barges, barge):
                    self.f_ck[c,barge] = 0
                    continue

                # 3) time‐window w/ up to one shift
                ok=False
                delay=dep_delay
                for attempt in (0,1):
                    D,O = get_timing(route, self.T_ij, self.handling, Lcur, delay)
                    viol=[v for v in Lcur.values() if not (
                        O[route.index(v["Terminal"])] <= v["Oc"] <= D[route.index(v["Terminal"])]
                     or O[route.index(v["Terminal"])] <= v["Dc"] <= D[route.index(v["Terminal"])]
                    )]
                    if not viol:
                        ok=True
                        break
                    if attempt==0:
                        delay += max(
                          delay_window(v,O,route,v["Terminal"],self.handling)
                          for v in viol
                        )
                if ok:
                    dep_delay = delay
                    to_ignore.append(c)
                else:
                    self.f_ck[c,barge] = 0

            barge += 1
            dep_delay = 0

    def _age_tabu(self):
        # decrement and purge expired tenures from T1, T2, T3
        for T in (self.T1, self.T2):
            expired = [m for m,t in T.items() if t<=1]
            for m in expired: del T[m]
            for m in T:       T[m] -= 1
        expired = [b for b,t in self.T3.items() if t<=1]
        for b in expired: del self.T3[b]
        for b in self.T3:  self.T3[b] -= 1


    def _shake(self):
        best_k=None; worst=1.0
        for k in range(self.K):
            if k in self.T3: continue
            assigned=[c for c in range(self.C) if self.f_ck[c,k]==1]
            if not assigned: continue
            Lcur={c:self.C_dict[c] for c in assigned}
            route=get_route(Lcur)
            loads=[]; load=sum(c["Wc"] for c in Lcur.values() if c["In_or_Out"]==2)
            loads.append(load)
            for node in route[1:]:
                for cont in Lcur.values():
                    if cont["Terminal"]==node:
                        load += (cont["Wc"] if cont["In_or_Out"]==1 else -cont["Wc"])
                loads.append(load)
            util=sum(loads)/(len(loads)*self.Barges[k])
            if util<worst: worst, best_k = util, k
        if best_k is not None:
            self.f_ck[:,best_k]=0
            self.T3[best_k]=self.ten_barban

    def operator_move(self):
        # 1) pick c: 60% of the time prefer already‐trucked “critical” candidates
        if random.random() < self.critical_move_prob:
            trucked = [c for c in range(self.C) if not any(self.f_ck[c])]
            # only pick a critical container if available
            crit_trucked = [c for c in trucked if c in self.critical]
            if crit_trucked:
                c = random.choice(crit_trucked)
            elif trucked:
                c = random.choice(trucked)
            else:
                c = random.randrange(self.C)
        else:
            c = random.randrange(self.C)

        # 2) locate its current barge (if any) and pick a new target
        from_b = next((k for k in range(self.K) if self.f_ck[c,k]), None)
        choices = list(range(self.K)) + ["truck"]
        to_b    = random.choice(choices)
        if to_b == from_b:
            return False

        move = (c, from_b, to_b)
        # 3) check all tabu‐lists
        if move in self.T1 or move in self.T2 or (from_b in self.T3) or (to_b in self.T3):
            return False

        # 4) tentatively apply
        old_row = self.f_ck[c].copy()
        if from_b  is not None:      self.f_ck[c, from_b] = 0
        if to_b    != "truck":       self.f_ck[c, to_b]    = 1

        # 5) quick capacity + 1-shift TW check on receiving barge
        feasible = True
        if to_b != "truck":
            assigned = [i for i in range(self.C) if self.f_ck[i,to_b] == 1]
            Lcur     = {i:self.C_dict[i] for i in assigned}
            route    = get_route(Lcur)

            # capacity
            if not check_for_cap(route, Lcur, self.Barges, to_b):
                feasible = False
            else:
                # up to one departure shift
                delay = 0
                fit   = False
                for attempt in (0,1):
                    D,O = get_timing(route, self.T_ij, self.handling, Lcur, delay)
                    viol = [
                        v for v in Lcur.values()
                        if not (
                            O[route.index(v["Terminal"])] <= v["Oc"] <= D[route.index(v["Terminal"])]
                         or O[route.index(v["Terminal"])] <= v["Dc"] <= D[route.index(v["Terminal"])]
                        )
                    ]
                    if not viol:
                        fit = True
                        break
                    if attempt == 0:
                        delay = max(
                            delay_window(v, O, route, v["Terminal"], self.handling)
                            for v in viol
                        )
                if not fit:
                    feasible = False

        # 6) if quick check failed, try full MILP repair for barge
        if not feasible and to_b != "truck":
            assigned = [i for i in range(self.C) if self.f_ck[i,to_b] == 1]
            new_route = repair_route(
                assigned,
                self.C_dict,
                self.Barges[to_b],
                self.T_ij,
                self.handling
            )
            if new_route is None:
                # unrecoverably infeasible → undo + T1‐tabu
                self.f_ck[c] = old_row
                self.T1[move] = self.ten_move
                return False
            # else: repair succeeded, we keep the move

        # 7) at this point move is accepted
        #    if it was a critical→truck, tabu it in T2
        if to_b == "truck" and c in self.critical:
            self.T2[move] = self.ten_crit

        return True



    def operator_swap(self):
        c1, c2 = random.sample(range(self.C), 2)
        bs1     = [k for k in range(self.K) if self.f_ck[c1,k]]
        bs2     = [k for k in range(self.K) if self.f_ck[c2,k]]
        if not bs1 or not bs2 or bs1[0] == bs2[0]:
            return False
        b1, b2 = bs1[0], bs2[0]

        move = ((c1,b1,b2),(c2,b2,b1))
        if move in self.T1 or b1 in self.T3 or b2 in self.T3:
            return False

        # tentatively swap
        old1 = self.f_ck[c1].copy()
        old2 = self.f_ck[c2].copy()
        self.f_ck[c1,b1] = 0; self.f_ck[c1,b2] = 1
        self.f_ck[c2,b2] = 0; self.f_ck[c2,b1] = 1

        def barge_ok(k):
            assigned = [i for i in range(self.C) if self.f_ck[i,k]]
            if not assigned:
                return True
            Lcur  = {i:self.C_dict[i] for i in assigned}
            route = get_route(Lcur)
            if not check_for_cap(route, Lcur, self.Barges, k):
                return False
            # one‐shift TW
            delay = 0
            for attempt in (0,1):
                D,O = get_timing(route, self.T_ij, self.handling, Lcur, delay)
                viol = [v for v in Lcur.values() if not (
                    O[route.index(v["Terminal"])] <= v["Oc"] <= D[route.index(v["Terminal"])]
                 or O[route.index(v["Terminal"])] <= v["Dc"] <= D[route.index(v["Terminal"])]
                )]
                if not viol:
                    return True
                if attempt == 0:
                    delay = max(delay_window(v, O, route, v["Terminal"], self.handling)
                                for v in viol)
                else:
                    return False
            return False

        ok1 = barge_ok(b1)
        ok2 = barge_ok(b2)
        if ok1 and ok2:
            return True

        # quick check failed on at least one barge: call repair on each
        for b in (b1,b2):
            assigned  = [i for i in range(self.C) if self.f_ck[i,b]]
            new_route = repair_route(
                assigned,
                self.C_dict,
                self.Barges[b],
                self.T_ij,
                self.handling
            )
            if new_route is None:
                # irreparable swap → undo + tabu
                self.f_ck[c1] = old1
                self.f_ck[c2] = old2
                self.T1[move] = self.ten_move
                return False

        # both repairs succeeded
        return True


    def evaluate(self):
        total_cost=0; total_stops=0; utils=[]
        for k in range(self.K):
            assigned=np.where(self.f_ck[:,k]==1)[0].tolist()
            if not assigned: continue
            total_cost+=self.H_b[k]
            Lcur={c:self.C_dict[c] for c in assigned}
            route=get_route(Lcur)
            # travel times
            for i in range(len(route)-1):
                total_cost+=self.T_ij[route[i]][route[i+1]]
            stops=len(route)-1
            total_stops+=stops
            total_cost+=stops  # 1€/stop penalty
            # util
            load=sum(c["Wc"] for c in Lcur.values() if c["In_or_Out"]==2)
            loads=[load]
            for node in route[1:]:
                for cont in Lcur.values():
                    if cont["Terminal"]==node:
                        load += (cont["Wc"] if cont["In_or_Out"]==1 else -cont["Wc"])
                loads.append(load)
            utils.append(sum(loads)/(len(loads)*self.Barges[k]))
        # truck
        unassigned=np.where(self.f_ck.sum(axis=1)==0)[0]
        for c in unassigned:
            total_cost+=self.H_t[self.C_dict[c]["Wc"]]
        return total_cost, total_stops, (sum(utils)/len(utils) if utils else 0)

    def local_search(self, max_iters=3000):
        self.initial_solution()
        self.best_cost, _, _ = self.evaluate()
        best_f = self.f_ck.copy()
        no_improve = 0

        self.it_list = []
        self.cost_list = []
        self.best_cost_list = []

        # # Set up interactive plotting
        # plt.ion()
        # fig, ax = plt.subplots()
        # line1, = ax.plot([], [], label='Current Cost')
        # line2, = ax.plot([], [], label='Best Cost', linestyle='--')
        # ax.set_xlabel('Iteration')
        # ax.set_ylabel('Cost')
        # ax.set_title('Meta-Heuristic Cost Over Iterations')
        # ax.legend()
        # ax.grid(True)

        for it in range(max_iters):
            if it % 100 == 0:
                print(f"Iteration {it}, Percent Complete: {100*it/max_iters:.1f}%")
            if random.random() < 0.8:
                moved = self.operator_move()
            else:
                moved = self.operator_swap()

            # *always* age your tabu after every move‐attempt
            self._age_tabu()

            if not moved:
                continue

            cost, _, _ = self.evaluate()

            if cost < self.best_cost:
                self.best_cost, best_f = cost, self.f_ck.copy()
                no_improve = 0
            else:
                self.f_ck = best_f.copy()
                no_improve += 1

            if no_improve >= self.shake_thr:
                self._shake()
                no_improve = 0

            self.it_list.append(it)
            self.cost_list.append(cost)
            self.best_cost_list.append(self.best_cost)

        #     # Update plot every iteration
        #     line1.set_data(self.it_list, self.cost_list)
        #     line2.set_data(self.it_list, self.best_cost_list)
        #     ax.relim()
        #     ax.autoscale_view()
        #     fig.canvas.draw()
        #     fig.canvas.flush_events()

        # plt.ioff()
        self.f_ck = best_f
        return self.best_cost, self.it_list, self.cost_list, self.best_cost_list

    def display_final_allocations(self):
        """Display the final allocation of containers to barges and trucks"""
        print("\n" + "="*80)
        print("FINAL CONTAINER-BARGE ALLOCATIONS")
        print("="*80)
        
        # Track which containers are assigned to each barge
        barge_assignments = {k: [] for k in range(self.K)}
        trucked_containers = []
        
        for c in range(self.C):
            assigned = False
            for k in range(self.K):
                if self.f_ck[c, k] == 1:
                    barge_assignments[k].append(c)
                    assigned = True
                    break
            if not assigned:
                trucked_containers.append(c)
        
        # Display barge assignments
        total_barge_cost = 0
        total_containers_on_barges = 0
        
        for k in range(self.K):
            containers = barge_assignments[k]
            if containers:
                print(f"\nBARGE {k+1} (Capacity: {self.Barges[k]} TEU, Fixed cost: €{self.H_b[k]}):")
                print(f"  Assigned containers: {len(containers)}")
                
                # Calculate total TEU and separate imports/exports
                total_teu = 0
                imports = []
                exports = []
                terminals_visited = set()
                
                for c in containers:
                    container_info = self.C_dict[c]
                    total_teu += container_info['Wc']
                    terminals_visited.add(container_info['Terminal'])
                    
                    if container_info['In_or_Out'] == 1:  # Import
                        imports.append(c)
                    else:  # Export
                        exports.append(c)
                
                print(f"  Total TEU load: {total_teu}/{self.Barges[k]} ({100*total_teu/self.Barges[k]:.1f}% utilization)")
                print(f"  Terminals visited: {sorted(terminals_visited)}")
                print(f"  Import containers: {len(imports)} containers")
                print(f"  Export containers: {len(exports)} containers")
                
                if len(containers) <= 20:  # Only show detailed list for small assignments
                    print(f"  Container IDs: {containers}")
                
                total_barge_cost += self.H_b[k]
                total_containers_on_barges += len(containers)
        
        # Display trucked containers
        print(f"\nTRUCKED CONTAINERS:")
        print(f"  Number of containers: {len(trucked_containers)}")
        
        if trucked_containers:
            truck_cost_20ft = 0
            truck_cost_40ft = 0
            
            for c in trucked_containers:
                container_info = self.C_dict[c]
                if container_info['Wc'] == 1:  # 20ft
                    truck_cost_20ft += self.H_t[1]
                else:  # 40ft
                    truck_cost_40ft += self.H_t[2]
            
            total_truck_cost = truck_cost_20ft + truck_cost_40ft
            print(f"  20ft containers: {sum(1 for c in trucked_containers if self.C_dict[c]['Wc'] == 1)}")
            print(f"  40ft containers: {sum(1 for c in trucked_containers if self.C_dict[c]['Wc'] == 2)}")
            print(f"  Total trucking cost: €{total_truck_cost}")
            
            if len(trucked_containers) <= 30:  # Only show detailed list for reasonable numbers
                print(f"  Trucked container IDs: {trucked_containers}")
        
        # Summary
        print(f"\n" + "-"*50)
        print("SUMMARY:")
        print(f"  Total containers: {self.C}")
        print(f"  Containers on barges: {total_containers_on_barges}")
        print(f"  Containers trucked: {len(trucked_containers)}")
        print(f"  Barges used: {sum(1 for k in range(self.K) if barge_assignments[k])}")
        print(f"  Final cost: €{self.best_cost}")
        print("="*80)


if __name__ == "__main__":
    # usage
    mh = MetaHeuristic(C_ordered, C_dict, Barges,
                       H_b, H_t_20, H_t_40,
                       T_ij_list, Handling_time)
    mh.initial_solution()
    print("greedy cost:", mh.evaluate()[0])
    mh.local_search()
    print("meta‐heuristic cost:", mh.best_cost)

    # Display final allocations
    mh.display_final_allocations()