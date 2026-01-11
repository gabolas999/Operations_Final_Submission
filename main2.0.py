from MILP import MILP_Algo
from Greedy_Algo import GreedyOptimizer
from Meta_Heuristics import MetaHeuristic


def main(reduced=False):
    milp_instance = MILP_Algo(reduced=reduced)

    greedy = GreedyOptimizer(problem_instance=milp_instance)

    init_solution = greedy.solve_greedy()

    mh = MetaHeuristic(problem_instance=milp_instance, init_solution=init_solution)

    mh.local_search()

    mh.display_final_allocations()

    return mh.best_cost


if __name__ == "__main__":
    final_cost = main(reduced=False)
    print(f"Final cost of the operations: €{final_cost}")
