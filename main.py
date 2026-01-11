from MILP import MILP_Algo
from Greedy_Algo import GreedyOptimizer
from Meta_Heuristics import MetaHeuristic


def main(reduced=False):
    milp_instance = MILP_Algo(reduced=reduced)

    greedy = GreedyOptimizer(problem_instance=milp_instance)

    init_solution = greedy.solve_greedy()

    print("Initial greedy solution cost: €" + str(init_solution.total_cost))

    mh = MetaHeuristic(
        problem_instance=milp_instance,
        init_solution=init_solution,
        get_route=greedy.get_route,
        get_timing=greedy.get_timing,
        check_for_cap=greedy.check_for_cap,
        delay_window=greedy.delay_window,
    )

    mh.local_search()

    mh.display_final_allocations()

    return mh.best_cost


if __name__ == "__main__":
    final_cost = main(reduced=False)
    print(f"Final cost of the operations: €{final_cost}")
