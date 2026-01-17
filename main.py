from pathlib import Path
from MILP import MILP_Algo
from Greedy_Algo import GreedyOptimizer
from Meta_Heuristics import MetaHeuristic

SCENARIO_SETTINGS_PATH_DEFAULT = Path(
    "./Storage/Settings/settings________2025_12_22_18_01_10.toml"
)


def toml_to_input_dict(toml_path: str) -> dict:
    import toml

    with open(toml_path, "r") as f:
        input_dict = toml.load(f)

    return input_dict


def main(
    scenario_path=SCENARIO_SETTINGS_PATH_DEFAULT,
):

    input_dict = toml_to_input_dict(scenario_path)

    milp_instance = MILP_Algo(**input_dict)

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

    return mh.best_cost, init_solution.total_cost


if __name__ == "__main__":
    final_cost_mh, final_cost_greedy = main(reduced=False)
    print(f"Final cost of the operations: €{final_cost_mh}")

    print(
        f"Improvement from greedy to meta heuristic: €{((final_cost_greedy - final_cost_mh)/final_cost_greedy)*100:.2f}%"
    )
    print("A positive value means we got cheaper, so that is a good thing!")
