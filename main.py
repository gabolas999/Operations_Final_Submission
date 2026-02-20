from pathlib import Path
from MILP import MILP_Algo
from Greedy_Algo import GreedyOptimizer
from Meta_Heuristics import MetaHeuristic

import copy
import numpy as np

from scenarios import SCENARIO_II, SCENARIO_III

from helpers import (
    sanitize_for_yaml,
    _edge_loads_along_route,
    yaml_to_compact_barge_table,
    toml_to_input_dict,
    display_final_allocations,
    export_instance_tables,
    print_instance_summary,
)


def main(
    scenario_info_toml_file_path=None,
    input_scenario_dict=None,
    scenario_name=None,
    max_iters=10000,
    tenure_for_sensitivity_analysis=80,
    shake_thresh=100,
):

    if input_scenario_dict is None and scenario_info_toml_file_path is not None:
        input_dict = copy.deepcopy(toml_to_input_dict(scenario_info_toml_file_path))
    else:
        input_dict = copy.deepcopy(input_scenario_dict)

    milp_instance = MILP_Algo(**input_dict)

    csv_path, _ = export_instance_tables(
        C_dict=milp_instance.C_dict,
        K_list=milp_instance.K_list,
        scenario_name=scenario_name,
    )

    print_instance_summary(csv_path=csv_path)

    greedy = GreedyOptimizer(
        scenario_name=scenario_name,
        problem_instance=milp_instance,
    )

    init_solution = greedy.solve_greedy()

    Greedy_result_dict, Greedy_result_yaml_path = display_final_allocations(
        K=greedy.K,
        C=greedy.C,
        f_ck=greedy.f_ck,
        C_dict=greedy.C_dict,
        route_dict=greedy.route_dict,
        Barge_cap=greedy.Barges,
        H_b=greedy.H_b,
        best_cost=greedy.total_cost,
        edge_loads_along_route=_edge_loads_along_route,
        scenario_name=scenario_name,
        sanitize_for_yaml=sanitize_for_yaml,
        Greedy_or_MH="Greedy",
    )

    yaml_to_compact_barge_table(
        yaml_path=Greedy_result_yaml_path,
        tex_path=Path(Greedy_result_yaml_path).parent,
        scenario_name=scenario_name,
        Greedy_or_MH="Greedy",
    )

    # ------------------------------------------------------------------

    mh = MetaHeuristic(
        scenario_name=scenario_name,
        problem_instance=milp_instance,
        init_solution=init_solution,
        get_route=greedy.get_route,
        get_timing=greedy.get_timing,
    )

    mh.tenure_barge_shake_ban = tenure_for_sensitivity_analysis
    mh.shake_threshold = shake_thresh

    MH_best_cost, best_fck, final_route_dict, *_ = mh.local_search(max_iters=max_iters)

    MH_result_dict, MH_result_yaml_path = display_final_allocations(
        K=mh.K,
        C=mh.C,
        f_ck=mh.f_ck,
        C_dict=mh.C_dict,
        route_dict=mh.route_dict,
        Barge_cap=mh.Barge_cap,
        H_b=mh.H_b,
        best_cost=MH_best_cost,
        edge_loads_along_route=_edge_loads_along_route,
        scenario_name=scenario_name,
        sanitize_for_yaml=sanitize_for_yaml,
        Greedy_or_MH="MH",
    )

    yaml_to_compact_barge_table(
        yaml_path=MH_result_yaml_path,
        tex_path=Path(MH_result_yaml_path).parent,
        scenario_name=scenario_name,
        Greedy_or_MH="MH",
    )

    return MH_best_cost, init_solution.total_cost, MH_result_dict, Greedy_result_dict


if __name__ == "__main__":

    for scenario, scenario_name in [
        # (SCENARIO_II, "Scenario_II"),
        (SCENARIO_III, "Scenario_III"),
    ]:
        final_cost_mh, final_cost_greedy, mh_result_dict, greedy_result_dict = main(
            input_scenario_dict=scenario,
            scenario_name=scenario_name,
            max_iters=10000,
        )
        print(f"Final cost of the operations: €{np.round(final_cost_mh, 2)}")

        print(
            f"Improvement from greedy to meta heuristic: €{((final_cost_greedy - final_cost_mh)/final_cost_greedy)*100:.2f}%"
        )
        if final_cost_greedy - final_cost_mh > 0:
            print("A positive improvement means we got cheaper. GOOD \n")
        else:
            print(
                "A negative or no improvement means did not get cheaper or even worse, we got more expensive. BAD \n"
            )
