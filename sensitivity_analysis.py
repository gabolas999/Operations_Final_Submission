from main import main
from scenarios import SCENARIO_II, SCENARIO_III

from pathlib import Path
import numpy as np
import json
import copy

from pie_chart import generate_sensitivity_pie_charts


VARIABLE_CHANGE_RESULT_SUMMARY_MAP = {
    "gamma": {
        "-60": None,
        "-40": None,
        "-20": None,
        "20": None,
        "40": None,
        "60": None,
    },
    "h_t": {
        "-60": None,
        "-40": None,
        "-20": None,
        "20": None,
        "40": None,
        "60": None,
    },
    "handling_time": {
        "-60": None,
        "-40": None,
        "-20": None,
        "20": None,
        "40": None,
        "60": None,
    },
}


def run_sensitivity_analysis(
    base_scenario=SCENARIO_II,
    output_path=Path(
        "./Storage/theo_results/sensitivity_analysis/sensitivity_analysis_results.json"
    ),
):

    _, _, result_dict, _ = main(
        input_scenario_dict=base_scenario,
        scenario_name=f"Base_Scenario",
        max_iters=10000,
    )

    result_summary = result_dict["summary"]

    result_summary["barge_share_%"] = (
        result_summary["containers_on_barges"] / result_summary["total_containers"]
    ) * 100
    result_summary["truck_share_%"] = (
        result_summary["containers_trucked"] / result_summary["total_containers"]
    ) * 100

    baseline_result_dict = result_dict
    baseline_barge_share = float(np.round(result_summary["barge_share_%"], 2))
    baseline_truck_share = float(np.round(result_summary["truck_share_%"], 2))
    baseline_total_cost = float(np.round(result_summary["final_cost"], 2))

    for variable in VARIABLE_CHANGE_RESULT_SUMMARY_MAP.keys():

        for change_percentage_str in VARIABLE_CHANGE_RESULT_SUMMARY_MAP[
            variable
        ].keys():
            change_percentage = int(change_percentage_str)
            modified_scenario = {}
            modified_scenario = copy.deepcopy(base_scenario)

            if variable == "gamma":
                assert "gamma" in modified_scenario.keys()
                modified_var_value = modified_scenario["gamma"] * (
                    1 + change_percentage / 100
                )
                modified_scenario["gamma"] = modified_var_value
            elif variable == "h_t":
                assert "h_t_20" in modified_scenario.keys()
                assert "h_t_40" in modified_scenario.keys()
                modified_var_value_20 = modified_scenario["h_t_20"] * (
                    1 + change_percentage / 100
                )
                modified_var_value_40 = modified_scenario["h_t_40"] * (
                    1 + change_percentage / 100
                )
                modified_scenario["h_t_20"] = modified_var_value_20
                modified_scenario["h_t_40"] = modified_var_value_40
            elif variable == "handling_time":
                assert "handling_time" in modified_scenario.keys()
                modified_var_value = modified_scenario["handling_time"] * (
                    1 + change_percentage / 100
                )
                modified_scenario["handling_time"] = modified_var_value

            _, _, result_dict, _ = main(
                input_scenario_dict=modified_scenario,
                scenario_name=f"Sensitivity Analysis - {variable} {change_percentage_str}%",
                max_iters=10000,
            )

            result_summary = result_dict["summary"]

            result_summary["modified_value"] = (
                modified_var_value
                if variable != "h_t"
                else (modified_var_value_20, modified_var_value_40)
            )

            result_summary["barge_share_%"] = float(
                np.round(
                    (
                        (
                            result_summary["containers_on_barges"]
                            / result_summary["total_containers"]
                        )
                        * 100
                    ),
                    2,
                )
            )
            result_summary["truck_share_%"] = float(
                np.round(
                    (
                        (
                            result_summary["containers_trucked"]
                            / result_summary["total_containers"]
                        )
                        * 100
                    ),
                    2,
                )
            )

            # pp = percentage points
            result_summary["barge_share_change_pp"] = float(
                np.round(
                    result_summary["barge_share_%"] - baseline_barge_share,
                    2,
                )
            )
            result_summary["truck_share_change_pp"] = float(
                np.round(
                    result_summary["truck_share_%"] - baseline_truck_share,
                    2,
                )
            )
            # percentage change (+ means increase, - means decrease)
            result_summary["total_cost_change_%"] = float(
                np.round(
                    (
                        (result_summary["final_cost"] - baseline_total_cost)
                        / baseline_total_cost
                        * 100
                    ),
                    2,
                )
            )

            # print(result_summary)

            VARIABLE_CHANGE_RESULT_SUMMARY_MAP[variable][change_percentage_str] = (
                result_dict["summary"]
            )

            print(
                f"Sensitivity Analysis Scenario for {variable} with {change_percentage_str}% change, completed."
            )

    # Save results to a JSON file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(VARIABLE_CHANGE_RESULT_SUMMARY_MAP, f, indent=4)

    print("Sensitivity Analysis Results:")
    print(VARIABLE_CHANGE_RESULT_SUMMARY_MAP)

    return VARIABLE_CHANGE_RESULT_SUMMARY_MAP


def analyse_sensitivity_analysis_results(
    results_json_path=Path(
        "./Storage/theo_results/sensitivity_analysis/sensitivity_analysis_results.json"
    ),
):
    with results_json_path.open("r") as f:
        results_data = json.load(f)

    generate_sensitivity_pie_charts(
        results=results_data,
        n_rows=6,
        n_cols=3,
        output_path="./Storage/theo_results/sensitivity_analysis/sensitivity_pies.png",
        dpi=1200,
    )


if __name__ == "__main__":
    final_variable_change_result_summary_map = run_sensitivity_analysis(
        base_scenario=SCENARIO_II
    )

    analyse_sensitivity_analysis_results()
