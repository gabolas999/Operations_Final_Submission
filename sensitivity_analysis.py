from main import main
from scenarios import SCENARIO_II, SCENARIO_III

from pathlib import Path
import json


VARIABLE_CHANGE_RESULT_SUMMARY_MAP = {
    "gamma": {
        "0": None,
        "-35": None,
        "-25": None,
        "-15": None,
        "15": None,
        "25": None,
        "35": None,
    },
    "h_t": {
        "0": None,
        "-35": None,
        "-25": None,
        "-15": None,
        "15": None,
        "25": None,
        "35": None,
    },
    "handling_time": {
        "0": None,
        "-35": None,
        "-25": None,
        "-15": None,
        "15": None,
        "25": None,
        "35": None,
    },
}


def run_sensitivity_analysis(
    base_scenario=SCENARIO_III,
    output_path=Path("./Storage/theo_results/sensitivity_analysis_results.json"),
):

    for variable in VARIABLE_CHANGE_RESULT_SUMMARY_MAP.keys():
        baseline_result_dict = None
        baseline_barge_share = None
        baseline_truck_share = None
        baseline_total_cost = None

        for change_percentage_str in VARIABLE_CHANGE_RESULT_SUMMARY_MAP[
            variable
        ].keys():
            change_percentage = int(change_percentage_str)
            modified_scenario = base_scenario.copy()

            if variable == "gamma":
                assert "gamma" in modified_scenario.keys()
                modified_scenario["gamma"] *= 1 + change_percentage / 100
            elif variable == "h_t":
                assert "h_t_20" in modified_scenario.keys()
                assert "h_t_40" in modified_scenario.keys()
                modified_scenario["h_t_20"] *= 1 + change_percentage / 100
                modified_scenario["h_t_40"] *= 1 + change_percentage / 100
            elif variable == "handling_time":
                assert "handling_time" in modified_scenario.keys()
                modified_scenario["handling_time"] *= 1 + change_percentage / 100

            _, _, result_dict = main(
                input_scenario_dict=modified_scenario,
                scenario_name=f"Sensitivity Analysis - {variable} {change_percentage_str}%",
            )

            result_summary = result_dict["summary"]

            result_summary["barge_share_%"] = (
                result_summary["containers_on_barges"]
                / result_summary["total_containers"]
            ) * 100
            result_summary["truck_share_%"] = (
                result_summary["containers_trucked"]
                / result_summary["total_containers"]
            ) * 100

            if change_percentage == 0:
                baseline_result_dict = result_dict
                baseline_barge_share = result_summary["barge_share_%"]
                baseline_truck_share = result_summary["truck_share_%"]
                baseline_total_cost = result_summary["final_cost"]
            else:

                # pp = percentage points
                result_summary["barge_share_change_pp"] = (
                    result_summary["barge_share_%"] - baseline_barge_share
                ) * 100
                result_summary["truck_share_change_pp"] = (
                    result_summary["truck_share_%"] - baseline_truck_share
                ) * 100

                # percentage change (+ means increase, - means decrease)
                result_summary["total_cost_change_%"] = (
                    (result_summary["final_cost"] - baseline_total_cost)
                    / baseline_total_cost
                    * 100
                )

            print(result_summary)

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
    results_json_path=Path("./Storage/sensitivity_analysis_results.json"),
):
    with results_json_path.open("r") as f:
        results_data = json.load(f)

    pass  # Further analysis can be implemented here


if __name__ == "__main__":
    final_variable_change_result_summary_map = run_sensitivity_analysis()
