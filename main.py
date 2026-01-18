from pathlib import Path
from MILP import MILP_Algo
from Greedy_Algo import GreedyOptimizer
from Meta_Heuristics import MetaHeuristic

from collections import defaultdict
from pathlib import Path
import csv
import copy

import numpy as np

from scenarios import SCENARIO_II, SCENARIO_III


from pathlib import Path


def print_instance_summary(csv_path: str | Path):
    """
    Read instance_tables.csv and print a formatted instance summary.
    """

    csv_path = Path(csv_path)

    table_a = {}
    table_b = []

    section = None

    with csv_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line == "[Table A]":
                section = "A"
                continue
            elif line == "[Table B]":
                section = "B"
                continue

            if section == "A":
                key, value = line.split(",", 1)
                table_a[key] = value

            elif section == "B":
                if line.startswith("Node"):
                    continue  # header
                node, imp, exp = line.split(",")
                table_b.append((int(node), int(imp), int(exp)))

    # -----------------------------
    # Pretty print
    # -----------------------------
    print("=" * 60)
    print("INSTANCE SUMMARY")
    print("=" * 60)
    print()

    print("GLOBAL PARAMETERS")
    print("-" * 17)
    print(f"Number of terminals (N)      : {table_a['N']}")
    print(f"Total containers (C)         : {table_a['C_total']}")
    print(f"  - Imports                  : {table_a['C_import']}")
    print(f"  - Exports                  : {table_a['C_export']}")
    print(f"Total TEU                    : {table_a['Total_TEU']}")
    print(
        f"Time window                  : "
        f"[{table_a['TimeWindow_start']}, {table_a['TimeWindow_end']}] h"
    )
    print(f"Available vehicles (K)       : {table_a['K_total']}")
    print(f"  - Barges                   : {table_a['K_barges']}")
    print(f"  - Trucks                   : {table_a['K_trucks']}")
    print()

    print("CONTAINER DISTRIBUTION PER NODE")
    print("-" * 31)

    total_imp = int(table_a["C_import"])
    total_exp = int(table_a["C_export"])

    print(
        f"Node  0 (Dry port)           : " f"Import = {total_imp}, Export = {total_exp}"
    )

    for node, imp, exp in table_b:
        print(
            f"Node {node:2d} (Sea terminal)       : "
            f"Import = {imp:3d}, Export = {exp:3d}"
        )

    print("=" * 60)


def export_instance_tables(
    C_dict: dict,
    K_list: list,
    output_dir=Path("./Storage/theo_results"),
    scenario_name=None,
):
    """
    Export Table A (global parameters) and Table B (container distribution)
    to CSV and LaTeX, matching the paper-style layout.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Aggregate global information
    # -----------------------------
    terminals = set()
    import_count = 0
    export_count = 0
    total_teu = 0
    Oc_all = []
    Dc_all = []

    node_counts = defaultdict(lambda: {"Import": 0, "Export": 0})

    for cdata in C_dict.values():
        term = cdata["Terminal"]
        io = cdata["In_or_Out"]
        teu = cdata["Wc"]

        terminals.add(term)
        total_teu += teu
        Oc_all.append(cdata["Oc"])
        Dc_all.append(cdata["Dc"])

        if io == 1:
            import_count += 1
            node_counts[term]["Import"] += 1
        elif io == 2:
            export_count += 1
            node_counts[term]["Export"] += 1
        else:
            raise ValueError(f"Invalid In_Or_Out value: {io}")

    K_barges = len(K_list[:-1])
    K_trucks = len(K_list[-1:])

    N = len(terminals)
    C_total = len(C_dict)
    K_total = K_barges + K_trucks
    time_start = min(Oc_all)
    time_end = max(Dc_all)

    # -----------------------------
    # Write CSV
    # -----------------------------

    output_path_csv = output_dir / f"instance_data_{scenario_name}.csv"
    with output_path_csv.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["[Table A]"])
        writer.writerow(["N", N])
        writer.writerow(["C_total", C_total])
        writer.writerow(["C_import", import_count])
        writer.writerow(["C_export", export_count])
        writer.writerow(["Total_TEU", total_teu])
        writer.writerow(["TimeWindow_start", time_start])
        writer.writerow(["TimeWindow_end", time_end])
        writer.writerow(["K_total", K_total])
        writer.writerow(["K_barges", K_barges])
        writer.writerow(["K_trucks", K_trucks])
        writer.writerow([])

        writer.writerow(["[Table B]"])
        writer.writerow(["Node", "Import", "Export"])
        for node in sorted(node_counts):
            writer.writerow(
                [node, node_counts[node]["Import"], node_counts[node]["Export"]]
            )

    # -----------------------------
    # Write LaTeX (camera-ready)
    # -----------------------------
    output_path_tex = output_dir / f"instance_latex_table_{scenario_name}.tex"
    with output_path_tex.open("w") as f:
        f.write(
            r"""\begin{table}[htbp]
\centering

\textbf{A. Global instance parameters}

\vspace{0.3em}

\begin{tabular}{l c c c}
\hline
Parameter & Symbol & Value & Units \\
\hline
Number of terminals & $N$ & """
            + f"{N}"
            + r""" & -- \\
Total containers & $C$ & """
            + f"{C_total} ({import_count} imp., {export_count} exp.)"
            + r""" & -- \\
Available vehicles & $K$ & """
            + f"{K_total} ({K_barges} Barges + {K_trucks} Truck)"
            + r""" & -- \\
Total TEU & -- & """
            + f"{total_teu} ({sum(1 for c in C_dict.values() if c['Wc']==1)} imp., {sum(1 for c in C_dict.values() if c['Wc']==2)} exp.)"
            + r""" & TEU \\
Global time window span & -- & [$"""
            + f"{time_start}, {time_end}"
            + r"""$] & h \\
\hline
\end{tabular}

\vspace{0.8em}

\textbf{B. Container distribution per node}

\vspace{0.3em}

\begin{tabular}{c l c c}
\hline
Node & Role & Import & Export \\
\hline
"""
        )

        f.write(f"Node 0 & Dry port & {import_count} & {export_count} \\\\\n")

        for node in sorted(node_counts):
            role = "Dry port" if node == 0 else "Sea terminal"
            f.write(
                f"{node} & {role} & {node_counts[node]['Import']} & {node_counts[node]['Export']} \\\\\n"
            )

        f.write(
            r"""\hline
\end{tabular}

\end{table}
"""
        )

    return output_path_csv, output_path_tex


def toml_to_input_dict(toml_path: str) -> dict:
    import toml

    with open(toml_path, "r") as f:
        input_dict = toml.load(f)

    return input_dict


def main(
    scenario_info_toml_file_path=None,
    input_scenario_dict=None,
    scenario_name=None,
):

    if input_scenario_dict is None and scenario_info_toml_file_path is not None:
        input_dict = copy.deepcopy(toml_to_input_dict(scenario_info_toml_file_path))
    else:
        input_dict = copy.deepcopy(input_scenario_dict)

    milp_instance = MILP_Algo(**input_dict)

    # while milp_instance.C > 115:
    #     print(
    #         f"Current scenario has {milp_instance.C} containers, which is too large for MILP solving in sensitivity analysis."
    #     )
    #     print("Regenerating scenario with a different seed...")

    #     input_dict["seed"] += 1

    #     milp_instance = MILP_Algo(**input_dict)

    # print("MILP instance successfully created.")
    # print("seeding info:", input_dict.get("seed", "N/A"))

    csv_path, _ = export_instance_tables(
        C_dict=milp_instance.C_dict,
        K_list=milp_instance.K_list,
        scenario_name=scenario_name,
    )

    print_instance_summary(csv_path=csv_path)

    greedy = GreedyOptimizer(problem_instance=milp_instance)

    init_solution = greedy.solve_greedy()

    mh = MetaHeuristic(
        problem_instance=milp_instance,
        init_solution=init_solution,
        get_route=greedy.get_route,
        get_timing=greedy.get_timing,
        check_for_cap=greedy.check_for_cap,
        delay_window=greedy.delay_window,
    )

    mh.local_search()

    result_dict, result_yaml_path = mh.display_final_allocations(
        scenario_name=scenario_name
    )

    return mh.best_cost, init_solution.total_cost, result_dict


if __name__ == "__main__":

    for scenario, scenario_name in [
        (SCENARIO_II, "Scenario II"),
        (SCENARIO_III, "Scenario III"),
    ]:
        final_cost_mh, final_cost_greedy, result_dict = main(
            input_scenario_dict=scenario, scenario_name=scenario_name
        )
        print(f"Final cost of the operations: €{np.round(final_cost_mh, 2)}")

        print(
            f"Improvement from greedy to meta heuristic: €{((final_cost_greedy - final_cost_mh)/final_cost_greedy)*100:.2f}%"
        )
        if final_cost_greedy - final_cost_mh > 0:
            print("A positive improvement means we got cheaper. GOOD \n")
        else:
            print("A negative improvement means we got more expensive. BAD \n")
