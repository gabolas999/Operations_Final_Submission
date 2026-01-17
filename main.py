from pathlib import Path
from MILP import MILP_Algo
from Greedy_Algo import GreedyOptimizer
from Meta_Heuristics import MetaHeuristic

SCENARIO_SETTINGS_PATH_DEFAULT = Path(
    "./Storage/Settings/settings________2025_12_22_18_01_10.toml"
)


from collections import defaultdict
from pathlib import Path
import csv


def export_instance_tables(
    C_dict: dict,
    K_list: list,
    output_dir=Path("./instance_tables"),
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

    output_path_csv = output_dir / "instance_data.csv"
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
    output_path_tex = output_dir / "instance_latex_table.tex"
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
    scenario_path=SCENARIO_SETTINGS_PATH_DEFAULT,
):

    input_dict = toml_to_input_dict(scenario_path)

    milp_instance = MILP_Algo(**input_dict)

    export_instance_tables(
        C_dict=milp_instance.C_dict,
        K_list=milp_instance.K_list,
    )

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
    final_cost_mh, final_cost_greedy = main()
    print(f"Final cost of the operations: €{final_cost_mh}")

    print(
        f"Improvement from greedy to meta heuristic: €{((final_cost_greedy - final_cost_mh)/final_cost_greedy)*100:.2f}%"
    )
    if final_cost_greedy - final_cost_mh > 0:
        print("A positive value means we got cheaper. GOOD")
    else:
        print("A negative value means we got more expensive. BAD")
