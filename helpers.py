import yaml
from pathlib import Path
from collections import defaultdict
import csv
import numpy as np


def _edge_loads_along_route(
    route,
    L_current,
):

    edge_load_list = []

    # Start at depot: all exports are loaded
    load = sum(cont["Wc"] for cont in L_current.values() if cont["In_or_Out"] == 2)
    edge_load_list.append(load)

    # Visit terminals once in the given route
    for terminal in route:
        if terminal == 0:
            continue

        exports_unloaded = sum(
            cont["Wc"]
            for cont in L_current.values()
            if cont["Terminal"] == terminal and cont["In_or_Out"] == 2
        )
        imports_loaded = sum(
            cont["Wc"]
            for cont in L_current.values()
            if cont["Terminal"] == terminal and cont["In_or_Out"] == 1
        )

        load -= exports_unloaded
        load += imports_loaded
        edge_load_list.append(load)

    return edge_load_list


def _get_L_current_for_barge(barge_idx, f_ck, C, C_dict):
    assigned = [cont for cont in range(C) if f_ck[cont, barge_idx] == 1]
    L_current = {cont: C_dict[cont] for cont in assigned}
    return L_current


def yaml_to_compact_barge_table(
    yaml_path,
    tex_path,
    scenario_name,
    Greedy_or_MH="MH",
    containers_per_row=2,
):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    barges = data["barges"]
    summary = data["summary"]
    trucked = data["trucked_containers"]

    def barge_block(b):
        return [
            ("Cap. (TEU)", b["capacity"]),
            ("Cost (€)", b["fixed_cost"]),
            ("# of Cont.", b["num_containers"]),
            (
                "Peak Util.",
                f'{b["peak_load"]} ({b["utilization_percent"]}\\%)',
            ),
            (
                "Imp / Exp",
                f'{len(b["imports"])} / {len(b["exports"])}',
            ),
        ]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Final barge allocation overview}")
    lines.append(r"\label{tab:barge_allocation}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{%s}" % ("ll" * containers_per_row))
    lines.append(r"\toprule")

    for i in range(0, len(barges), containers_per_row):
        row_barges = barges[i : i + containers_per_row]

        # Header row (Barge X)
        header = []
        for b in row_barges:
            header.extend(
                [rf"\multicolumn{{2}}{{c}}{{\textbf{{Barge {b['barge_id']}}}}}"]
            )
        lines.append(" & ".join(header) + r" \\")
        lines.append(r"\midrule")

        blocks = [barge_block(b) for b in row_barges]

        for row_idx in range(len(blocks[0])):
            row = []
            for block in blocks:
                label, value = block[row_idx]
                row.extend([label, str(value)])
            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Trucker info
    if trucked["container_ids"]:
        ids = ", ".join(str(c) for c in trucked["container_ids"])
        lines.append(
            rf"\vspace{{0.5em}}\\\footnotesize " rf"Trucked container(s): ID {ids}"
        )

    # Final cost
    lines.append(
        rf"\vspace{{0.5em}}\\\footnotesize "
        rf"Total operational cost: €{summary['final_cost']}"
    )

    lines.append(r"\end{table}")

    tex_path = (
        Path(tex_path)
        / f"{scenario_name}_{Greedy_or_MH}_final_barge_allocation_table.tex"
    )

    Path(tex_path).write_text("\n".join(lines))


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


def timing_window_plot(C, K, C_dict, f_ck, MH_or_Greedy, final_route_dict: dict):
    """
    Plot container time windows with actual barge arrival times.

    - One row per container
    - Grouped by barge
    - Green = import, Red = export
    - Square marker = export release time Rc
    - Cross marker = actual barge arrival time at container terminal
    """

    import matplotlib.pyplot as plt
    import math

    # --------------------------------------------------
    # 1) Collect containers per barge (final solution)
    # --------------------------------------------------
    barge_to_containers = {k: [] for k in range(K)}
    trucked = []

    for c in range(C):
        assigned = False
        for k in range(K):
            if f_ck[c, k] == 1:
                barge_to_containers[k].append(c)
                assigned = True
                break
        if not assigned:
            trucked.append(c)

    # --------------------------------------------------
    # 2) Compute global time horizon
    # --------------------------------------------------
    max_D = max(C_dict[c]["Dc"] for c in range(C))
    Tmax = int(math.ceil(max_D / 50.0) * 50)

    # --------------------------------------------------
    # 3) Precompute arrival times per (barge, terminal)
    #    using waiting logic
    # --------------------------------------------------
    arrival_time = {}  # (k, terminal) -> time

    for k, info_dict in final_route_dict.items():

        route = info_dict["route"]
        if not route or len(route) <= 1:
            continue

        containers = barge_to_containers[k]
        if not containers:
            continue

        Lcur = {c: C_dict[c] for c in containers}

        timing = info_dict["timing"]

        if timing is None:
            print(
                f"Warning: could not compute arrival times for barge {k} in final plot"
            )
            print(f"Debug: route: {route}, Lcur: {Lcur}")
            continue  # or mark route as infeasible

        for node, arrival in timing.items():
            arrival_time[(k, node)] = arrival

    # --------------------------------------------------
    # 4) Build plot rows (barge, terminal, container)
    # --------------------------------------------------
    rows = []

    # --- barges in ascending order ---
    for k in sorted(barge_to_containers.keys()):
        containers = barge_to_containers[k]

        # sort by (terminal, container)
        containers_sorted = sorted(containers, key=lambda c: (C_dict[c]["Terminal"], c))

        for c in containers_sorted:
            rows.append((k, C_dict[c]["Terminal"], c))

    # --- trucked containers last (optional) ---
    trucked_sorted = sorted(trucked, key=lambda c: (C_dict[c]["Terminal"], c))

    for c in trucked_sorted:
        rows.append(("Truck", C_dict[c]["Terminal"], c))

    # --------------------------------------------------
    # 5) Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 0.3 * len(rows)))

    yticks = []
    ylabels = []

    for y, (k, terminal, c) in enumerate(rows):
        info = C_dict[c]
        Oc, Dc = info["Oc"], info["Dc"]

        color = "green" if info["In_or_Out"] == 1 else "red"

        # time window bar
        ax.barh(
            y,
            Dc - Oc,
            left=Oc,
            height=0.6,
            color=color,
            alpha=0.6,
            edgecolor="black",
        )

        # export release time
        if info["In_or_Out"] == 2 and info["Rc"] > 0:
            ax.scatter(info["Rc"], y, marker="s", color="black", zorder=3)

        # barge arrival time
        if k != "Truck":
            t_arr = arrival_time[(k, terminal)]
            if t_arr is not None:
                ax.scatter(t_arr, y, marker="x", color="black", zorder=3)

        yticks.append(y)
        ylabels.append(f"B{k} | T{terminal} | C{c}")

    # --------------------------------------------------
    # 6) Final formatting
    # --------------------------------------------------
    ax.set_xlim(0, Tmax)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time [hours]")
    ax.set_title("Container Time Windows and Barge Arrival Times")

    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    file_path = f"./Storage/theo_results/{MH_or_Greedy}_timing_window_plot.png"
    plt.savefig(file_path, dpi=600)
    plt.close()

    print("Saved timing window plot to:", file_path)

    return fig, file_path


def build_final_allocation_report(
    K,
    C,
    f_ck,
    C_dict,
    route_dict,
    Barge_cap,
    H_b,
    best_cost,
    edge_loads_along_route=_edge_loads_along_route,
):
    report = {"summary": {}, "barges": [], "trucked_containers": {}}

    barge_assignments = {k: [] for k in range(K)}
    trucked_containers = []
    total_containers_on_barges = 0

    for cont in range(C):
        assigned = False
        for k in range(K):
            if f_ck[cont, k] == 1:
                barge_assignments[k].append(cont)
                total_containers_on_barges += 1
                assigned = True
                break
        if not assigned:
            trucked_containers.append(cont)

    for k in range(K):
        containers = barge_assignments[k]
        if not containers:
            continue

        Lcur = {cont: C_dict[cont] for cont in containers}
        route = route_dict[k]

        cap = Barge_cap[k]

        edge_loads = edge_loads_along_route(route, Lcur)
        peak = max(edge_loads)

        imports = [cont for cont in containers if C_dict[cont]["In_or_Out"] == 1]
        exports = [cont for cont in containers if C_dict[cont]["In_or_Out"] == 2]

        report["barges"].append(
            {
                "barge_id": k + 1,
                "capacity": cap,
                "fixed_cost": H_b[k],
                "num_containers": len(containers),
                "peak_load": peak,
                "utilization_percent": round(100 * peak / cap, 1),
                "imports": imports,
                "exports": exports,
                "container_ids": containers,
            }
        )

    report["trucked_containers"] = {
        "container_ids": trucked_containers,
        "num_20ft": sum(1 for cont in trucked_containers if C_dict[cont]["Wc"] == 1),
        "num_40ft": sum(1 for cont in trucked_containers if C_dict[cont]["Wc"] == 2),
    }

    report["summary"] = {
        "total_containers": C,
        "containers_on_barges": total_containers_on_barges,
        "containers_trucked": len(trucked_containers),
        "barges_used": len(report["barges"]),
        "final_cost": best_cost,
    }

    return report


def sanitize_for_yaml(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_yaml(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_yaml(v) for v in obj)
    elif isinstance(obj, np.generic):  # catches np.float64, np.int64, etc.
        return obj.item()
    else:
        return obj


def display_final_allocations(
    K,
    C,
    f_ck,
    C_dict,
    route_dict,
    Barge_cap,
    H_b,
    best_cost,
    edge_loads_along_route,
    scenario_name,
    sanitize_for_yaml,
    yaml_dir="./Storage/theo_results",
    Greedy_or_MH="MH",
):
    import yaml

    yaml_path = f"{yaml_dir}/{scenario_name}_{Greedy_or_MH}_final_allocations.yaml"
    report = build_final_allocation_report(
        K,
        C,
        f_ck,
        C_dict,
        route_dict,
        Barge_cap,
        H_b,
        best_cost,
        edge_loads_along_route=edge_loads_along_route,
    )

    # Pretty print (terminal)
    print("\n" + "=" * 80)
    print("FINAL CONTAINER-BARGE ALLOCATIONS")
    print("=" * 80)

    for b in report["barges"]:
        print(
            f"\nBARGE {b['barge_id']} (Capacity: {b['capacity']} TEU, Fixed cost: €{b['fixed_cost']}):"
        )
        print(f"  Assigned containers: {b['num_containers']}")
        print(
            f"  Peak onboard load: {b['peak_load']}/{b['capacity']} "
            f"({b['utilization_percent']}% utilization)"
        )
        print(f"  Import containers: {len(b['imports'])}")
        print(f"  Export containers: {len(b['exports'])}")

    print("\nSUMMARY:")
    for k, v in report["summary"].items():
        print(f"  {k.replace('_',' ').title()}: {v}")

    # Save YAML
    report = sanitize_for_yaml(report)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(report, f, sort_keys=False)

    return report, yaml_path
