import matplotlib.pyplot as plt
from typing import List, Dict


def plot_sensitivity_pie_grid(
    dataset: List[Dict],
    n_rows: int = 7,
    n_cols: int = 3,
    output_path: str = "./Storage/sensitivity_pies.png",
    dpi: int = 300,
):
    """
    Plot a grid of pie charts for sensitivity analysis.

    Parameters
    ----------
    dataset : list of dict
        Each dict must contain:
            - barge_pct
            - truck_pct
            - total_cost
            - delta_barge
            - delta_truck
            - variable
            - variation
    n_rows, n_cols : int
        Grid size (default 7x3 = 21)
    output_path : str
        Path to save PNG
    dpi : int
        Resolution for LaTeX / Overleaf
    """

    # A4 size in inches (portrait)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(8.27, 11.69), constrained_layout=True
    )

    axes = axes.flatten()

    for ax, data in zip(axes, dataset):
        values = [data["barge_pct"], data["truck_pct"]]

        ax.pie(values, autopct="%1.0f%%", startangle=90, textprops={"fontsize": 7})

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Subplot title (compact but informative)
        title = (
            f'{data["variable"]} ({data["variation"]:+.0f}%)\n'
            f'Cost: €{data["total_cost"]:,}\n'
            f'ΔBarge: {data["delta_barge"]:+.1f}%, '
            f'ΔTruck: {data["delta_truck"]:+.1f}%'
        )
        ax.set_title(title, fontsize=7)

    # Hide unused axes if dataset < grid size
    for ax in axes[len(dataset) :]:
        ax.axis("off")

    # Shared legend
    fig.legend(
        ["Barge allocation", "Truck allocation"],
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=False,
    )

    # Save figure
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def parse_sensitivity_results(results: dict):
    """
    Convert nested sensitivity-analysis results into
    flat dictionaries for pie-grid plotting.

    Parameters
    ----------
    results : dict
        {
            variable_name: {
                variation_label: summary_dict,
                ...
            },
            ...
        }

    Returns
    -------
    list of dict
        Ready for plot_sensitivity_pie_grid()
    """

    dataset = []

    for variable, variations in results.items():
        for variation_label, summary in variations.items():

            # Skip baseline entries
            if "barge_share_change_pp" not in summary:
                continue

            entry = {
                "barge_pct": summary["barge_share_%"],
                "truck_pct": summary["truck_share_%"],
                "total_cost": summary["final_cost"],
                "delta_barge": summary["barge_share_change_pp"],
                "delta_truck": summary["truck_share_change_pp"],
                "variable": variable,
                "variation": _parse_variation_label(variation_label),
            }

            dataset.append(entry)

    return dataset


def _parse_variation_label(label: str) -> float:
    return float(label)


def generate_sensitivity_pie_charts(
    results: dict,
    n_rows: int = 7,
    n_cols: int = 3,
    output_path: str = "./Storage/sensitivity_pies.png",
    dpi: int = 600,
):
    """
    Generate and save sensitivity analysis pie charts grid.

    Parameters
    ----------
    results : dict
        Nested sensitivity analysis results.
    n_rows, n_cols : int
        Grid size (default 7x3 = 21)
    output_path : str
        Path to save PNG
    dpi : int
        Resolution for LaTeX / Overleaf
    """

    dataset = parse_sensitivity_results(results)

    plot_sensitivity_pie_grid(
        dataset=dataset,
        n_rows=n_rows,
        n_cols=n_cols,
        output_path=output_path,
        dpi=dpi,
    )


if __name__ == "__main__":
    import json
    from pathlib import Path

    # Load results
    results_path = Path("./Storage/sensitivity_analysis_results.json")
    with open(results_path, "r") as f:
        results = json.load(f)

    # Generate pie charts
    generate_sensitivity_pie_charts(
        results=results,
        n_rows=7,
        n_cols=3,
        output_path="./Storage/sensitivity_pies.png",
        dpi=600,
    )
