import toml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import GRB
import os

# Import your existing class
from MILP import MILP_Algo


class SensitivityAnalysis:
    def __init__(self, toml_path):
        """
        Initialize with a path to a base configuration file.
        """
        print(f"--- Loading Base Settings from {toml_path} ---")
        self.base_settings = toml.load(toml_path)
        self.results_df = None

    def run_experiment(self, param_name, param_values, run_label="Experiment", initial_sol_file=None):
        """
        Runs the MILP model for a range of values for a specific parameter.
        Now also generates Time Window plots for every optimal run.
        """
        results = []
        previous_solution = initial_sol_file

        print(f"\nStarting Experiment: {run_label}")
        print(f"Varying '{param_name}' over: {param_values}\n")

        # Create a dedicated folder for these plots
        plot_dir = f"Storage_orig/Figures/Sensitivity/{param_name}"
        os.makedirs(plot_dir, exist_ok=True)

        for val in param_values:
            print(f"Running {run_label} | {param_name} = {val} ...")

            # Create a fresh copy of settings
            current_settings = self.base_settings.copy()

            # --- SPECIAL LOGIC FOR TRUCK MULTIPLIER ---
            if param_name == "truck_cost_multiplier":
                base_40 = self.base_settings.get('h_t_40', 200)
                base_20 = self.base_settings.get('h_t_20', 140)
                current_settings['h_t_40'] = base_40 * val
                current_settings['h_t_20'] = base_20 * val
            else:
                current_settings[param_name] = val

            # Update run name
            current_settings['run_name'] = f"Sens_{param_name}_{val}"
            current_settings['enable_arc_elimination'] = True

            try:
                # Initialize
                solver = MILP_Algo(**current_settings)

                # Run (Optimization) - Note: with_plots=False to skip default generation
                solver.run(with_plots=False, warm_start_sol=previous_solution)

                # Extract Metrics
                metrics = self._extract_metrics(solver, param_name, val)
                results.append(metrics)

                # --- PLOT GENERATION LOGIC ---
                if metrics["Status"] == "Optimal":
                    previous_solution = solver.get_solution_dict()

                    try:
                        print(f"   -> Generating Time Window Plot...")
                        # 1. Generate the plot (saves to default folder with timestamp)
                        solver.plot_time_windows()

                        # 2. Identify the generated file and move/rename it
                        # MILP_Algo naming convention: time_windows_{run_name}_{time}.pdf
                        src_path = f"Storage_orig/Figures/time_windows_{solver.file_name}.pdf"

                        # Create a clean filename (e.g., time_windows_100.pdf)
                        # Handle tuples for saturation exp: (80,80) -> 80_80
                        safe_val = str(val).replace(" ", "").replace("(", "").replace(")", "").replace(",", "_")
                        dest_path = f"{plot_dir}/time_windows_{safe_val}.pdf"

                        if os.path.exists(src_path):
                            os.replace(src_path, dest_path)
                            print(f"      Saved plot to: {dest_path}")
                        else:
                            print(f"      [!] Warning: Source plot not found at {src_path}")

                    except Exception as plot_err:
                        print(f"      [!] Plot generation failed: {plot_err}")

                else:
                    previous_solution = None

            except Exception as e:
                print(f"  Error running scenario {val}: {e}")
                previous_solution = None

        self.results_df = pd.DataFrame(results)
        print("\n--- Experiment Complete ---")
        return self.results_df

    def _extract_metrics(self, solver, param_name, param_val):
        """
        Internal helper to pull KPIs from the Gurobi model.
        """
        m = solver.model

        metrics = {
            param_name: param_val,
            "Status": "Infeasible/Error",
            "Total_Cost": None,
            "Truck_Cost": None,
            "Barge_Cost": None,
            "Containers_Total": len(solver.C_list),
            "Containers_Trucked": 0,
            "Containers_Barged": 0,
            "Barges_Used": 0,
            "Avg_Utilization": 0,
            "Total_Stops": 0
        }

        if m is None:
            return metrics

        has_solution = False

        if m.status == GRB.OPTIMAL:
            metrics["Status"] = "Optimal"
            has_solution = True
        elif m.status == GRB.TIME_LIMIT:
            if m.SolCount > 0:
                metrics["Status"] = "TimeLimit (Suboptimal)"
                has_solution = True
            else:
                metrics["Status"] = "TimeLimit (No Solution)"
                return metrics
        else:
            return metrics

        if has_solution:
            metrics["Total_Cost"] = m.objVal
            truck_idx = solver.K_t
            truck_cost = sum(solver.H_T[c] * solver.f_ck[c, truck_idx].X for c in solver.C_list)
            metrics["Truck_Cost"] = truck_cost
            metrics["Barge_Cost"] = m.objVal - truck_cost

            trucked_count = sum(1 for c in solver.C_list if solver.f_ck[c, truck_idx].X > 0.5)
            metrics["Containers_Trucked"] = trucked_count
            metrics["Containers_Barged"] = metrics["Containers_Total"] - trucked_count

            used_barges = [k for k in solver.K_b if sum(solver.x_ijk[0, j, k].X for j in solver.N_list if j != 0) > 0.5]
            metrics["Barges_Used"] = len(used_barges)

            if used_barges:
                total_util = 0
                for k in used_barges:
                    teu_on_barge = sum(solver.W_c[c] for c in solver.C_list if solver.f_ck[c, k].X > 0.5)
                    cap = solver.Qk[k]
                    total_util += (teu_on_barge / cap)
                metrics["Avg_Utilization"] = (total_util / len(used_barges)) * 100
            else:
                metrics["Avg_Utilization"] = 0

            total_stops = sum(
                solver.x_ijk[i, j, k].X
                for k in solver.K_b
                for i in solver.N_list
                for j in solver.N_list
                if j != 0 and i != j
            )
            metrics["Total_Stops"] = total_stops

        return metrics

    # -------------------------------------------------------
    # PLOTTING FUNCTIONS
    # -------------------------------------------------------

    def plot_topology_tradeoff(self, save_name="Gamma_Tradeoff"):
        if self.results_df is None:
            print("No data to plot.")
            return

        df = self.results_df
        param_col = df.columns[0]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("white")

        color1 = 'tab:blue'
        ax1.set_xlabel(f'Parameter: {param_col}', fontsize=12)
        ax1.set_ylabel('Total Cost (€)', color=color1, fontsize=12)
        ax1.plot(df[param_col], df['Total_Cost'], color=color1, marker='o', linewidth=2, label="Total Cost")
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Total Sea Terminal Stops', color=color2, fontsize=12)
        ax2.plot(df[param_col], df['Total_Stops'], color=color2, marker='s', linestyle='--', linewidth=2, label="Stops")
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title(f"Topology Trade-off: Cost vs. Stops\n(Varying {param_col})", fontsize=14, fontweight='bold')
        fig.tight_layout()

        import os
        os.makedirs("Storage_orig/Figures/Sensitivity", exist_ok=True)
        path = f"Storage_orig/Figures/Sensitivity/{save_name}.pdf"
        plt.savefig(path)
        print(f"Plot saved to {path}")
        plt.close()

    def plot_modal_shift(self, save_name="Modal_Shift"):
        if self.results_df is None: return

        df = self.results_df
        param_col = df.columns[0]

        fig, ax = plt.subplots(figsize=(10, 6))
        x_vals = range(len(df))
        width = 0.6

        ax.bar(x_vals, df['Containers_Barged'], width, label='Barged', color='#3498db')
        ax.bar(x_vals, df['Containers_Trucked'], width, bottom=df['Containers_Barged'], label='Trucked',
               color='#e74c3c')

        ax.set_xticks(x_vals)
        ax.set_xticklabels(df[param_col], rotation=45)
        ax.set_xlabel(f"Parameter: {param_col}")
        ax.set_ylabel("Number of Containers")
        ax.set_title(f"Modal Shift Analysis\n(Varying {param_col})", fontweight='bold')
        ax.legend()

        for i, val in enumerate(df['Containers_Trucked']):
            total = df['Containers_Barged'].iloc[i] + val
            pct_truck = (val / total * 100) if total > 0 else 0
            if pct_truck > 0:
                ax.text(i, total + 1, f"{pct_truck:.1f}% Truck", ha='center', fontsize=9)

        plt.tight_layout()
        path = f"Storage_orig/Figures/Sensitivity/{save_name}.pdf"
        plt.savefig(path)
        print(f"Plot saved to {path}")
        plt.close()

    def plot_performance_curve(self, save_name="Performance_Curve"):
        if self.results_df is None: return

        df = self.results_df
        param_col = df.columns[0]
        df['Pct_Trucked'] = (df['Containers_Trucked'] / df['Containers_Total']) * 100

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = 'tab:blue'
        ax1.set_xlabel(f'Parameter: {param_col}', fontsize=12)
        ax1.set_ylabel('Avg Barge Utilization (%)', color=color1, fontsize=12)
        ax1.plot(df[param_col], df['Avg_Utilization'], color=color1, marker='o', label="Barge Util.")
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, 105)
        ax1.grid(True)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('% Containers Trucked', color=color2, fontsize=12)
        ax2.plot(df[param_col], df['Pct_Trucked'], color=color2, marker='x', linestyle='--', linewidth=2,
                 label="% Trucked")
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, 105)

        plt.title(f"Performance Degradation Curve\n(Varying {param_col})", fontweight='bold')
        fig.tight_layout()

        path = f"Storage_orig/Figures/Sensitivity/{save_name}.pdf"
        plt.savefig(path)
        print(f"Plot saved to {path}")
        plt.close()


if __name__ == "__main__":
    import os

    # ==========================================
    # CONFIGURATION & SETTINGS
    # ==========================================

    # Path to base settings file
    # Ensure this file exists and contains a valid 'seed'
    toml_file = "Storage_orig/Settings/settings________2026_02_24_22_53_13.toml"

    # Path to EXISTING solution file (set to None if not available)
    initial_sol = "Storage_orig/Solutions/solved________2026_02_24_22_53_13.sol"

    # Experiment Flags (Set to True to run)
    RUN_GAMMA_ANALYSIS = False  # Exp 1: Topology
    RUN_MODAL_SHIFT = True  # Exp 2: Economic
    RUN_CONGESTION = False  # Exp 3: Time
    RUN_SATURATION = False  # Exp 4: Demand

    # Initialize Analyzer
    analyzer = SensitivityAnalysis(toml_file)

    # =========================================================
    # EXPERIMENT 1: GAMMA SENSITIVITY (Topology Trade-off)
    # =========================================================
    if RUN_GAMMA_ANALYSIS:
        print("\n\n>>> RUNNING EXPERIMENT 1: GAMMA (TOPOLOGY) <<<")
        gammas = [0, 50, 100, 200, 500, 750, 1000]

        analyzer.run_experiment(
            param_name="gamma",
            param_values=gammas,
            run_label="Exp1_Gamma",
            initial_sol_file=initial_sol
        )
        analyzer.plot_topology_tradeoff("Exp1_Gamma_Tradeoff")

    # =========================================================
    # EXPERIMENT 2: ECONOMIC SENSITIVITY (Modal Shift)
    # =========================================================
    if RUN_MODAL_SHIFT:
        print("\n\n>>> RUNNING EXPERIMENT 2: MODAL SHIFT (COST MULTIPLIER) <<<")
        multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2, 3]

        analyzer.run_experiment(
            param_name="truck_cost_multiplier",
            param_values=multipliers,
            run_label="Exp2_CostMultiplier",
            initial_sol_file=initial_sol
        )
        analyzer.plot_modal_shift("Exp2_Modal_Shift_Multiplier")

    # =========================================================
    # EXPERIMENT 3: TIME SENSITIVITY (Congestion)
    # =========================================================
    if RUN_CONGESTION:
        print("\n\n>>> RUNNING EXPERIMENT 3: HANDLING TIME (CONGESTION) <<<")
        handling_times = [0.1, 0.2, 0.3, 0.5, 0.75, 1, 2]

        analyzer.run_experiment(
            param_name="handling_time",
            param_values=handling_times,
            run_label="Exp3_Time",
            initial_sol_file=initial_sol
        )
        analyzer.plot_performance_curve("Exp3_Congestion_Curve")

    # =========================================================
    # EXPERIMENT 4: DEMAND SATURATION (Capacity)
    # =========================================================
    if RUN_SATURATION:
        print("\n\n>>> RUNNING EXPERIMENT 4: DEMAND SATURATION <<<")
        demand_levels = [(80, 80), (100, 100), (120, 120), (150, 150), (170, 170)]

        df_demand = analyzer.run_experiment(
            param_name="C_range_reduced",
            param_values=demand_levels,
            run_label="Exp4_Capacity",
            initial_sol_file=None
        )

        if not df_demand.empty:
            df_demand['Container_Count'] = df_demand['C_range_reduced'].apply(lambda x: x[0])
            df_demand.drop(columns=['C_range_reduced'], inplace=True)
            cols = ['Container_Count'] + [c for c in df_demand.columns if c != 'Container_Count']
            analyzer.results_df = df_demand[cols]
            analyzer.plot_modal_shift("Exp4_Fleet_Saturation")

    print("\n\n------------------------------------------------")
    print("All requested sensitivity experiments completed.")
    print("------------------------------------------------")