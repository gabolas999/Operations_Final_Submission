import toml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import GRB
import os
import re
import glob
from collections import defaultdict

# Import your existing class
# Assuming your main file is named MILP.py
from MILP import MILP_Algo


class SensitivityAnalysis:
    def __init__(self, toml_path):
        """
        Initialize with a path to a base configuration file.
        """
        print(f"--- Loading Base Settings from {toml_path} ---")
        self.base_settings = toml.load(toml_path)
        self.results_df = None

    # =========================================================================
    # NEW METHOD: PRINT TABLE TO TERMINAL
    # =========================================================================
    def print_metrics_table(self):
        """
        Prints the current results DataFrame to the terminal formatted as a table.
        This allows access to the specific data points of the curves.
        """
        if self.results_df is None or self.results_df.empty:
            print(">> No results available to print.")
            return

        print("\n" + "=" * 100)
        print(" DATA TABLE: Metrics for Current Experiment")
        print("=" * 100)

        # Create a copy for printing to round numbers for better readability
        df_print = self.results_df.copy()

        # Round numeric columns to 2 decimal places
        numeric_cols = df_print.select_dtypes(include=[np.number]).columns
        df_print[numeric_cols] = df_print[numeric_cols].round(2)

        # Print the full table (prevent truncation of rows/cols)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(df_print.to_string(index=False))

        print("=" * 100 + "\n")

    # =========================================================================
    # DIRECT PARSE METHOD WITH INDIVIDUAL PLOT EXTRACTION
    # =========================================================================
    def run_recovery(self, param_name, run_label="Recovery"):
        """
        Scans the solutions folder, finds the LATEST .sol file for each unique
        parameter value. Parses the text directly and generates the corresponding
        MILP plots in dedicated subfolders.
        """
        solutions_path = "Storage_orig/Solutions"
        print(f"\n--- Starting Direct Text Parsing for: {param_name} ---")

        all_files = glob.glob(os.path.join(solutions_path, f"solved_Sens_{param_name}_*.sol"))

        if not all_files:
            print(f"No files found for {param_name} in {solutions_path}")
            self.results_df = None
            return None

        # 1. Group files by parameter value and pick the latest one
        regex = rf"solved_Sens_{param_name}_(?P<val>.+?)_(?P<time>\d{{4}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}}_\d{{2}})\.sol"
        latest_files = {}

        for fpath in all_files:
            match = re.search(regex, os.path.basename(fpath))
            if match:
                val_str = match.group("val")
                mtime = os.path.getmtime(fpath)
                if val_str not in latest_files or mtime > latest_files[val_str][0]:
                    latest_files[val_str] = (mtime, fpath)

        # Helper classes to "Spoof" Gurobi's variable.X and model.objVal syntax
        class DummyVar:
            def __init__(self, val=0.0):
                self.X = float(val)

        class DummyModel:
            def __init__(self, objVal):
                self.status = GRB.OPTIMAL
                self.objVal = float(objVal)

        results = []
        sorted_vals = sorted(latest_files.keys(), key=lambda x: self._try_convert(x))

        # Setup subfolder for individual MILP plots
        plot_subfolder = f"Storage_orig/Figures/Sensitivity/{param_name}"
        os.makedirs(plot_subfolder, exist_ok=True)

        for val_str in sorted_vals:
            _, fpath = latest_files[val_str]
            val = self._try_convert(val_str)

            # Clean string for file naming (e.g., convert (80,80) to 80_80)
            safe_val_str = str(val).replace(" ", "").replace(",", "_").replace("(", "").replace(")", "")

            print(f"  -> Parsing & Plotting {param_name} = {val}...")

            # 2. Prepare Settings
            current_settings = self.base_settings.copy()
            if param_name == "truck_cost_multiplier":
                base_40 = self.base_settings.get('h_t_40', 200)
                base_20 = self.base_settings.get('h_t_20', 140)
                current_settings['h_t_40'] = base_40 * val
                current_settings['h_t_20'] = base_20 * val
            else:
                current_settings[param_name] = val

            # 3. Create MILP_Algo just to generate the list structures
            solver = MILP_Algo(**current_settings)

            # 4. Extract Text Data Directly from .sol File
            obj_val = 0.0
            sol_vars = {}
            with open(fpath, 'r') as f:
                for line in f:
                    if line.startswith('# Objective value ='):
                        try:
                            obj_val = float(line.split('=')[1].strip())
                        except:
                            pass
                    elif line.startswith('#') or not line.strip():
                        continue
                    else:
                        parts = line.split()
                        if len(parts) >= 2:
                            sol_vars[parts[0]] = float(parts[1])

            # 5. Inject the Dummy Model & Variables into the solver object
            solver.model = DummyModel(obj_val)
            solver.f_ck = defaultdict(lambda: DummyVar(0.0))
            solver.x_ijk = defaultdict(lambda: DummyVar(0.0))
            solver.t_jk = defaultdict(lambda: DummyVar(0.0))  # Required for plot_time_windows

            for vname, vval in sol_vars.items():
                if vval > 0.001:
                    # f_ck
                    m_f = re.search(r"f_ck\[(\d+),(\d+)\]", vname)
                    if m_f:
                        solver.f_ck[int(m_f.group(1)), int(m_f.group(2))] = DummyVar(vval)
                        continue
                    # x_ijk
                    m_x = re.search(r"x_ijk\[(\d+),(\d+),(\d+)\]", vname)
                    if m_x:
                        solver.x_ijk[int(m_x.group(1)), int(m_x.group(2)), int(m_x.group(3))] = DummyVar(vval)
                        continue
                    # t_jk
                    m_t = re.search(r"t_jk\[(\d+),(\d+)\]", vname)
                    if m_t:
                        solver.t_jk[int(m_t.group(1)), int(m_t.group(2))] = DummyVar(vval)

            # 6. Extract Metrics
            metrics = self._extract_metrics(solver, param_name, val)
            results.append(metrics)

            # -------------------------------------------------------------
            # 7. GENERATE INDIVIDUAL PLOTS FROM MILP CODE
            # -------------------------------------------------------------
            # We temporarily spoof solver.file_name so we know exactly
            # what the generated PDF will be called, then move it.
            temp_suffix = f"_temp_recovery_{param_name}_{safe_val_str}"
            solver.file_name = temp_suffix

            # A. Plot Time Windows
            try:
                solver.plot_time_windows()
                temp_tw_path = f"Storage_orig/Figures/time_windows{temp_suffix}.pdf"
                final_tw_path = os.path.join(plot_subfolder, f"time_windows_{safe_val_str}.pdf")
                if os.path.exists(temp_tw_path):
                    os.replace(temp_tw_path, final_tw_path)
            except Exception as e:
                print(f"     [!] Could not plot time windows for {val}: {e}")

            # B. Plot Barge Topology Map
            try:
                # We pass the sol_file_path directly so MILP_Algo can parse y_ijk and z_ijk for the stacks
                offsets = {
                    1: (-3, -1),  # Move node 1 up
                    4: (-3, 1),  # Move node 4 down
                    2: (1 + 2, -1),  # Move node 2 left
                    5: (-1 + 2, 1)  # Move node 5 right
                }
                solver.plot_barge_solution_map_report_3(
                    node_offsets=offsets,
                    curvature_multiplier=3,
                    size_scale=2.0,
                    sol_file_path=fpath
                )
                temp_map_path = f"Storage_orig/Figures/solution_map{temp_suffix}.pdf"
                final_map_path = os.path.join(plot_subfolder, f"solution_map_{safe_val_str}.pdf")
                if os.path.exists(temp_map_path):
                    os.replace(temp_map_path, final_map_path)
            except Exception as e:
                print(f"     [!] Could not plot solution map for {val}: {e}")

        self.results_df = pd.DataFrame(results)
        print(f"--- Parsing Complete: {len(results)} scenarios loaded ---\n")
        return self.results_df

    def _try_convert(self, val_str):
        """Helper to convert strings from filenames back to numbers/tuples."""
        try:
            if val_str.startswith('('): return eval(val_str)
            return float(val_str) if '.' in val_str else int(val_str)
        except:
            return val_str

    # =========================================================================
    # EXPERIMENT RUNNER
    # =========================================================================

    def run_experiment(self, param_name, param_values, run_label="Experiment", initial_sol_file=None):
        results = []
        previous_solution = initial_sol_file
        print(f"\nStarting Experiment: {run_label}")
        print(f"Varying '{param_name}' over: {param_values}\n")
        for val in param_values:
            print(f"Running {run_label} | {param_name} = {val} ...")
            current_settings = self.base_settings.copy()
            if param_name == "truck_cost_multiplier":
                base_40 = self.base_settings.get('h_t_40', 200)
                base_20 = self.base_settings.get('h_t_20', 140)
                current_settings['h_t_40'] = base_40 * val
                current_settings['h_t_20'] = base_20 * val
            else:
                current_settings[param_name] = val
            current_settings['run_name'] = f"Sens_{param_name}_{val}"
            current_settings['enable_arc_elimination'] = True
            try:
                solver = MILP_Algo(**current_settings)
                solver.run(with_plots=False, warm_start_sol=previous_solution)
                metrics = self._extract_metrics(solver, param_name, val)
                results.append(metrics)
                if metrics["Status"] == "Optimal":
                    previous_solution = solver.get_solution_dict()
                else:
                    previous_solution = None
            except Exception as e:
                print(f"  Error running scenario {val}: {e}")
                previous_solution = None
        self.results_df = pd.DataFrame(results)
        print("\n--- Experiment Complete ---")
        return self.results_df

    def _extract_metrics(self, solver, param_name, param_val):
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
            "TEU_Trucked": 0,
            "TEU_Barged": 0,
            "Barges_Used": 0,
            "Avg_Utilization": 0,
            "Total_Stops": 0,
            "Avg_Containers_per_Leg": 0,
            "Avg_TEU_per_Leg": 0,
            "Export_Import_Ratio_Containers": 0,
            "Export_Import_Ratio_TEU": 0
        }

        if m is None:
            return metrics

        # Check solution status
        if m.status == GRB.OPTIMAL:
            metrics["Status"] = "Optimal"
        elif m.status == GRB.TIME_LIMIT and hasattr(m, "SolCount") and m.SolCount > 0:
            metrics["Status"] = "TimeLimit (Suboptimal)"
        elif hasattr(m, "status") and m.status == GRB.OPTIMAL:  # For DummyModel
            metrics["Status"] = "Optimal"
        else:
            return metrics

        # -------------------------------------------------
        # COSTS
        # -------------------------------------------------
        metrics["Total_Cost"] = m.objVal
        truck_idx = solver.K_t

        # Calculate Truck Cost
        truck_cost = sum(
            solver.H_T[c] * solver.f_ck[c, truck_idx].X
            for c in solver.C_list if (c, truck_idx) in solver.f_ck
        )
        metrics["Truck_Cost"] = truck_cost
        metrics["Barge_Cost"] = m.objVal - truck_cost

        # -------------------------------------------------
        # CONTAINER & TEU COUNTS
        # -------------------------------------------------
        trucked = [c for c in solver.C_list if solver.f_ck[c, truck_idx].X > 0.5]
        barged = [c for c in solver.C_list if solver.f_ck[c, truck_idx].X <= 0.5]

        metrics["Containers_Trucked"] = len(trucked)
        metrics["Containers_Barged"] = len(barged)

        teu_total = sum(solver.W_c[c] for c in solver.C_list)
        teu_truck = sum(solver.W_c[c] for c in trucked)
        metrics["TEU_Trucked"] = teu_truck
        metrics["TEU_Barged"] = teu_total - teu_truck

        # -------------------------------------------------
        # EXPORT / IMPORT RATIO (CORRECTED)
        # -------------------------------------------------
        # In MILP_Algo: self.E and self.I are lists of container indices.
        export_indices = solver.E
        import_indices = solver.I

        export_cont = len(export_indices)
        import_cont = len(import_indices)

        export_teu = sum(solver.W_c[c] for c in export_indices)
        import_teu = sum(solver.W_c[c] for c in import_indices)

        # Ratio: Containers
        if import_cont > 0:
            metrics["Export_Import_Ratio_Containers"] = export_cont / import_cont
        else:
            metrics["Export_Import_Ratio_Containers"] = float('inf') if export_cont > 0 else 0

        # Ratio: TEU
        if import_teu > 0:
            metrics["Export_Import_Ratio_TEU"] = export_teu / import_teu
        else:
            metrics["Export_Import_Ratio_TEU"] = float('inf') if export_teu > 0 else 0

        # -------------------------------------------------
        # BARGE UTILIZATION & STOPS
        # -------------------------------------------------
        used_barges = [
            k for k in solver.K_b
            if sum(solver.x_ijk[0, j, k].X for j in solver.N_list if (0, j, k) in solver.x_ijk) > 0.5
        ]
        metrics["Barges_Used"] = len(used_barges)

        if used_barges:
            total_util = 0
            for k in used_barges:
                teu_on_barge = sum(solver.W_c[c] for c in barged if solver.f_ck[c, k].X > 0.5)
                total_util += (teu_on_barge / solver.Qk[k])
            metrics["Avg_Utilization"] = (total_util / len(used_barges)) * 100

        total_stops = sum(
            solver.x_ijk[i, j, k].X
            for (i, j, k) in solver.x_ijk
            if j != 0 and i != j
        )
        metrics["Total_Stops"] = int(total_stops)

        if total_stops > 0:
            metrics["Avg_Containers_per_Leg"] = metrics["Containers_Barged"] / total_stops
            metrics["Avg_TEU_per_Leg"] = metrics["TEU_Barged"] / total_stops

        return metrics

    # =========================================================================
    # PLOTTING
    # =========================================================================

    def plot_topology_tradeoff(self, save_name="Gamma_Tradeoff"):
        if self.results_df is None: return
        df = self.results_df
        param_col = df.columns[0]
        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("white")
        color1 = 'tab:blue'
        ax1.set_xlabel('$\gamma (€)$', fontsize=12)
        ax1.set_ylabel('Total Cost (€)', color=color1, fontsize=12)
        ax1.plot(df[param_col], df['Total_Cost'], color=color1, marker='o', linewidth=2, label="Total Cost")
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Total Sea Terminal Stops', color=color2, fontsize=12)
        ax2.plot(df[param_col], df['Total_Stops'], color=color2, marker='s', linestyle='--', linewidth=2, label="Stops")
        ax2.tick_params(axis='y', labelcolor=color2)
        # plt.title(f"Topology Trade-off: Cost vs. Stops\n(Varying {param_col})", fontsize=14, fontweight='bold')
        fig.tight_layout()
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
        ax.set_xlabel('Truck Cost Multiplier', fontsize=12)
        ax.set_ylabel("Number of Containers")
        # ax.set_title(f"Modal Shift Analysis\n(Varying {param_col})", fontweight='bold')
        # ax.legend()
        for i, val in enumerate(df['Containers_Trucked']):
            total = df['Containers_Barged'].iloc[i] + val
            pct_truck = (val / total * 100) if total > 0 else 0
            if pct_truck > 0: ax.text(i, total + 1, f"{pct_truck:.1f}% Truck", ha='center', fontsize=9)
        plt.tight_layout()
        path = f"Storage_orig/Figures/Sensitivity/{save_name}.pdf"
        plt.savefig(path)
        print(f"Plot saved to {path}")
        plt.close()

    def plot_performance_curve(self, save_name="Avg_Containers_per_Leg"):
        if self.results_df is None:
            return

        df = self.results_df
        param_col = df.columns[0]  # Assuming the first column is the varied parameter

        fig, ax = plt.subplots(figsize=(10, 6))
        color = 'tab:blue'
        ax.set_xlabel('Handling Time (h)', fontsize=12)
        ax.set_ylabel('Average Containers per Barge Leg', color=color, fontsize=12)
        ax.plot(df[param_col], df['Avg_Containers_per_Leg'], color=color, marker='o', linewidth=2,
                label="Avg Containers/Leg")
        ax.tick_params(axis='y', labelcolor=color)
        ax.grid(True)
        fig.tight_layout()

        path = f"Storage_orig/Figures/Sensitivity/{save_name}.pdf"
        plt.savefig(path)
        print(f"Plot saved to {path}")
        plt.close()


if __name__ == "__main__":
    # ==========================================
    # ONLY RUNNING PARSE RECOVERY & SUB-PLOTTING
    # ==========================================
    toml_file = "Storage_orig/Settings/settings________2026_02_24_22_53_13.toml"
    analyzer = SensitivityAnalysis(toml_file)

    print("\n\n>>> RECOVERING EXPERIMENT 1: GAMMA (TOPOLOGY) <<<")
    if analyzer.run_recovery("gamma") is not None:
        analyzer.print_metrics_table()  # <--- PRINT TABLE
        analyzer.plot_topology_tradeoff("Exp1_Gamma_Tradeoff_Recovered")

    print("\n\n>>> RECOVERING EXPERIMENT 2: MODAL SHIFT (COST MULTIPLIER) <<<")
    if analyzer.run_recovery("truck_cost_multiplier") is not None:
        analyzer.print_metrics_table()  # <--- PRINT TABLE
        analyzer.plot_modal_shift("Exp2_Modal_Shift_Multiplier_Recovered")

    print("\n\n>>> RECOVERING EXPERIMENT 3: HANDLING TIME (CONGESTION) <<<")
    if analyzer.run_recovery("handling_time") is not None:
        analyzer.print_metrics_table()  # <--- PRINT TABLE
        analyzer.plot_performance_curve("Exp3_Congestion_Curve_Recovered")

    print("\n\n>>> RECOVERING EXPERIMENT 4: DEMAND SATURATION <<<")
    if analyzer.run_recovery("C_range_reduced") is not None:
        # Re-apply the tuple-to-int conversion for the summary X-axis mapping
        df_demand = analyzer.results_df
        df_demand['Container_Count'] = df_demand['C_range_reduced'].apply(lambda x: x[0])
        df_demand.drop(columns=['C_range_reduced'], inplace=True)
        cols = ['Container_Count'] + [c for c in df_demand.columns if c != 'Container_Count']
        analyzer.results_df = df_demand[cols]

        analyzer.print_metrics_table()  # <--- PRINT TABLE (Updated with Container_Count)
        analyzer.plot_modal_shift("Exp4_Fleet_Saturation_Recovered")

    print("\n\n------------------------------------------------")
    print("Log Parsing and Individual Sub-Plot Generation Completed!")
    print("Check Storage_orig/Figures/Sensitivity/ for results.")
    print("------------------------------------------------")