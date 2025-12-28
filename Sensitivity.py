import toml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import GRB

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

    def run_experiment(self, param_name, param_values, run_label="Experiment", initial_sol_file=None):
        """
        Runs the MILP model for a range of values for a specific parameter.
        
        Args:
            param_name (str): The exact name of the argument in MILP_Algo.__init__ 
                              (e.g., 'gamma', 'handling_time', 'h_t_40').
            param_values (list): List of values to test.
            run_label (str): Name for the experiment (used for logging).
        
        Returns:
            pd.DataFrame: Collected results.
        """
        results = []
        
        # Initialize the "previous solution" with your file for the first run
        # If initial_sol_file is None, it starts from scratch.
        previous_solution = initial_sol_file 
        
        print(f"\nStarting Experiment: {run_label}")
        print(f"Varying '{param_name}' over: {param_values}\n")

        for val in param_values:
            print(f"Running {run_label} | {param_name} = {val} ...")
            
            current_settings = self.base_settings.copy()
            current_settings[param_name] = val
            current_settings['run_name'] = f"Sens_{param_name}_{val}"
            
            try:
                # 1. Initialize
                solver = MILP_Algo(**current_settings)
                
                # 2. Run with Warm Start
                # (Passes the file path on 1st loop, and dict on later loops)
                solver.run(with_plots=False, warm_start_sol=previous_solution)
                
                # 3. Extract Metrics
                metrics = self._extract_metrics(solver, param_name, val)
                results.append(metrics)
                
                # 4. CAPTURE SOLUTION for next iteration
                if metrics["Status"] == "Optimal":
                    # Switch to using in-memory dictionary for speed
                    previous_solution = solver.get_solution_dict()
                else:
                    previous_solution = None 
                
            except Exception as e:
                print(f"  Error running scenario {val}: {e}")
                # Keep using the old file if the current run failed, 
                # or reset to None? Usually resetting is safer.
                previous_solution = None
                
        self.results_df = pd.DataFrame(results)
        print("\n--- Experiment Complete ---")
        return self.results_df

    def _extract_metrics(self, solver, param_name, param_val):
        """
        Internal helper to pull KPIs from the Gurobi model.
        """
        m = solver.model
        
        # Default failure metrics
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

        if m is None or m.status != GRB.OPTIMAL:
            return metrics

        # --- SUCCESSFUL SOLUTION ---
        metrics["Status"] = "Optimal"
        metrics["Total_Cost"] = m.objVal
        
        # Costs Breakdown
        # Truck Cost
        truck_idx = solver.K_t
        truck_cost = sum(solver.H_T[c] * solver.f_ck[c, truck_idx].X for c in solver.C_list)
        metrics["Truck_Cost"] = truck_cost
        metrics["Barge_Cost"] = m.objVal - truck_cost

        # Container Counts
        trucked_count = sum(1 for c in solver.C_list if solver.f_ck[c, truck_idx].X > 0.5)
        metrics["Containers_Trucked"] = trucked_count
        metrics["Containers_Barged"] = metrics["Containers_Total"] - trucked_count

        # Barge Stats
        # A barge is used if it leaves node 0
        used_barges = [k for k in solver.K_b if sum(solver.x_ijk[0, j, k].X for j in solver.N_list if j!=0) > 0.5]
        metrics["Barges_Used"] = len(used_barges)

        # Utilization (TEU carried / Capacity)
        if used_barges:
            total_util = 0
            for k in used_barges:
                # TEU on barge
                teu_on_barge = sum(solver.W_c[c] for c in solver.C_list if solver.f_ck[c, k].X > 0.5)
                cap = solver.Qk[k]
                total_util += (teu_on_barge / cap)
            metrics["Avg_Utilization"] = (total_util / len(used_barges)) * 100
        else:
            metrics["Avg_Utilization"] = 0

        # Topology: Stops (Total visits to sea terminals)
        # Sum of x_ijk where j != 0 (Sea terminal arrival)
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
        """
        Specific plot for Gamma Sensitivity.
        Dual Axis:
        - Left Y: Total Operational Cost (Blue)
        - Right Y: Total Number of Stops (Red)
        - X: Gamma Value
        """
        if self.results_df is None:
            print("No data to plot.")
            return

        df = self.results_df
        param_col = df.columns[0]  # The parameter we varied (e.g., "gamma")
        
        # Setup Figure
        fig, ax1 = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("white")
        
        # Axis 1: Cost
        color1 = 'tab:blue'
        ax1.set_xlabel(f'Parameter: {param_col}', fontsize=12)
        ax1.set_ylabel('Total Cost (€)', color=color1, fontsize=12)
        ax1.plot(df[param_col], df['Total_Cost'], color=color1, marker='o', linewidth=2, label="Total Cost")
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, linestyle=':', alpha=0.6)

        # Axis 2: Stops
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color2 = 'tab:red'
        ax2.set_ylabel('Total Sea Terminal Stops', color=color2, fontsize=12)
        ax2.plot(df[param_col], df['Total_Stops'], color=color2, marker='s', linestyle='--', linewidth=2, label="Stops")
        ax2.tick_params(axis='y', labelcolor=color2)

        # Title and Layout
        plt.title(f"Topology Trade-off: Cost vs. Stops\n(Varying {param_col})", fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        # Save
        import os
        os.makedirs("Storage/Figures/Sensitivity", exist_ok=True)
        path = f"Storage/Figures/Sensitivity/{save_name}.pdf"
        plt.savefig(path)
        print(f"Plot saved to {path}")
        plt.close()

    def plot_modal_shift(self, save_name="Modal_Shift"):
        """
        Specific plot for Cost Sensitivity.
        Stacked Bar Chart:
        - Y: Number of Containers
        - X: Cost Parameter
        - Stack: Trucked (Red) vs Barged (Blue)
        """
        if self.results_df is None: return

        df = self.results_df
        param_col = df.columns[0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Indices for bars
        x_vals = range(len(df))
        width = 0.6
        
        # Plotting
        ax.bar(x_vals, df['Containers_Barged'], width, label='Barged', color='#3498db')
        ax.bar(x_vals, df['Containers_Trucked'], width, bottom=df['Containers_Barged'], label='Trucked', color='#e74c3c')
        
        # Formatting
        ax.set_xticks(x_vals)
        ax.set_xticklabels(df[param_col], rotation=45)
        ax.set_xlabel(f"Parameter: {param_col}")
        ax.set_ylabel("Number of Containers")
        ax.set_title(f"Modal Shift Analysis\n(Varying {param_col})", fontweight='bold')
        ax.legend()
        
        # Add labels on top
        for i, val in enumerate(df['Containers_Trucked']):
            total = df['Containers_Barged'].iloc[i] + val
            pct_truck = (val / total * 100) if total > 0 else 0
            if pct_truck > 0:
                ax.text(i, total + 1, f"{pct_truck:.1f}% Truck", ha='center', fontsize=9)

        plt.tight_layout()
        path = f"Storage/Figures/Sensitivity/{save_name}.pdf"
        plt.savefig(path)
        print(f"Plot saved to {path}")
        plt.close()

    def plot_performance_curve(self, save_name="Performance_Curve"):
        """
        Specific plot for Time/Congestion Sensitivity.
        Shows how system degrades.
        - X: Time Parameter (e.g. Handling Time)
        - Left Y: Barge Utilization % (Blue)
        - Right Y: % of Cargo Trucked (Infeasibility Proxy) (Red)
        """
        if self.results_df is None: return

        df = self.results_df
        param_col = df.columns[0]
        
        # Calculate % Trucked
        df['Pct_Trucked'] = (df['Containers_Trucked'] / df['Containers_Total']) * 100

        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Axis 1: Utilization
        color1 = 'tab:blue'
        ax1.set_xlabel(f'Parameter: {param_col}', fontsize=12)
        ax1.set_ylabel('Avg Barge Utilization (%)', color=color1, fontsize=12)
        ax1.plot(df[param_col], df['Avg_Utilization'], color=color1, marker='o', label="Barge Util.")
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, 105)
        ax1.grid(True)

        # Axis 2: % Trucked
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('% Containers Trucked', color=color2, fontsize=12)
        ax2.plot(df[param_col], df['Pct_Trucked'], color=color2, marker='x', linestyle='--', linewidth=2, label="% Trucked")
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, 105)

        plt.title(f"Performance Degradation Curve\n(Varying {param_col})", fontweight='bold')
        fig.tight_layout()
        
        path = f"Storage/Figures/Sensitivity/{save_name}.pdf"
        plt.savefig(path)
        print(f"Plot saved to {path}")
        plt.close()

if __name__ == "__main__":
    toml_file = "Storage/Settings/settings________2025_12_28_09_00_01.toml"
    
    # Path to your EXISTING solution file
    my_sol_file = "Storage/Solutions/solved________2025_12_28_09_00_01.sol" 
    
    analyzer = SensitivityAnalysis(toml_file)
    
    gammas = [50, 100, 200, 300, 500]
    
    df = analyzer.run_experiment(
        param_name="gamma", 
        param_values=gammas, 
        run_label="Gamma_Analysis",
        initial_sol_file=my_sol_file  # <--- Pass your file here
    )

    print("\nExperiment Results:")
    print(df)
    
    analyzer.plot_topology_tradeoff("Gamma_Analysis_Plot")