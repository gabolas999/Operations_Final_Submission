
# File to set up multiple runs of the MILP algorithm, with different settings.


from MILP import MILP_Algo

import traceback



def multi_run_setup():
    print("Starting multi-run MILP batch...")

    runs = [
        {"run_name": "MILP_Run__for_6_3",   "reduced": True,  "N_range_reduced": (6, 6), "seed": 20, "travel_time_long_range": (5, 7)},
        {"run_name": "MILP_Run__for_6_4",   "reduced": True,  "N_range_reduced": (6, 6), "seed": 30, "travel_time_long_range": (5, 7)},
        {"run_name": "MILP_Run__for_6_5",   "reduced": True,  "N_range_reduced": (6, 6), "seed": 40, "travel_time_long_range": (5, 7)},

    ]

    for cfg in runs:
        print("\n--------------------------------------------------")
        print(f"Launching run: {cfg['run_name']}")
        print("--------------------------------------------------")

        try:
            milp = MILP_Algo(**cfg)
            milp.run(with_plots=True)
            print(f"Run completed successfully: {cfg['run_name']}")

        except Exception as e:
            print(f"Error in run {cfg['run_name']}: {e}")
            traceback.print_exc()
            print(f"Continuing to next run...\n")

    print("\nAll runs attempted.")



if __name__ == "__main__":
    multi_run_setup()