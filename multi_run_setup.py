
# File to set up multiple runs of the MILP algorithm, with different settings.


from MILP import MILP_Algo

import traceback



def multi_run_setup():
    print("Starting multi-run MILP batch...")

    runs = [
        # {"run_name": "MILP_Run_1",   "reduced": True,  "N_range_reduced": (2, 2), "seed": 100},
        # {"run_name": "MILP_Run_2",   "reduced": True,  "N_range_reduced": (2, 2), "seed": 100},
        # {"run_name": "MILP_Run_3",   "reduced": True,  "N_range_reduced": (3, 3), "seed": 100},
        # {"run_name": "MILP_Run_4",   "reduced": True,  "N_range_reduced": (4, 4), "seed": 100},
        # {"run_name": "MILP_Run_5",   "reduced": True,  "N_range_reduced": (5, 5), "seed": 100},
        # {"run_name": "MILP_Run_5_1", "reduced": True,  "N_range_reduced": (5, 5), "seed": 101},
        {"run_name": "MILP_Run_5_2", "reduced": True,  "N_range_reduced": (5, 5), "seed": 0},
        {"run_name": "MILP_Run_5_3", "reduced": True,  "N_range_reduced": (5, 5), "seed": 1},
        {"run_name": "MILP_Run_6",   "reduced": True,  "N_range_reduced": (6, 6), "seed": 100},
        {"run_name": "MILP_Run_6_1",   "reduced": True,  "N_range_reduced": (6, 6), "seed": 101},
        {"run_name": "MILP_Run_7",   "reduced": True,  "N_range_reduced": (7, 7), "seed": 100},
        {"run_name": "MILP_Run_8",   "reduced": True,  "N_range_reduced": (8, 8), "seed": 100},

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