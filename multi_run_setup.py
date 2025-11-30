
# File to set up multiple runs of the MILP algorithm, with different settings.


from MILP import MILP_Algo

import traceback



def multi_run_setup():
    print("Starting multi-run MILP batch...")

    runs = [
        {"run_name": "MILP_Run_1",   "reduced": True,  "N_range_reduced": (2, 2), "seed": 100},

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