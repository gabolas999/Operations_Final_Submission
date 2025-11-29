
# File to set up multiple runs of the MILP algorithm, with different settings.


from MILP import MILP_Algo

def multi_run_setup():

    """Set up and run multiple instances of the MILP algorithm with different settings."""
    

    print(f"Running MILP for different settings troughout the night.")
    
    try:
        milp_instance = MILP_Algo(file_name="run_2_n", reduced=True, N_range_reduced=(2, 2))
    except:
        print("Error initializing MILP_Algo instance.")
        return None

