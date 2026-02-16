from MILP import MILP_Algo
from scenarios import SCENARIO_III

for seed in range(1000):
    SCENARIO_III["seed"] = seed
    milp = MILP_Algo(**SCENARIO_III)
