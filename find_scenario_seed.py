from MILP import MILP_Algo
from scenarios import SCENARIO_III

for seed in range(10000000000000000000000):
    print(f"Testing seed: {seed}")
    scenario = SCENARIO_III.copy()
    scenario["seed"] = seed
    milp = MILP_Algo(**scenario)

    if seed == 0:
        print(f"Initial scenario parameters: C={milp.C}, N={milp.N}, for seed={seed}")

    if milp.C == 141 and milp.N == 12:
        print(f"Found seed: {seed}")
        break

print(f"Scenario parameters: C={milp.C}, N={milp.N}, for seed={seed}")
