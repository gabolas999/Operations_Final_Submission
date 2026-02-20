from MILP import MILP_Algo
from scenarios import SCENARIO_IV


def compute_total_TEU(c_dict):
    total_TEU = 0
    for info in c_dict.values():
        total_TEU += info["Wc"]

    return total_TEU


for seed in range(10000000000000000000000):
    print(f"Testing seed: {seed}")
    scenario = SCENARIO_IV.copy()
    scenario["seed"] = seed

    total_barge_capacity = sum(scenario["qk"])
    if seed == 0:
        print(f"Total barge capacity for seed {seed}: {total_barge_capacity} TEU")
    #     pause = input("Press Enter to continue...")

    milp = MILP_Algo(**scenario)

    Total_TEU = compute_total_TEU(milp.C_dict)

    # Define the stopping criterion
    stopping_criterion = (
        Total_TEU >= 1.8 * total_barge_capacity
        and Total_TEU <= 2.2 * total_barge_capacity
        and milp.C <= 200
    )

    stopping_criterion = (
        (milp.C <= 300) and (milp.C >= 280) and (milp.N <= 13) and (milp.N >= 10)
    )

    if stopping_criterion:
        print(f"Found seed: {seed}")
        break

print(
    f"Scenario parameters: C={milp.C}, N={milp.N}, Total_TEU={Total_TEU}, for seed={seed}"
)
