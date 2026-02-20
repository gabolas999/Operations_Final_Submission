import math

SCENARIO_II = {
    "run_name": "______",
    # Barge capacities in TEU
    "qk": [
        20,  # Barge 0
        20,  # Barge 1
        20,  # Barge 2
        20,  # Barge 3
    ],
    # Barge fixed costs in euros
    "h_b": [
        1100,  # Barge 0
        1700,  # Barge 1
        1800,  # Barge 2
        1900,  # Barge 3
    ],
    "seed": 0,
    "reduced": False,
    # Trucking costs
    "h_t_40": 200,  # 40ft container trucking cost in euros
    "h_t_20": 140,  # 20ft container trucking cost in euros
    # Time parameters
    "handling_time": 1 / 6,  # hours
    # Container / terminal ranges
    "C_range": (65, 75),  # when reduced=False
    "N_range": (5, 5),  # when reduced=False
    "Oc_range": (24, 190),  # opening time in hours
    "Oc_offset_range": (110, 350),
    # Travel parameters
    "travel_time_long_range": (84, 140),  # hours
    "travel_angle": math.pi,
    "travel_time_scale": 21,
    # Probabilities
    "P40_range": (0.2, 0.22),
    "PExport_range": (0.05, 0.75),
    # Reduced instance ranges
    "C_range_reduced": (65, 75),
    "N_range_reduced": (5, 5),
    # MILP parameters
    "gamma": 100,  # penalty per sea terminal visit [euros]
    "big_m": 1000,
}


SCENARIO_III = {
    "run_name": "______",
    # Barge capacities in TEU
    "qk": [
        104,  # Barge 0
        99,  # Barge 1
        81,  # Barge 2
        52,  # Barge 3
        28,  # Barge 4
    ],
    # Barge fixed costs in euros
    "h_b": [
        3700,  # Barge 0
        3600,  # Barge 1
        3400,  # Barge 2
        2800,  # Barge 3
        1800,  # Barge 4
    ],
    "seed": 0,
    "reduced": False,
    # Trucking costs
    "h_t_40": 200,  # 40ft container trucking cost in euros
    "h_t_20": 140,  # 20ft container trucking cost in euros
    # Time parameters
    "handling_time": 1 / 6,  # hours
    # Container / terminal ranges
    "C_range": (100, 600),  # when reduced=False
    "N_range": (10, 20),  # when reduced=False
    "Oc_range": (24, 190),  # opening time in hours
    "Oc_offset_range": (110, 350),
    # Travel parameters
    "travel_time_long_range": (84, 140),  # hours
    "travel_angle": math.pi,
    "travel_time_scale": 21,
    # Probabilities
    "P40_range": (0.2, 0.22),
    "PExport_range": (0.05, 0.75),
    # Reduced instance ranges
    "C_range_reduced": (65, 75),
    "N_range_reduced": (5, 5),
    # MILP parameters
    "gamma": 100,  # penalty per sea terminal visit [euros]
    "big_m": 1000,
}


SCENARIO_IV = {
    "run_name": "______",
    # Barge capacities in TEU
    "qk": [
        104,  # Barge 0
        99,  # Barge 1
        81,  # Barge 2
        52,  # Barge 3
        28,  # Barge 4
    ],
    # Barge fixed costs in euros
    "h_b": [
        3700,  # Barge 0
        3600,  # Barge 1
        3400,  # Barge 2
        2800,  # Barge 3
        1800,  # Barge 4
    ],
    "seed": 25,
    "reduced": False,
    # Trucking costs
    "h_t_40": 200,  # 40ft container trucking cost in euros
    "h_t_20": 140,  # 20ft container trucking cost in euros
    # Time parameters
    "handling_time": 1 / 6,  # hours
    # Container / terminal ranges
    "C_range": (100, 600),  # when reduced=False
    "N_range": (10, 20),  # when reduced=False
    "Dc_range": (24, 196),  # closing time in hours
    "Rc_range": (0, 24),  # release time in hours
    "Oc_offset_range": (-120, -24),  # (max_offset, min_offset)
    # Travel parameters
    "travel_time_long_range": (84, 140),  # hours
    "travel_angle": math.pi,
    "travel_time_scale": 21,
    # Probabilities
    "P40_range": (0.75, 0.9),
    "PExport_range": (0.05, 0.7),
    # Reduced instance ranges
    "C_range_reduced": (65, 75),
    "N_range_reduced": (5, 5),
    # MILP parameters
    "gamma": 100,  # penalty per sea terminal visit [euros]
    "big_m": 1000,
}
