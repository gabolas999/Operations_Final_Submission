from Meta_Heuristics_new import repair_route

assigned_containers = [0]

C_dict = {
    0: {
        "Terminal": 1,
        "In_or_Out": 1,  # import
        "Wc": 1,
        "Rc": 0,
        "Oc": 0,
        "Dc": 1000,
    }
}

Qk = 10

T_ij = {
    0: {0: 0, 1: 10},
    1: {0: 10, 1: 0},
}

route = repair_route(
    assigned_containers,
    C_dict,
    Qk,
    T_ij,
)

print("Repaired route:", route)
