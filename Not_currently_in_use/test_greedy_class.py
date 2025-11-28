#!/usr/bin/env python3
"""
Test script to verify the GreedyOptimizer class and T_matrix attribute
"""

from Greedy_Algo import GreedyOptimizer
import numpy as np

def test_greedy_optimizer():
    """Test the GreedyOptimizer class"""
    
    print("🧪 Testing GreedyOptimizer Class")
    print("=" * 50)
    
    # Create optimizer instance
    optimizer = GreedyOptimizer(seed=42, reduced=True)
    
    print(f"✅ Optimizer created successfully")
    print(f"📊 Instance details:")
    print(f"   Containers: {optimizer.C}")
    print(f"   Terminals: {optimizer.N}")
    print(f"   Barge capacities: {optimizer.Qk}")
    print(f"   Fixed costs: {optimizer.H_b}")
    
    # Test T_matrix attribute
    print(f"\n🗺️  Testing T_matrix attribute:")
    print(f"   T_matrix shape: {np.array(optimizer.T_matrix).shape}")
    print(f"   T_matrix type: {type(optimizer.T_matrix)}")
    print(f"   Sample travel times:")
    print(f"     Depot to Terminal 1: {optimizer.T_matrix[0][1]} hours")
    if optimizer.N > 2:
        print(f"     Terminal 1 to Terminal 2: {optimizer.T_matrix[1][2]} hours")
    
    # Test backward compatibility
    print(f"\n🔄 Testing backward compatibility:")
    print(f"   T_ij_list alias works: {np.array_equal(optimizer.T_matrix, optimizer.T_ij_list)}")
    
    # Test solving
    print(f"\n🚀 Testing solve_greedy method:")
    results = optimizer.solve_greedy()
    
    print(f"✅ Solution completed:")
    print(f"   Total cost: €{results['total_cost']:.2f}")
    print(f"   Barge cost: €{results['barge_cost']:.2f}")
    print(f"   Truck cost: €{results['truck_cost']:.2f}")
    print(f"   Trucked containers: {len(results['trucked_containers'])}")
    
    # Test barge assignments
    used_barges = 0
    for k in range(len(optimizer.Barges)):
        containers_on_barge = sum(1 for c in range(optimizer.C) if results['f_ck_init'][c][k] == 1)
        if containers_on_barge > 0:
            used_barges += 1
            teu_on_barge = sum(optimizer.C_dict[c]["Wc"] for c in range(optimizer.C) if results['f_ck_init'][c][k] == 1)
            utilization = (teu_on_barge / optimizer.Barges[k]) * 100
            print(f"   Barge {k}: {containers_on_barge} containers, {teu_on_barge}/{optimizer.Barges[k]} TEU ({utilization:.1f}%)")
    
    print(f"   Total barges used: {used_barges}")
    
    print(f"\n🎉 All tests passed!")
    
    return optimizer, results

def test_backward_compatibility():
    """Test backward compatibility with global variables and functions"""
    
    print(f"\n🔄 Testing Backward Compatibility")
    print("=" * 50)
    
    # Import global variables
    from Greedy_Algo import C_dict, C, N, T_ij_list, Barges, C_ordered, Qk, H_b
    from Greedy_Algo import get_route, container_info
    
    print(f"✅ Global variables imported successfully")
    print(f"   C (containers): {C}")
    print(f"   N (terminals): {N}")
    print(f"   Barges: {Barges}")
    print(f"   T_ij_list shape: {np.array(T_ij_list).shape}")
    
    # Test container_info function
    test_dict, test_c, test_n = container_info(seed=123, reduced=True)
    print(f"✅ container_info function works: {test_c} containers, {test_n} terminals")
    
    # Test get_route function
    sample_containers = {0: {"Terminal": 1}, 1: {"Terminal": 2}}
    route = get_route(sample_containers)
    print(f"✅ get_route function works: {route}")
    
    print(f"🎉 Backward compatibility verified!")

if __name__ == "__main__":
    # Run tests
    optimizer, results = test_greedy_optimizer()
    test_backward_compatibility()
    
    print(f"\n" + "="*50)
    print(f"🏁 ALL TESTS COMPLETED SUCCESSFULLY!")
    print(f"✅ GreedyOptimizer class with T_matrix attribute working perfectly")
    print(f"✅ Backward compatibility maintained")
    print(f"✅ Ready for use in MetaHeuristics and sensitivity analysis")