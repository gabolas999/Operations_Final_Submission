#!/usr/bin/env python3
"""
Demonstration script showing how to use the plot_only() function in MILP.py

This script demonstrates three ways to use plot_only():
1. From Python code with auto-detection
2. From Python code with explicit file path
3. From command line

Note: Requires Gurobi license for model loading, but does NOT run optimization.
"""

def demo_usage():
    """Demonstrate how to use plot_only function"""
    
    print("="*70)
    print("MILP plot_only() Function - Usage Demonstration")
    print("="*70)
    print()
    
    print("The plot_only() function allows you to generate visualizations from")
    print("a previously saved optimization solution WITHOUT running the expensive")
    print("optimization process again.")
    print()
    
    print("-" * 70)
    print("Method 1: From Python with auto-detection")
    print("-" * 70)
    print("""
from MILP import MILP_Algo

# Create MILP instance with same settings as the saved solution
milp = MILP_Algo(reduced=True, seed=0)

# Automatically find and plot the most recent solution
milp.plot_only()
""")
    
    print("-" * 70)
    print("Method 2: From Python with explicit file path")
    print("-" * 70)
    print("""
from MILP import MILP_Algo

# Create MILP instance
milp = MILP_Algo(reduced=True, seed=0)

# Plot from a specific solution file
milp.plot_only("Storage_orig/Solutions/solved_______2024_01_15_10_30_45.sol")
""")
    
    print("-" * 70)
    print("Method 3: From command line (easiest!)")
    print("-" * 70)
    print("""
# Plot from most recent solution (auto-detect)
python MILP.py --plot-only

# Plot from specific solution file
python MILP.py --plot-only Storage_orig/Solutions/solved_______2024_01_15_10_30_45.sol

# Show help
python MILP.py --help
""")
    
    print("-" * 70)
    print("What gets generated:")
    print("-" * 70)
    print("""
The plot_only() function generates:
1. Result tables (printed to console):
   - Summary statistics
   - Node assignments
   - Distance matrix
   - Barge utilization
   - Container assignments
   - Time schedules

2. Visualization plots (saved as PDFs):
   - Barge solution map showing routes
   - Time window diagrams
   - Container allocation visualizations
   
All plots are saved to: Storage_orig/Figures/
""")
    
    print("="*70)
    print()

if __name__ == "__main__":
    demo_usage()
