# MILP.py Plotting Function

## Overview

The `plot_only()` function in MILP.py allows you to generate visualizations from previously saved optimization solutions **WITHOUT** running the expensive optimization process again.

This is useful when you:
- Want to regenerate plots with different settings
- Need to create visualizations for presentations
- Want to compare multiple solutions visually
- Have a saved solution but lost the original plots

## Quick Start

### Command Line (Easiest)

```bash
# Plot from the most recent solution (auto-detect)
python MILP.py --plot-only

# Plot from a specific solution file
python MILP.py --plot-only Storage_orig/Solutions/solved_______2024_01_15_10_30_45.sol

# Show help
python MILP.py --help
```

### From Python Code

```python
from MILP import MILP_Algo

# Create MILP instance with same settings as the saved solution
milp = MILP_Algo(reduced=True, seed=0)

# Option 1: Auto-detect the most recent solution
milp.plot_only()

# Option 2: Use a specific solution file
milp.plot_only("Storage_orig/Solutions/solved_______2024_01_15_10_30_45.sol")
```

## What Gets Generated

### Console Output
- Summary statistics (total cost, containers, terminals)
- Node assignment table
- Distance/travel time matrix
- Barge utilization details
- Container assignment table
- Time schedules for each barge

### Plot Files (PDFs)
All plots are saved to `Storage_orig/Figures/`:
- `solution_map_<timestamp>.pdf` - Barge routes and container allocations
- `time_windows_<timestamp>.pdf` - Container time windows and service times

## Requirements

- Python 3.8+
- gurobipy (Gurobi optimization solver)
- matplotlib
- numpy
- pandas
- tabulate
- scikit-learn
- toml

**Note:** While Gurobi license is required to load the solution file, the `plot_only()` function does NOT run optimization, so it's much faster than a full solve.

## How It Works

1. **Setup**: Creates the model structure (variables and constraints) without solving
2. **Load**: Reads the saved solution file (.sol format)
3. **Validate**: Checks that a valid solution exists in the file
4. **Visualize**: Generates all plots and reports

## Troubleshooting

### "No solution files found"
Make sure you have `.sol` files in `Storage_orig/Solutions/`. Run a full optimization first with `python MILP.py` to generate a solution.

### "No solution found in the file"
The solution file may be corrupted or incompatible. Make sure the instance parameters (seed, reduced flag) match those used to generate the solution.

### "Solution file not found"
Check that the file path is correct and the file exists.

## Example Workflow

```bash
# Step 1: Run optimization once (may take several minutes)
python MILP.py

# Step 2: Later, regenerate plots without re-optimizing (fast!)
python MILP.py --plot-only

# Step 3: Plot a specific historical solution
python MILP.py --plot-only Storage_orig/Solutions/solved_______2024_01_15_10_30_45.sol
```

## See Also

- `demo_plot_only.py` - Demonstration script with usage examples
- `test_plot_only.py` - Test script to verify the function works
