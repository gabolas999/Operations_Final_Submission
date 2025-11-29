#!/usr/bin/env python3
"""
Main runner for the Container Allocation Optimization Project

This script runs:
1. The original MetaHeuristics algorithm
2. A sensitivity analysis of the parameters

Usage: python main.py
Testing gabo branch
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from Meta_Heuristics import MetaHeuristic
from _not_currently_in_use.sensitivity_analysis import SensitivityAnalyzer
from Greedy_Algo import (
    C_ordered, C_dict, Barges, H_b, H_t_20, H_t_40, 
    T_ij_list, Handling_time
)

def run_original_metaheuristics():
    """Run the original meta-heuristics algorithm"""
    
    print("Starting Original MetaHeuristics Algorithm")
    print("=" * 60)
    
    # Create MetaHeuristic instance
    mh = MetaHeuristic(C_ordered, C_dict, Barges,
                       H_b, H_t_20, H_t_40,
                       T_ij_list, Handling_time)
    
    # Run initial greedy solution
    print("Generating initial greedy solution...")
    mh.initial_solution()
    greedy_cost = mh.evaluate()[0]
    print(f"Greedy cost: €{greedy_cost}")
    
    # Run meta-heuristic optimization
    print("\nRunning meta-heuristic optimization...")
    mh.local_search()
    final_cost = mh.best_cost
    print(f"Meta-heuristic cost: €{final_cost}")
    
    # Calculate improvement
    improvement = ((greedy_cost - final_cost) / greedy_cost) * 100
    print(f"Improvement: {improvement:.2f}%")
    
    # Display final allocations
    print("\nFinal Allocation Results:")
    print("-" * 40)
    mh.display_final_allocations()
    
    return {
        'greedy_cost': greedy_cost,
        'final_cost': final_cost,
        'improvement_pct': improvement,
        'metaheuristic_instance': mh
    }

def run_sensitivity_analysis():
    """Run sensitivity analysis"""
    
    print("\nStarting Sensitivity Analysis")
    print("=" * 60)
    
    # Create analyzer
    analyzer = SensitivityAnalyzer()
    
    # Ask user for analysis type
    print("Choose analysis type:")
    print("1. Quick analysis (15 scenarios, ~5 min)")
    print("2. Comprehensive analysis (50+ scenarios, ~45 min)")
    print("3. Skip sensitivity analysis")
    
    while True:
        try:  
            choice = input("Enter choice (1, 2, or 3): ").strip()
            if choice in ['1', '2', '3']:
                break
            else:
                print("Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\nAnalysis cancelled by user")
            return None
    
    if choice == '3':
        print("Skipping sensitivity analysis")
        return None
    
    try:
        if choice == '1':
            print("Running quick analysis...")
            results = analyzer.run_quick_analysis()
        else:
            print("Running comprehensive analysis...")
            results = analyzer.run_comprehensive_analysis()
        
        # Show summary
        if results:
            print(f"\nAnalysis complete! {len(results)} scenarios tested")
            
            # Basic statistics
            successful_results = [r for r in results if r.get('success', False)]
            if successful_results:
                costs = [r['final_cost'] for r in successful_results]
                improvements = [r['improvement_pct'] for r in successful_results]
                
                print(f"Results Summary:")
                print(f"   Successful scenarios: {len(successful_results)}")
                print(f"   Cost range: €{min(costs):.0f} - €{max(costs):.0f}")
                print(f"   Average improvement: {np.mean(improvements):.1f}%")
                
                # Ask about plotting
                plot_choice = input("\nGenerate plots? (y/n): ").strip().lower()
                if plot_choice == 'y':
                    print("Available variables for plotting:")
                    variables = analyzer.get_available_variables()
                    print(f"   {', '.join(variables)}")
                    
                    print("\nGenerating sample plots...")
                    try:
                        # Create some example plots
                        analyzer.plot_2d_custom('variation_pct', 'final_cost', 
                                              title="Cost vs Parameter Variation")
                        analyzer.plot_3d_custom('final_cost', 'improvement_pct', 'barge_utilization_pct',
                                              title="3D Analysis: Cost vs Improvement vs Utilization")
                        analyzer.show_plots()
                        print("Plots generated successfully!")
                    except Exception as e:
                        print(f"Plotting error: {e}")
        
        return results
        
    except Exception as e:
        print(f"Error during sensitivity analysis: {e}")
        print("This might be due to import issues or missing dependencies")
        return None

def main():
    """Main function to run both analyses"""
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    print("Container Allocation Optimization Project")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Track overall results
    overall_results = {}
    
    # 1. Run original meta-heuristics
    try:
        meta_results = run_original_metaheuristics()
        overall_results['metaheuristics'] = meta_results
    except Exception as e:
        print(f"Error in meta-heuristics: {e}")
        overall_results['metaheuristics'] = None
    
    # Ask if user wants to continue to sensitivity analysis
    print("\n" + "="*60)
    # continue_choice = input("Continue to sensitivity analysis? (y/n): ").strip().lower()
    continue_choice = 'no'          #+#+# Disabled for now, because we don't have that template. 

    if continue_choice == 'y':
        # 2. Run sensitivity analysis  
        try:
            sensitivity_results = run_sensitivity_analysis()
            overall_results['sensitivity'] = sensitivity_results
        except Exception as e:
            print(f"Error in sensitivity analysis: {e}")
            overall_results['sensitivity'] = None
    else:
        print("Skipping sensitivity analysis")
        overall_results['sensitivity'] = None
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    if overall_results.get('metaheuristics'):
        meta = overall_results['metaheuristics']
        print(f"MetaHeuristics completed successfully")
        print(f"   Final cost: €{meta['final_cost']}")
        print(f"   Improvement: {meta['improvement_pct']:.2f}%")
    else:
        print("MetaHeuristics failed")
    
    if overall_results.get('sensitivity'):
        sens_results = overall_results['sensitivity']
        successful = [r for r in sens_results if r.get('success', False)]
        print("Sensitivity analysis completed successfully")
        print(f"   Scenarios tested: {len(sens_results)}")
        print(f"   Successful scenarios: {len(successful)}")
    elif overall_results.get('sensitivity') is None and continue_choice == 'y':
        print("Sensitivity analysis failed")
    else:
        print("Sensitivity analysis skipped")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Analysis complete!")

if __name__ == "__main__":
    # Import numpy for calculations
    import numpy as np
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
        print("Goodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check your environment and try again.")