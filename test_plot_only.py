#!/usr/bin/env python3
"""
Simple test script to verify the plot_only() function exists and is callable.
This test checks the API without running Gurobi optimization.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_plot_only_function_exists():
    """Test that the plot_only method exists in MILP_Algo class"""
    try:
        from MILP import MILP_Algo
        
        # Check if plot_only method exists
        assert hasattr(MILP_Algo, 'plot_only'), "plot_only method not found in MILP_Algo class"
        
        # Check if it's callable
        milp_instance = MILP_Algo.__new__(MILP_Algo)  # Create without __init__
        assert callable(getattr(milp_instance, 'plot_only', None)), "plot_only is not callable"
        
        print("✓ plot_only() method exists and is callable")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except AssertionError as e:
        print(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_plot_only_docstring():
    """Test that plot_only has proper documentation"""
    try:
        from MILP import MILP_Algo
        
        docstring = MILP_Algo.plot_only.__doc__
        assert docstring is not None, "plot_only has no docstring"
        assert "plot" in docstring.lower(), "Docstring doesn't mention 'plot'"
        assert "solution" in docstring.lower(), "Docstring doesn't mention 'solution'"
        
        print("✓ plot_only() has proper documentation")
        print(f"\nDocstring preview:\n{docstring[:200]}...")
        return True
        
    except AssertionError as e:
        print(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_help_command():
    """Test that the --help command works"""
    import subprocess
    
    try:
        result = subprocess.run(
            [sys.executable, "MILP.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0, f"--help command failed with code {result.returncode}"
        assert "--plot-only" in result.stdout, "--plot-only not mentioned in help"
        assert "Usage:" in result.stdout, "Usage section not found in help"
        
        print("✓ --help command works correctly")
        return True
        
    except subprocess.TimeoutExpired:
        print("✗ --help command timed out")
        return False
    except AssertionError as e:
        print(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("Testing plot_only() functionality")
    print("="*60)
    print()
    
    tests = [
        ("Function exists", test_plot_only_function_exists),
        ("Documentation", test_plot_only_docstring),
        ("Help command", test_help_command),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTest: {name}")
        print("-" * 40)
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
