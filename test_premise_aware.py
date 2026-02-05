"""
Test script for Premise-Aware Counterfactual Reasoning

This script tests the improved CausalOS v4 with premise-aware reasoning
on the "bed vs street" problem that previously failed.
"""

import sys
import torch
from CausalOS_v4 import UnifiedCausalOSV4

def test_bed_street_problem():
    """
    Test Case: Bed vs Street Problem
    
    Problem: "A man walks on a street. What would have happened if a man had walked on a bed?"
    
    Original issue: System assumed the man was traveling somewhere with time constraints,
    leading to answers about being "late" or "on time" - but the original text never 
    mentioned any destination or time constraint!
    
    Expected behavior: The improved system should recognize that options A and C 
    (about being late/on time) rely on invalid premises, and select option B.
    """
    
    print("=" * 80)
    print("TEST: Bed vs Street Problem")
    print("=" * 80)
    
    # Initialize CausalOS with premise-aware reasoning
    print("\n[Test] Initializing CausalOS v4 with premise-aware reasoning...")
    
    # Use smaller model for testing
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    osys = UnifiedCausalOSV4(model_id=model_id, use_premise_aware=True)
    
    # Test data
    factual = "A man walks on a street"
    counterfactual = "A man walks on a bed"
    options = {
        "A": "He would have been late",
        "B": "Nothing special would have happened",
        "C": "He would have arrived on time"
    }
    
    print(f"\n[Test] Factual scenario: {factual}")
    print(f"[Test] Counterfactual scenario: {counterfactual}")
    print(f"[Test] Options:")
    for key, value in options.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("RUNNING COUNTERFACTUAL REASONING")
    print("=" * 80)
    
    # Run counterfactual reasoning
    try:
        result = osys.solve_counterfactual(
            factual=factual,
            cf=counterfactual,
            options=options
        )
        
        print("\n" + "=" * 80)
        print("RESULT")
        print("=" * 80)
        print(f"\nSelected option: {result}")
        print(f"Option text: {options[result]}")
        
        # Check if result is correct
        if result == "B":
            print("\n✅ CORRECT! The system correctly identified that:")
            print("   - Options A and C assume a destination/time constraint")
            print("   - This premise was NEVER stated in the original text")
            print("   - Option B is the only one that doesn't rely on invalid premises")
        else:
            print(f"\n❌ INCORRECT! Selected {result} instead of B")
            print("   The system may still be relying on invalid premises")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


def test_minimal_example():
    """
    Simpler test to verify the modules are working.
    """
    
    print("\n" + "=" * 80)
    print("MINIMAL TEST: Module Loading")
    print("=" * 80)
    
    try:
        from PremiseAwareCausal import PremiseAwareCausalExtractor, PremiseAwareCounterfactual
        print("✅ PremiseAwareCausal module loaded successfully")
        
        from CausalOS_v4 import UnifiedCausalOSV4
        print("✅ CausalOS_v4 module loaded successfully")
        
        print("\n[Test] Checking if premise-aware reasoning is available...")
        osys = UnifiedCausalOSV4(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            use_premise_aware=True
        )
        
        if hasattr(osys, 'premise_solver') and osys.premise_solver is not None:
            print("✅ Premise-aware solver initialized successfully")
        else:
            print("❌ Premise-aware solver not initialized")
        
        print("\n✅ All modules loaded successfully!")
        
    except Exception as e:
        print(f"❌ Module loading failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              CausalOS v4 - Premise-Aware Reasoning Test Suite                ║
║                                                                              ║
║  This test suite validates the improved counterfactual reasoning system     ║
║  that addresses the problem of invalid premise assumptions.                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run minimal test first
    test_minimal_example()
    
    # Ask user if they want to continue to full test
    print("\n" + "=" * 80)
    user_input = input("\nContinue to full bed-street test? (y/n): ").strip().lower()
    
    if user_input == 'y':
        test_bed_street_problem()
    else:
        print("\nTest suite complete. Run again with 'y' to execute full test.")
