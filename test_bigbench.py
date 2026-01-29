import os
import torch
from CausalOS_v1 import UnifiedCausalOSV1

# Enable debug for detailed output
os.environ["DEBUG_LLM"] = "1"

def run_test_case(osys, factual, cf, options, expected_key):
    print(f"\n{'='*60}")
    print(f"Testing: {factual} -> {cf}")
    print(f"{'='*60}")
    
    selected = osys.solve_counterfactual(factual, cf, options)
    
    print(f"\nExpected: {expected_key}, Got: {selected}")
    if selected == expected_key:
        print("✅ TEST PASSED")
        return True
    else:
        print("❌ TEST FAILED")
        return False

def main():
    print("Initializing CausalOS...")
    osys = UnifiedCausalOSV1()
    
    tests = [
        {
            "factual": "A woman sees a fire.",
            "cf": "What would have happened if the woman had touched the fire?",
            "options": {
                "A": "She would have not been burned.",
                "B": "Everything would have been fine.",
                "C": "She would have been burned.",
                "D": "She would have seen fire."
            },
            "expected": "C"
        },
        {
            "factual": "A bird lands in a forest.",
            "cf": "What would have happened if a plane had landed in the forest?",
            "options": {
                "A": "Everything would have been fine.",
                "B": "The plane would have crashed.",
                "C": "The plane would have landed safe and sound.",
                "D": "In a forest you will find lots of planes."
            },
            "expected": "B"
        },
        {
            "factual": "A bird lands in a forest.",
            "cf": "What would have happened if a meteor had landed in the forest?",
            "options": {
                "A": "Meteors are so cool.",
                "B": "That is not possible.",
                "C": "A big one would have started a wildfire.",
                "D": "The bird would have liked the meteor."
            },
            "expected": "C"
        }
    ]
    
    passed = 0
    for t in tests:
        if run_test_case(osys, t["factual"], t["cf"], t["options"], t["expected"]):
            passed += 1
            
    print(f"\n{'='*60}")
    print(f"Total Results: {passed}/{len(tests)} Passed")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
