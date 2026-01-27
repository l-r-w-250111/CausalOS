from CausalOS_v1 import UnifiedCausalOSV1 as UnifiedCausalOS
import torch

def test_crass():
    # Initialize CausalOS
    osys = UnifiedCausalOS()
    
    # Sample CRASS cases
    # Format: (factual, counterfactual, options, correct_answer)
    test_cases = [
        (
            "A man walks on a street.",
            "What would have happened if a man had walked on a bed?",
            {"A": "He would have been late.", "B": "Nothing special would have happened.", "C": "He would have arrived on time."},
            "B"
        ),
        (
            "A girl eats an apple.",
            "What would have happened if the girl had eaten a stone?",
            {"A": "She would have been happy.", "B": "She would have broken her teeth.", "C": "She would have felt full."},
            "B"
        ),
        (
            "A student understands an idea.",
            "What would have happened if the student had not understood the idea?",
            {"A": "The student would have increased the probability of performing better on the next test.", "B": "The student would have been happy.", "C": "That is not possible.", "D": "The student would have needed to study more."},
            "D"
        ),
        (
            "A plant grows in a planter.",
            "What would have happened if the planter had grown in the plant?",
            {"A": "That is not possible.", "B": "It would have grown more quickly.", "C": "The plant would have suffered.", "D": "The planter would have cultivated the plant."},
            "A"
        )
    ]
    
    # Update mapping for the new nodes if necessary
    # In a real scenario, this would be done dynamically by node_extractor
    
    correct_count = 0
    for factual, cf, options, expected in test_cases:
        print(f"\nFactual: {factual}")
        print(f"Counterfactual: {cf}")
        
        # Pass the options to the solver
        answer = osys.solve_counterfactual(factual, cf, options=options)
        print(f"Expected: {expected}, Got: {answer}")
        
        if answer == expected:
            correct_count += 1
            
    print(f"\nCRASS Accuracy: {correct_count}/{len(test_cases)} ({correct_count/len(test_cases)*100:.2f}%)")

if __name__ == "__main__":
    test_crass()
