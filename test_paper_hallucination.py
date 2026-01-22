from CausalOS_v0 import UnifiedCausalOS
import torch

def test_paper_hallucination():
    # Initialize CausalOS
    osys = UnifiedCausalOS()
    
    # 1. Register a "factual" paper title in the S-Matrix
    # Using a real paper to simulate a "known fact"
    paper_title = "Attention Is All You Need"
    token_ids = osys.observer.tokenizer.encode(paper_title, add_special_tokens=False)
    osys.s_matrix.register_sequence(token_ids, rigidity=100.0)
    
    print(f"Registered factual sequence for: '{paper_title}'")
    
    # 2. Test generation
    # Prompt that expects the paper title
    prompt = "The title of the 2017 transformer paper is 'Attention Is"
    
    print(f"\nPrompt: {prompt}")
    
    # Generate with causal check
    # We set a low hesitation threshold to ensure S-Matrix is triggered for demonstration
    output = osys.generate_with_causal_check(prompt, max_new_tokens=10, hesitation_threshold=0.1)
    
    print(f"\nGenerated Output: {output}")
    
    if "All You Need" in output:
        print("\nResult: SUCCESS - Hallucination prevented/Correct sequence induced.")
    else:
        print("\nResult: FAILURE - Correct sequence not induced.")

if __name__ == "__main__":
    test_paper_hallucination()
