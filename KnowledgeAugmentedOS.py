"""
Knowledge-Augmented CausalOS

This module wraps CausalOS v4 with web knowledge integration,
enabling fact-checking and hallucination prevention through
real-time web search.
"""

import torch
from typing import Dict, List, Tuple
from WebKnowledgeRetriever import WebKnowledgeRetriever


class KnowledgeAugmentedCausalOS:
    """
    CausalOS wrapper with web knowledge integration.
    
    Features:
    - Fact-checking during counterfactual reasoning
    - Automatic S-matrix updates with verified facts
    - Real-time hallucination detection
    """
    
    def __init__(self, base_os, retriever: WebKnowledgeRetriever = None):
        """
        Initialize knowledge-augmented CausalOS.
        
        Args:
            base_os: UnifiedCausalOSV4 instance
            retriever: WebKnowledgeRetriever instance (optional, will create if None)
        """
        self.osys = base_os
        
        if retriever is None:
            # Create retriever with LLM from base OS
            retriever = WebKnowledgeRetriever(
                llm_model=base_os.model,
                tokenizer=base_os.tokenizer,
                cache_size=100
            )
        
        self.retriever = retriever
        self.fact_corrections = {}  # Track corrections made
        self.verified_facts = []  # Track verified facts
    
    def solve_counterfactual_with_facts(
        self,
        factual: str,
        counterfactual: str,
        options: Dict[str, str],
        verify_facts: bool = True
    ) -> Dict:
        """
        Solve counterfactual with optional fact-checking.
        
        Args:
            factual: Factual scenario
            counterfactual: Counterfactual scenario
            options: Answer options
            verify_facts: Whether to verify facts before reasoning
            
        Returns:
            {
                "selected_option": "A"/"B"/"C",
                "reasoning": "...",
                "verified_facts": [...],
                "fact_corrections": {...}
            }
        """
        
        print(f"\n[KnowledgeOS] Solving counterfactual with fact-checking")
        
        verified_facts = []
        corrections = {}
        
        if verify_facts and self.retriever.ddg_available:
            # Extract and verify claims from factual scenario
            print(f"[KnowledgeOS] Extracting factual claims...")
            claims = self.retriever.extract_factual_claims(factual)
            
            if claims:
                print(f"[KnowledgeOS] Found {len(claims)} claims to verify")
                
                for claim in claims:
                    print(f"[KnowledgeOS] Verifying: {claim}")
                    
                    is_verified, evidence, confidence = self.retriever.verify_fact(
                        claim, 
                        use_llm=True
                    )
                    
                    if is_verified and confidence > 0.6:
                        print(f"[KnowledgeOS] ✓ Verified (confidence: {confidence:.2f})")
                        verified_facts.append({
                            "claim": claim,
                            "confidence": confidence,
                            "evidence": evidence
                        })
                        
                        # Anchor verified fact in S-matrix
                        try:
                            self.osys.anchor_fact(claim, rigidity=150.0)
                            print(f"[KnowledgeOS] Anchored in S-matrix")
                        except Exception as e:
                            print(f"[KnowledgeOS] Warning: Could not anchor fact: {e}")
                    
                    elif not is_verified and confidence > 0.6:
                        print(f"[KnowledgeOS] ✗ Not verified (confidence: {confidence:.2f})")
                        corrections[claim] = {
                            "verified": False,
                            "confidence": confidence,
                            "reason": evidence
                        }
            else:
                print(f"[KnowledgeOS] No specific factual claims detected")
        
        # Run premise-aware counterfactual reasoning
        print(f"[KnowledgeOS] Running premise-aware reasoning...")
        
        result = self.osys.solve_counterfactual(
            factual=factual,
            cf=counterfactual,
            options=options
        )
        
        # Return extended result
        return {
            "selected_option": result,
            "reasoning": f"Used {len(verified_facts)} verified facts from web sources",
            "verified_facts": verified_facts,
            "fact_corrections": corrections
        }
    
    def generate_with_fact_checking(
        self,
        prompt: str,
        max_tokens: int = 100,
        check_interval: int = 10
    ) -> str:
        """
        Generate text with periodic fact-checking.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            check_interval: Check facts every N tokens
            
        Returns:
            Generated text
        """
        
        if not self.retriever.ddg_available:
            print("[KnowledgeOS] Web search not available, using standard generation")
            return self._standard_generation(prompt, max_tokens)
        
        print(f"[KnowledgeOS] Generating with fact-checking (interval: {check_interval})")
        
        # Use CausalOS's generation method
        inputs = self.osys.tokenizer(prompt, return_tensors="pt")
        if next(self.osys.model.parameters()).is_cuda:
            inputs = inputs.to("cuda")
        
        generated_ids = inputs.input_ids.clone()
        
        for step in range(max_tokens):
            # Generate next token
            with torch.no_grad():
                outputs = self.osys.model(generated_ids)
                next_token_logits = outputs.logits[0, -1, :]
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            
            # Fact-check at intervals
            if step > 0 and step % check_interval == 0:
                current_text = self.osys.tokenizer.decode(
                    generated_ids[0][inputs.input_ids.shape[-1]:],
                    skip_special_tokens=True
                )
                
                print(f"\n[KnowledgeOS] Fact-checking at step {step}...")
                
                # Extract claims
                claims = self.retriever.extract_factual_claims(current_text)
                
                for claim in claims:
                    is_verified, evidence, conf = self.retriever.verify_fact(
                        claim,
                        use_llm=True
                    )
                    
                    if is_verified and conf > 0.7:
                        # Anchor verified fact
                        try:
                            self.osys.anchor_fact(claim, rigidity=150.0)
                            print(f"[KnowledgeOS] ✓ Verified and anchored: {claim[:50]}...")
                            self.verified_facts.append(claim)
                        except:
                            pass
                    
                    elif not is_verified and conf > 0.7:
                        print(f"[KnowledgeOS] ⚠ Unverified claim: {claim[:50]}...")
                        # Could trigger regeneration here
            
            # Check for EOS token
            if next_token.item() == self.osys.tokenizer.eos_token_id:
                break
        
        result = self.osys.tokenizer.decode(
            generated_ids[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        
        print(f"\n[KnowledgeOS] Generation complete")
        print(f"[KnowledgeOS] Verified {len(self.verified_facts)} facts during generation")
        
        return result
    
    def _standard_generation(self, prompt: str, max_tokens: int) -> str:
        """Fallback to standard generation without fact-checking"""
        
        inputs = self.osys.tokenizer(prompt, return_tensors="pt")
        if next(self.osys.model.parameters()).is_cuda:
            inputs = inputs.to("cuda")
        
        with torch.no_grad():
            outputs = self.osys.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.osys.tokenizer.eos_token_id
            )
        
        return self.osys.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )
    
    def get_fact_statistics(self) -> Dict:
        """Get statistics about fact-checking"""
        
        return {
            "verified_facts_count": len(self.verified_facts),
            "corrections_count": len(self.fact_corrections),
            "cache_size": len(self.retriever.cache)
        }


if __name__ == "__main__":
    print("[KnowledgeOS] Module loaded successfully")
    print("[KnowledgeOS] To use:")
    print("  from CausalOS_v4 import UnifiedCausalOSV4")
    print("  from KnowledgeAugmentedOS import KnowledgeAugmentedCausalOS")
    print("  osys = UnifiedCausalOSV4(use_web_knowledge=True)")
    print("  result = osys.knowledge_augmented.solve_counterfactual_with_facts(...)")
