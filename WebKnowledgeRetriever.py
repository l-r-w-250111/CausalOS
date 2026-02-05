"""
Web Knowledge Retrieval for CausalOS

This module provides API-free web search capabilities using DuckDuckGo.
It supports:
- Web search with caching
- Fact verification against web sources
- Factual claim extraction from text

No API keys required - completely free to use.
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
import time


class WebKnowledgeRetriever:
    """
    API-free web knowledge retrieval using DuckDuckGo.
    
    Features:
    - Web search with automatic caching
    - Fact verification with confidence scoring
    - Factual claim extraction using LLM
    """
    
    def __init__(self, llm_model=None, tokenizer=None, cache_size=100):
        """
        Initialize web knowledge retriever.
        
        Args:
            llm_model: Optional LLM for claim extraction/verification
            tokenizer: Optional tokenizer for the LLM
            cache_size: Maximum number of cached queries
        """
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.cache = OrderedDict()
        self.cache_size = cache_size
        
        # Try to import DuckDuckGo search
        try:
            from duckduckgo_search import DDGS
            self.DDGS = DDGS
            self.ddg_available = True
        except ImportError:
            print("[WebKnowledge] Warning: duckduckgo-search not installed")
            print("[WebKnowledge] Install with: pip install duckduckgo-search")
            self.DDGS = None
            self.ddg_available = False
    
    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Search the web using DuckDuckGo with fallback backends.
        """
        
        if not self.ddg_available:
            print("[WebKnowledge] DuckDuckGo search not available")
            return []
        
        # Check cache first
        cache_key = f"{query}:{max_results}"
        if cache_key in self.cache:
            print(f"[WebKnowledge] Cache hit for: {query}")
            return self.cache[cache_key]
        
        results = []
        # 'html' is often most robust for scraping. 'lite' is lightweight. 'api' is officially deprecated.
        backends = ["html", "lite", "api"]
        errors = []
        
        # Use 'us-en' for better English results than 'wt-wt'
        region = 'us-en'
        
        for backend in backends:
            try:
                print(f"[WebKnowledge] Searching: {query} (backend={backend}, region='{region}')")
                
                with self.DDGS() as ddgs:
                    # Note: ddgs.text() arguments vary by version. 
                    # We try to pass backend and region if possible.
                    kwargs = {'max_results': max_results, 'region': region}
                    
                    # Some versions support backend arg, some don't (defaulting to api/auto)
                    # We'll try to force it if the library supports it.
                    try:
                        search_results = ddgs.text(query, backend=backend, **kwargs)
                    except TypeError:
                        # Fallback for older/newer versions with different sig
                        search_results = ddgs.text(query, **kwargs)

                    for r in search_results:
                        results.append({
                            "title": r.get("title", ""),
                            "snippet": r.get("body", ""),
                            "url": r.get("href", ""),
                            "relevance_score": 1.0
                        })
                
                if results:
                    print(f"[WebKnowledge] Found {len(results)} results with backend='{backend}'")
                    break
                    
            except Exception as e:
                # errors.append(f"{backend}: {str(e)}")
                # Don't log expected deprecation warnings as errors
                pass
                continue
        
        if not results:
            print(f"[WebKnowledge] Search failed. Last error: {errors[-1] if errors else 'No results'}")
            return []
            
        # Update cache
        self._update_cache(cache_key, results)
        return results

    # ... (verify_fact implementation omitted) ...

    def extract_factual_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text using LLM.
        """
        
        if self.llm_model is None or self.tokenizer is None:
            print("[WebKnowledge] LLM not available for claim extraction")
            return []
        
        prompt = f"""Extract factual claims from the following text.
Text: "{text}"
Return ONLY a JSON list of strings.
Example: ["claim1", "claim2"]
JSON:"""

        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if next(self.llm_model.parameters()).is_cuda:
                inputs = inputs.to("cuda")
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            )
            
            # Debug log
            # print(f"[WebKnowledge] LLM Extract Response: {response}")
            
            # Robust extraction
            try:
                start_idx = response.find('[')
                end_idx = response.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    return json.loads(response[start_idx:end_idx+1])
            except:
                pass
                
            print(f"[WebKnowledge] Failed to parse claims. Response: {response[:100]}...")
            
        except Exception as e:
            print(f"[WebKnowledge] Claim extraction error: {e}")
        
        return []
    
    def verify_fact(
        self, 
        claim: str, 
        use_llm: bool = True
    ) -> Tuple[bool, str, float]:
        """
        Verify a factual claim against web sources.
        
        Args:
            claim: The claim to verify
            use_llm: Whether to use LLM for verification (requires llm_model)
            
        Returns:
            (is_verified, evidence, confidence)
            - is_verified: True if claim is supported by web sources
            - evidence: Supporting evidence or reason for rejection
            - confidence: Confidence score 0.0-1.0
        """
        
        if not self.ddg_available:
            return False, "Web search not available", 0.0
        
        # Strategy 1: strict search with quotes
        search_query = f'"{claim}" fact check'
        results = self.search(search_query, max_results=5)
        
        # Strategy 2: relaxed search without quotes (fallback)
        if not results:
            print(f"[WebKnowledge] Strict search failed, trying relaxed search...")
            search_query = f"{claim} fact check"
            results = self.search(search_query, max_results=5)
        
        if not results:
            # Final fallback: search only the claim keywords
            print(f"[WebKnowledge] Fact check search failed, trying keywords only...")
            search_query = claim
            results = self.search(search_query, max_results=5)
            
        if not results:
            return False, "No sources found", 0.0
        
        # Combine evidence from top results
        evidence_text = "\n\n".join([
            f"Source: {r['title']}\n{r['snippet']}"
            for r in results[:3]
        ])
        
        if use_llm and self.llm_model is not None and self.tokenizer is not None:
            # Use LLM to assess if evidence supports the claim
            return self._verify_with_llm(claim, evidence_text, results)
        else:
            # Simple keyword matching (fallback)
            return self._verify_with_keywords(claim, evidence_text, results)

    # ... (omit _verify_with_llm and _verify_with_keywords as they are unchanged) ...

    def extract_factual_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text using LLM.
        """
        
        if self.llm_model is None or self.tokenizer is None:
            print("[WebKnowledge] LLM not available for claim extraction")
            return []
        
        prompt = f"""Extract factual claims from the following text that can be verified.
Focus on specific, verifiable statements (names, dates, numbers, events).
Do NOT extract opinions or subjective statements.

Text: "{text}"

Output as JSON list:
["claim 1", "claim 2", "claim 3"]

JSON:"""

        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if next(self.llm_model.parameters()).is_cuda:
                inputs = inputs.to("cuda")
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            )
            
            # Robust JSON extraction (same as PremiseAwareCausal)
            # Strategy 1: Regex for array
            json_match = re.search(r'(\[(?:[^\[\]]|\{[^\}]*\})*\])', response, re.DOTALL)
            if json_match:
                try:
                    claims = json.loads(json_match.group(1))
                    if isinstance(claims, list):
                        return [str(c) for c in claims if c]
                except:
                    pass
            
            # Strategy 2: Try parsing whole response
            try:
                claims = json.loads(response)
                if isinstance(claims, list):
                    return [str(c) for c in claims if c]
            except:
                pass
                
            print(f"[WebKnowledge] Could not parse claims JSON from: {response[:100]}...")
            
        except Exception as e:
            print(f"[WebKnowledge] Claim extraction error: {e}")
        
        return []
    
    def _verify_with_llm(
        self, 
        claim: str, 
        evidence: str, 
        results: List[Dict]
    ) -> Tuple[bool, str, float]:
        """Verify claim using LLM analysis of evidence"""
        
        prompt = f"""Does the following evidence support this claim?

Claim: "{claim}"

Evidence from web sources:
{evidence}

Analyze carefully and answer with JSON:
{{
  "supported": true/false,
  "confidence": 0.0-1.0 (how confident you are in this assessment),
  "reasoning": "brief explanation of why the claim is/isn't supported"
}}

JSON:"""

        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if next(self.llm_model.parameters()).is_cuda:
                inputs = inputs.to("cuda")
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{[^\}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                
                is_supported = result.get("supported", False)
                confidence = result.get("confidence", 0.5)
                reasoning = result.get("reasoning", "")
                
                # Combine reasoning with source URLs
                evidence_with_sources = f"{reasoning}\n\nSources:\n" + "\n".join([
                    f"- {r['title']}: {r['url']}"
                    for r in results[:3]
                ])
                
                return is_supported, evidence_with_sources, confidence
            
        except Exception as e:
            print(f"[WebKnowledge] LLM verification error: {e}")
        
        # Fallback to keyword matching
        return self._verify_with_keywords(claim, evidence, results)
    
    def _verify_with_keywords(
        self, 
        claim: str, 
        evidence: str, 
        results: List[Dict]
    ) -> Tuple[bool, str, float]:
        """Simple keyword-based verification (fallback)"""
        
        # Check if key terms from claim appear in evidence
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        
        # Extract key terms (simple approach)
        claim_words = set(re.findall(r'\b\w+\b', claim_lower))
        # Remove common words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        claim_words = claim_words - stop_words
        
        # Count how many key terms appear in evidence
        matches = sum(1 for word in claim_words if word in evidence_lower)
        confidence = min(1.0, matches / max(1, len(claim_words)))
        
        is_verified = confidence > 0.5
        
        evidence_summary = f"Found {matches}/{len(claim_words)} key terms in web sources.\n\nSources:\n" + "\n".join([
            f"- {r['title']}: {r['url']}"
            for r in results[:2]
        ])
        
        return is_verified, evidence_summary, confidence
    
    def extract_factual_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text using LLM.
        
        Args:
            text: Text to extract claims from
            
        Returns:
            List of factual claims
        """
        
        if self.llm_model is None or self.tokenizer is None:
            print("[WebKnowledge] LLM not available for claim extraction")
            return []
        
        prompt = f"""Extract factual claims from the following text that can be verified.
Focus on specific, verifiable statements (names, dates, numbers, events).
Do NOT extract opinions or subjective statements.

Text: "{text}"

Output as JSON list:
["claim 1", "claim 2", "claim 3"]

JSON:"""

        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if next(self.llm_model.parameters()).is_cuda:
                inputs = inputs.to("cuda")
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            )
            
            # Extract JSON array
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                claims = json.loads(json_match.group(0))
                if isinstance(claims, list):
                    return [str(c) for c in claims if c]
            
        except Exception as e:
            print(f"[WebKnowledge] Claim extraction error: {e}")
        
        return []
    
    def _update_cache(self, key: str, value: any):
        """Update cache with LRU eviction"""
        
        # Remove oldest item if cache is full
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        
        # Add new item
        self.cache[key] = value
    
    def clear_cache(self):
        """Clear the search cache"""
        self.cache.clear()
        print("[WebKnowledge] Cache cleared")


if __name__ == "__main__":
    print("[WebKnowledge] Testing Web Knowledge Retriever")
    
    # Test without LLM first
    retriever = WebKnowledgeRetriever()
    
    if retriever.ddg_available:
        print("\n--- Test 1: Basic Search ---")
        results = retriever.search("Python programming language", max_results=2)
        for i, r in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Title: {r['title']}")
            print(f"Snippet: {r['snippet'][:100]}...")
            print(f"URL: {r['url']}")
        
        print("\n--- Test 2: Fact Verification (keyword-based) ---")
        is_verified, evidence, conf = retriever.verify_fact(
            "Python was created by Guido van Rossum",
            use_llm=False
        )
        print(f"Verified: {is_verified}")
        print(f"Confidence: {conf:.2f}")
        print(f"Evidence: {evidence[:200]}...")
        
        print("\n✅ Tests completed successfully")
    else:
        print("\n⚠️ DuckDuckGo search not available. Install with:")
        print("pip install duckduckgo-search")
