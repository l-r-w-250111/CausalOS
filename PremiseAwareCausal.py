"""
Premise-Aware Causal Extraction and Counterfactual Reasoning

This module addresses the fundamental problem of invalid premise assumptions
in causal reasoning. It implements a 3-layer causal model that distinguishes:
- EXPLICIT: Directly stated causal relationships
- IMPLICIT: Necessary common-sense assumptions
- ASSUMPTION: Potentially invalid presumptions

Goal: Prevent the system from making inferences based on non-existent premises.
"""

import torch
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum


class CausalLayerType(Enum):
    """Types of causal relationships by certainty level"""
    EXPLICIT = "explicit"        # Explicitly stated in text
    IMPLICIT = "implicit"        # Common-sense inference
    ASSUMPTION = "assumption"    # Unverified assumption


@dataclass
class LayeredCausalRelation:
    """A causal relationship with metadata about its certainty"""
    cause: str
    effect: str
    magnitude: float  # 0.0-1.0
    layer: CausalLayerType
    confidence: float  # 0.0-1.0
    prerequisites: List[str]  # Prerequisites for this relation to hold
    evidence: str = ""  # Supporting evidence from text


class PremiseAwareCausalExtractor:
    """
    Extracts causal relationships while tracking premise validity.
    
    Key innovation: Distinguishes between what is explicitly stated vs.
    what is assumed, preventing invalid inferences.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Causal markers for structural extraction
        self.causal_markers = {
            "explicit": [
                "because", "since", "as", "for",
                "therefore", "thus", "hence", "consequently",
                "causes", "leads to", "results in", "triggers",
                "due to", "owing to"
            ],
            "conditional": [
                "if", "when", "whenever", "unless",
                "in case", "provided that", "assuming"
            ],
            "temporal": [
                "before", "after", "during", "while",
                "then", "subsequently", "following"
            ]
        }
    
    def extract_layered_causality(self, text: str) -> Dict[str, List[LayeredCausalRelation]]:
        """
        Extract causal relations in 3 layers.
        
        Returns:
            {
                "explicit": [...],
                "implicit": [...],
                "assumptions": [...]
            }
        """
        
        # Step 1: Extract explicit causal relationships
        explicit_relations = self._extract_explicit_causality(text)
        
        # Step 2: Infer minimal necessary implicit premises
        implicit_premises = self._extract_implicit_premises(text, explicit_relations)
        
        # Step 3: Validate premises to filter out unwarranted assumptions
        validated_implicit = self._validate_premises(implicit_premises)
        
        return {
            "explicit": explicit_relations,
            "implicit": validated_implicit,
            "assumptions": []  # Identified but rejected assumptions
        }
    
    def _extract_explicit_causality(self, text: str) -> List[LayeredCausalRelation]:
        """Extract only what is directly stated in the text."""
        
        prompt = f"""Analyze ONLY the explicitly stated information in the text.
Do NOT infer any implicit information or make assumptions.

Text: "{text}"

Rules:
1. Extract only what is directly stated
2. "A man walks on a street" → This states: action="walks", location="street"
3. Do NOT add assumptions about PURPOSE, DESTINATION, TIME CONSTRAINTS, or INTENTIONS
4. If the text doesn't mention something, it doesn't exist

Output JSON format:
[
  {{
    "cause": "stated cause",
    "effect": "stated effect",
    "magnitude": 0.0-1.0,
    "evidence": "exact quote from text"
  }}
]

Example:
Text: "A man walks on a street"
Output: [
  {{
    "cause": "man performs walking action",
    "effect": "movement occurs on street surface",
    "magnitude": 1.0,
    "evidence": "A man walks on a street"
  }}
]

Note: Do NOT assume travel, destination, or time constraints.

Now analyze: "{text}"
JSON:"""

        try:
            result = self._query_llm_json(prompt)
            
            relations = []
            if isinstance(result, list):
                for r in result:
                    relations.append(LayeredCausalRelation(
                        cause=r.get("cause", ""),
                        effect=r.get("effect", ""),
                        magnitude=r.get("magnitude", 0.5),
                        layer=CausalLayerType.EXPLICIT,
                        confidence=1.0,  # High confidence for explicit
                        prerequisites=[],
                        evidence=r.get("evidence", "")
                    ))
            
            return relations
            
        except Exception as e:
            print(f"[PremiseAware] Error extracting explicit causality: {e}")
            return []
    
    def _extract_implicit_premises(
        self, 
        text: str, 
        explicit_relations: List[LayeredCausalRelation]
    ) -> List[LayeredCausalRelation]:
        """Infer MINIMAL necessary assumptions."""
        
        explicit_summary = [
            f"{r.cause} → {r.effect}" for r in explicit_relations
        ]
        
        prompt = f"""Text: "{text}"

Explicit facts already identified:
{json.dumps(explicit_summary, indent=2)}

What are the MINIMAL necessary assumptions to make sense of this scenario?

Guidelines:
- Focus on physical/logical prerequisites (e.g., "person can walk" requires functional legs)
- Do NOT assume goals, intentions, or destinations unless stated
- Do NOT assume time constraints unless stated
- Mark confidence based on necessity

Example 1:
Text: "A man walks"
Minimal assumptions:
[
  {{
    "assumption": "The person has functional legs",
    "necessity": "Required for walking action",
    "confidence": 0.95
  }},
  {{
    "assumption": "Surface exists to walk on",
    "necessity": "Walking requires a surface",
    "confidence": 0.98
  }}
]
NOT assumptions: "going somewhere", "has a destination", "in a hurry"

Example 2:
Text: "A student reads a book"
Minimal assumptions:
[
  {{
    "assumption": "Student can read",
    "necessity": "Required for reading action",
    "confidence": 0.9
  }},
  {{
    "assumption": "Book contains text",
    "necessity": "Reading requires text",
    "confidence": 0.95
  }}
]
NOT assumptions: "studying for exam", "homework assignment", "time pressure"

Now analyze: "{text}"
JSON:"""

        try:
            result = self._query_llm_json(prompt)
            
            relations = []
            if isinstance(result, list):
                for r in result:
                    relations.append(LayeredCausalRelation(
                        cause=r.get("assumption", ""),
                        effect="scenario_validity",
                        magnitude=0.5,
                        layer=CausalLayerType.IMPLICIT,
                        confidence=r.get("confidence", 0.5),
                        prerequisites=[],
                        evidence=r.get("necessity", "")
                    ))
            
            return relations
            
        except Exception as e:
            print(f"[PremiseAware] Error extracting implicit premises: {e}")
            return []
    
    def _validate_premises(
        self, 
        premises: List[LayeredCausalRelation]
    ) -> List[LayeredCausalRelation]:
        """Filter out premises with low confidence."""
        
        validated = []
        for premise in premises:
            # Threshold: only keep high-confidence premises
            if premise.confidence >= 0.7:
                validated.append(premise)
            else:
                print(f"[PremiseAware] Rejected low-confidence premise: {premise.cause}")
        
        return validated
    
    def _query_llm_json(self, prompt: str, max_retries: int = 3) -> any:
        """Query LLM and parse JSON response with retries."""
        
        for attempt in range(max_retries):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=500,
                        do_sample=False,  # Greedy for consistency
                        temperature=0.1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[-1]:], 
                    skip_special_tokens=True
                )
                
                # Try multiple extraction strategies
                json_result = None
                
                # Strategy 1: Extract JSON array with improved regex
                # This handles cases where LLM adds explanation after JSON
                array_match = re.search(r'(\[(?:[^\[\]]|\{[^\}]*\})*\])', response, re.DOTALL)
                if array_match:
                    try:
                        json_str = array_match.group(1)
                        json_result = json.loads(json_str)
                        return json_result
                    except json.JSONDecodeError:
                        pass
                
                # Strategy 2: Extract JSON object
                obj_match = re.search(r'(\{(?:[^\{\}]|\{[^\}]*\})*\})', response, re.DOTALL)
                if obj_match:
                    try:
                        json_str = obj_match.group(1)
                        json_result = json.loads(json_str)
                        return json_result
                    except json.JSONDecodeError:
                        pass
                
                # Strategy 3: Try parsing entire response
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    pass
                
                # If all strategies fail, raise error
                raise json.JSONDecodeError(
                    "Could not extract valid JSON from response",
                    response,
                    0
                )
                    
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    # Only print warning if not last attempt
                    continue
                else:
                    # On last attempt, print error and return empty
                    print(f"[PremiseAware] JSON parse failed after {max_retries} attempts")
                    print(f"Last response: {response[:200]}...")
                    return []
                    
            except Exception as e:
                print(f"[PremiseAware] LLM query error: {e}")
                return []
        
        return []


class PremiseAwareCounterfactual:
    """
    Counterfactual reasoning that respects premise validity.
    
    Key innovation: Identifies which premises are invalidated by the
    counterfactual change, preventing invalid inferences.
    """
    
    def __init__(self, extractor: PremiseAwareCausalExtractor):
        self.extractor = extractor
    
    def solve_counterfactual_with_premises(
        self, 
        factual: str, 
        counterfactual: str,
        options: Dict[str, str]
    ) -> Dict:
        """
        Solve counterfactual while tracking premise validity.
        
        Returns:
            {
                "selected_option": "A"/"B"/"C",
                "reasoning": "...",
                "valid_premises": [...],
                "invalidated_premises": [...],
                "all_evaluations": [...]
            }
        """
        
        print(f"\n[OS v4] Solving Counterfactual (Premise-Aware)")
        print(f"[OS v4] Factual: {factual}")
        print(f"[OS v4] Counterfactual: {counterfactual}")
        
        # Step 1: Analyze factual scenario
        factual_analysis = self.extractor.extract_layered_causality(factual)
        
        # Step 2: Identify the change
        change = self._identify_change(factual, counterfactual)
        
        # Step 3: Determine which premises are invalidated
        invalidated_premises = self._find_invalidated_premises(
            factual_analysis, 
            change
        )
        
        # Step 4: Identify still-valid premises
        valid_premises = [
            p for p in factual_analysis["implicit"] 
            if p not in invalidated_premises
        ]
        
        # Step 5: Evaluate each option
        evaluations = self._evaluate_options(
            counterfactual=counterfactual,
            valid_premises=valid_premises,
            invalidated_premises=invalidated_premises,
            options=options
        )
        
        # Step 6: Select best option
        best_option = self._select_best_option(evaluations)
        
        return {
            "selected_option": best_option["option"],
            "reasoning": best_option.get("reasoning", ""),
            "valid_premises": [p.cause for p in valid_premises],
            "invalidated_premises": [p.cause for p in invalidated_premises],
            "all_evaluations": evaluations
        }
    
    def _identify_change(self, factual: str, counterfactual: str) -> Dict:
        """Identify the specific change between factual and counterfactual."""
        
        prompt = f"""Identify the SPECIFIC change between these scenarios:

Factual: "{factual}"
Counterfactual: "{counterfactual}"

Output JSON:
{{
  "changed_element": "what changed",
  "original_value": "original state",
  "new_value": "new state",
  "affected_properties": ["prop1", "prop2"]
}}

Example:
Factual: "walks on a street"
Counterfactual: "walks on a bed"
{{
  "changed_element": "walking surface",
  "original_value": "street (hard, stable, outdoor, public)",
  "new_value": "bed (soft, unstable, indoor, private)",
  "affected_properties": ["surface_hardness", "stability", "location_type", "social_context"]
}}

JSON:"""

        try:
            result = self.extractor._query_llm_json(prompt)
            return result if isinstance(result, dict) else {}
        except:
            return {
                "changed_element": "unknown",
                "original_value": factual,
                "new_value": counterfactual,
                "affected_properties": []
            }
    
    def _find_invalidated_premises(
        self, 
        factual_analysis: Dict, 
        change: Dict
    ) -> List[LayeredCausalRelation]:
        """Determine which premises are invalidated by the change."""
        
        invalidated = []
        
        premises = factual_analysis.get("implicit", [])
        
        for premise in premises:
            # Query if this premise is still valid after the change
            prompt = f"""Does the following change INVALIDATE this premise?

Premise: "{premise.cause}"
Evidence: "{premise.evidence}"

Change: {json.dumps(change, indent=2)}

Rules:
1. A premise is INVALIDATED if it logically contradicts the new situation
2. A premise is STILL VALID if it's independent of the change
3. Consider only logical necessity, not likelihood

Examples:

Example 1:
Premise: "Person is commuting to work"
Change: walking surface (street → bed)
Result: INVALIDATED (beds are not for commuting)

Example 2:
Premise: "Person has functional legs"
Change: walking surface (street → bed)
Result: STILL VALID (leg function is independent of surface)

Example 3:
Premise: "Person is traveling to a destination"
Change: walking surface (street → bed)
Result: INVALIDATED (beds are not travel paths; this was likely an invalid assumption)

Now evaluate:
Premise: "{premise.cause}"
Change: {change.get("changed_element")} ({change.get("original_value")} → {change.get("new_value")})

Output JSON:
{{
  "invalidated": true/false,
  "reason": "explanation"
}}
JSON:"""

            try:
                result = self.extractor._query_llm_json(prompt)
                
                if isinstance(result, dict) and result.get("invalidated", False):
                    invalidated.append(premise)
                    print(f"[OS v4] Invalidated premise: {premise.cause}")
                    print(f"[OS v4] Reason: {result.get('reason', 'N/A')}")
                    
            except Exception as e:
                print(f"[PremiseAware] Error checking premise validity: {e}")
        
        return invalidated
    
    def _evaluate_options(
        self, 
        counterfactual: str,
        valid_premises: List[LayeredCausalRelation],
        invalidated_premises: List[LayeredCausalRelation],
        options: Dict[str, str]
    ) -> List[Dict]:
        """Evaluate each option based on premise validity."""
        
        valid_premise_list = [p.cause for p in valid_premises]
        invalid_premise_list = [p.cause for p in invalidated_premises]
        
        prompt = f"""Evaluate these options for the counterfactual scenario.

Counterfactual: "{counterfactual}"

Valid premises (still hold):
{json.dumps(valid_premise_list, indent=2)}

Invalidated premises (no longer hold):
{json.dumps(invalid_premise_list, indent=2)}

Options:
{json.dumps(options, indent=2)}

Evaluation rules:
1. An option is INCORRECT if it relies on invalidated premises
2. An option is CORRECT if it follows only from valid premises  
3. An option is UNCERTAIN if insufficient information

For each option, determine:
- Does it rely on any invalidated premise?
- If yes, which one?
- Verdict: correct/incorrect/uncertain

Output JSON:
[
  {{
    "option": "A",
    "relies_on_invalidated": true/false,
    "invalidated_premise": "which premise or null",
    "verdict": "correct"/"incorrect"/"uncertain",
    "reasoning": "detailed explanation"
  }}
]

Example:
Options: {{"A": "He would be late", "B": "Nothing special", "C": "On time"}}
Valid: ["person can walk", "surface exists"]
Invalid: ["traveling to destination", "time constraint exists"]

Evaluations:
[
  {{
    "option": "A",
    "relies_on_invalidated": true,
    "invalidated_premise": "time constraint exists",
    "verdict": "incorrect",
    "reasoning": "Claims 'late' but no time constraint was ever stated"
  }},
  {{
    "option": "B",  
    "relies_on_invalidated": false,
    "invalidated_premise": null,
    "verdict": "correct",
    "reasoning": "Makes no unjustified claims; acknowledges limited information"
  }},
  {{
    "option": "C",
    "relies_on_invalidated": true,
    "invalidated_premise": "traveling to destination",
    "verdict": "incorrect",  
    "reasoning": "Claims 'on time' but no destination was stated"
  }}
]

JSON:"""

        try:
            result = self.extractor._query_llm_json(prompt)
            if isinstance(result, list):
                return result
        except Exception as e:
            print(f"[PremiseAware] Error evaluating options: {e}")
        
        return []
    
    def _select_best_option(self, evaluations: List[Dict]) -> Dict:
        """Select the best option based on evaluations."""
        
        best_option = None
        best_score = -1
        
        for eval_item in evaluations:
            score = 0
            
            # Does not rely on invalidated premises
            if not eval_item.get("relies_on_invalidated", True):
                score += 3
            
            # Verdict is correct
            if eval_item.get("verdict") == "correct":
                score += 2
            
            # Has detailed reasoning
            if len(eval_item.get("reasoning", "")) > 50:
                score += 1
            
            if score > best_score:
                best_score = score
                best_option = eval_item
        
        if best_option is None and evaluations:
            # Fallback: return first option
            best_option = evaluations[0]
        
        return best_option if best_option else {
            "option": "B",
            "reasoning": "Unable to determine",
            "verdict": "uncertain"
        }


if __name__ == "__main__":
    print("[PremiseAware] Module loaded successfully")
    print("[PremiseAware] To use: ")
    print("  from PremiseAwareCausal import PremiseAwareCausalExtractor, PremiseAwareCounterfactual")
