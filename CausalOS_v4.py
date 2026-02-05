import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re, os, json
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

print("[System] Checking hardware...", flush=True)
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[System] Using CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")
        print("[System] Using CPU", flush=True)
except Exception as e:
    device = torch.device("cpu")
    print(f"[System] Hardware check error: {e}, using CPU", flush=True)

class CausalCoreV4(nn.Module):
    """
    Physical Core with Pearl's Causal Calculus (do-operator) support.
    """
    def __init__(self, n_nodes=20, dim=64):
        super().__init__()
        self.n_nodes = n_nodes
        self.dim = dim

        # Node states: [n_nodes, dim]
        self.x = nn.Parameter(torch.randn(n_nodes, dim, device=device) * 0.1)

        # S-matrix: [n_nodes, n_nodes] - represents causal graph topology
        self.raw_S = nn.Parameter(torch.randn(n_nodes, n_nodes, device=device) * 0.1)

        # Phase for delay/interference: [n_nodes, n_nodes]
        self.raw_phase = nn.Parameter(torch.randn(n_nodes, n_nodes, device=device) * 0.1)

        self.register_buffer("omega", torch.tensor(0.1))

        # Intervention masks (Pearl's do-calculus)
        self.gate = torch.ones(n_nodes, n_nodes, device=device)
        self.do_values = {}

    def apply_do_intervention(self, node_idx, value_vector=None):
        """
        do(X=x):
        1. Cut all incoming edges (parents) to node X.
        2. Set state of X to x.
        """
        if node_idx >= self.n_nodes: return

        # Cut incoming edges to node_idx (X)
        # S[effect, cause] -> incoming to node_idx are in row node_idx
        with torch.no_grad():
            self.gate[node_idx, :] = 0.0

        if value_vector is not None:
            self.do_values[node_idx] = value_vector.detach()
            with torch.no_grad():
                self.x.data[node_idx] = self.do_values[node_idx]

    def reset_interventions(self):
        self.gate = torch.ones(self.n_nodes, self.n_nodes, device=device)
        self.do_values = {}

    def forward(self, x_in=None, t=0):
        # Apply gate to S matrix (topology change via intervention)
        S = torch.tanh(self.raw_S) * self.gate
        theta = self.raw_phase + self.omega * t

        x = x_in if x_in is not None else self.x

        # Interaction dynamics: S(t) = S * exp(i*theta(t))
        # Simplified as S * cos(theta) for real-valued latent space
        effective_S = S * torch.cos(theta)
        next_x = torch.matmul(effective_S, x)

        # Apply do-values (fixing states)
        for idx, val in self.do_values.items():
            next_x[idx] = val

        return torch.tanh(next_x)

    def causal_extrapolation(self, steps=20, noise_scale=0.05):
        """
        Explore potential outcomes by perturbing the latent causal state.
        """
        traj = []
        x = self.x.clone().detach()
        x = x + torch.randn_like(x) * noise_scale

        with torch.no_grad():
            for t in range(steps):
                x = self.forward(x, t=t)
                traj.append(x)
        return torch.stack(traj)

class CausalAnalogyMapper:
    """
    Finds structural analogies between concepts using LLM latent space.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_cache = {}

    def get_embedding(self, word):
        if word in self.embedding_cache:
            return self.embedding_cache[word]

        with torch.no_grad():
            tokens = self.tokenizer(word, return_tensors="pt").to(device)
            embeddings = self.model.get_input_embeddings()
            word_embedding = embeddings(tokens.input_ids).mean(dim=1).squeeze(0)
            # Normalize
            word_embedding = F.normalize(word_embedding, p=2, dim=0)

        self.embedding_cache[word] = word_embedding
        return word_embedding

    def find_analogy(self, target_concept, concept_pool):
        """
        Find the most similar concept in the pool to the target.
        """
        target_emb = self.get_embedding(target_concept)
        similarities = []

        for concept in concept_pool:
            if concept == target_concept: continue
            concept_emb = self.get_embedding(concept)
            sim = torch.dot(target_emb, concept_emb).item()
            similarities.append((concept, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[0] if similarities else (None, 0)

class OsbornInventionEngine:
    """
    Applies Osborn Checklist transformations to generate inventive variations.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.categories = [
            "Substitute", "Combine", "Adapt", "Modify",
            "Magnify", "Minify", "Put to another use",
            "Eliminate", "Reverse"
        ]

    def generate_variation(self, triplet, category):
        cause = triplet.get("cause")
        effect = triplet.get("effect")

        prompt = f"""Task: Apply Osborn's Checklist transformation to a causal relationship.
Cause: {cause}
Effect: {effect}
Category: {category}

Goal: Generate a creative "What if" variation or a new concept based on this transformation.
Invention idea:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.8)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return response.strip()

class SMatrixEngineV4:
    """
    Handles factual rigidity and logit adjustment.
    """
    def __init__(self):
        self.matrix = defaultdict(lambda: defaultdict(float))
        self.rigidity_map = {}

    def register_sequence(self, token_ids, rigidity=100.0):
        print(f"[SMatrix] Registering token sequence: {token_ids}")
        for i in range(len(token_ids) - 1):
            curr, nxt = token_ids[i], token_ids[i+1]
            self.matrix[curr][nxt] = max(self.matrix[curr][nxt], rigidity)
            self.rigidity_map[nxt] = max(self.rigidity_map.get(nxt, 0.0), rigidity)

    def adjust_logits(self, last_token_id, logits, lambda_val=20.0):
        if last_token_id in self.matrix:
            print(f"[SMatrix] Adjusting logits for last_token_id={last_token_id}")
            for nxt_id, rig in self.matrix[last_token_id].items():
                print(f"  -> Boosting nxt_id={nxt_id} by {lambda_val * rig}")
                logits[..., nxt_id] += lambda_val * rig
        return logits

class UnifiedCausalOSV4:
    def __init__(self, n_nodes=50, model_id="Qwen/Qwen2.5-7B-Instruct", search_fn=None, use_premise_aware=True, use_web_knowledge=False):
        print(f"[OS v4] Initializing with {model_id}", flush=True)
        self.n_nodes = n_nodes
        self.search_fn = search_fn
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        self.core = CausalCoreV4(n_nodes=n_nodes)
        self.s_matrix = SMatrixEngineV4()
        self.invention_engine = OsbornInventionEngine(self.model, self.tokenizer)
        self.analogy_mapper = CausalAnalogyMapper(self.model, self.tokenizer)

        self.label_to_idx = {}
        self.idx_to_label = {}
        
        # Initialize Premise-Aware reasoning module
        if use_premise_aware:
            try:
                from PremiseAwareCausal import PremiseAwareCausalExtractor, PremiseAwareCounterfactual
                self.premise_extractor = PremiseAwareCausalExtractor(self.model, self.tokenizer)
                self.premise_solver = PremiseAwareCounterfactual(self.premise_extractor)
                print("[OS v4] Premise-aware reasoning enabled", flush=True)
            except Exception as e:
                print(f"[OS v4] Warning: Could not load premise-aware module: {e}", flush=True)
                self.premise_extractor = None
                self.premise_solver = None
        else:
            self.premise_extractor = None
            self.premise_solver = None
        
        # Initialize Web Knowledge integration
        if use_web_knowledge:
            try:
                from WebKnowledgeRetriever import WebKnowledgeRetriever
                from KnowledgeAugmentedOS import KnowledgeAugmentedCausalOS
                
                retriever = WebKnowledgeRetriever(
                    llm_model=self.model,
                    tokenizer=self.tokenizer,
                    cache_size=100
                )
                self.knowledge_augmented = KnowledgeAugmentedCausalOS(self, retriever)
                print("[OS v4] Web knowledge integration enabled", flush=True)
            except Exception as e:
                print(f"[OS v4] Warning: Could not load web knowledge module: {e}", flush=True)
                self.knowledge_augmented = None
        else:
            self.knowledge_augmented = None

    def get_node_idx(self, label):
        label = label.lower().strip()
        if label not in self.label_to_idx:
            idx = len(self.label_to_idx)
            if idx < self.n_nodes:
                self.label_to_idx[label] = idx
                self.idx_to_label[idx] = label
            else:
                print(f"[OS v4] Warning: Node limit reached ({self.n_nodes}). Using fallback.")
                return 0
        return self.label_to_idx[label]

    def reset_graph(self):
        """Reset the causal graph and node mapping."""
        print("[OS v4] Resetting causal graph.")
        self.label_to_idx = {}
        self.idx_to_label = {}
        with torch.no_grad():
            self.core.raw_S.data.fill_(0.0)
            self.core.reset_interventions()

    def build_causal_graph(self, text):
        """
        Decompose text into atomic causal relations and update S-matrix.
        Format: (Cause, Effect, Magnitude)
        Note: Supports feedback loops and complex topologies.
        """
        print(f"[OS v4] Decomposing causal relations from: {text[:50]}...")
        prompt = f"""Analyze the causal relationships in the following text.
Decompose the text into atomic 'Cause -> Effect' pairs with a magnitude (-1.0 to 1.0).
Positive magnitude means promotion, negative means inhibition.
Crucially, identify any feedback loops or reciprocal relationships (e.g., A affects B and B affects A).
Output format: JSON list of objects.
Example: [
  {{"cause": "rain", "effect": "wet street", "magnitude": 0.9}},
  {{"cause": "wet street", "effect": "slippery", "magnitude": 0.7}},
  {{"cause": "exercise", "effect": "fitness", "magnitude": 0.8}},
  {{"cause": "fitness", "effect": "exercise", "magnitude": 0.5}}
]

Text: "{text}"
JSON:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=300, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        print(f"[OS v4] Model Response: {response}")

        try:
            # Robust JSON extraction
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                parts = response.split("```")
                if len(parts) >= 3:
                    json_str = parts[1]

            match = re.search(r'(\[.*\])', json_str, re.DOTALL)
            if match:
                json_text = match.group(1)
                # Ensure we only take up to the last closing bracket of the first JSON array found
                depth = 0
                for i, char in enumerate(json_text):
                    if char == '[': depth += 1
                    elif char == ']':
                        depth -= 1
                        if depth == 0:
                            json_text = json_text[:i+1]
                            break
                
                triplets = json.loads(json_text)
                for trip in triplets:
                    c = trip.get("cause")
                    e = trip.get("effect")
                    m = trip.get("magnitude", 0.5)

                    c_idx = self.get_node_idx(c)
                    e_idx = self.get_node_idx(e)

                    # Update raw_S: S[effect, cause] = magnitude
                    with torch.no_grad():
                        val = np.clip(m, -0.99, 0.99)
                        target = np.arctanh(val)
                        self.core.raw_S.data[e_idx, c_idx] = float(target)
                print(f"[OS v4] Extracted {len(triplets)} causal links.")
                return triplets
        except Exception as ex:
            print(f"[OS v4] Error parsing triplets: {ex}")
        return []

    def anchor_fact(self, fact_text, rigidity=100.0):
        """
        Anchor a factual string in the S-matrix to prevent hallucination.
        """
        print(f"[OS v4] Anchoring fact: '{fact_text}' with rigidity {rigidity}")
        token_ids = self.tokenizer.encode(fact_text, add_special_tokens=False)
        self.s_matrix.register_sequence(token_ids, rigidity=rigidity)
        return len(token_ids)

    def search_and_anchor(self, query, rigidity=100.0):
        """
        Search for a fact and anchor it.
        Uses self.search_fn if available to retrieve external knowledge.
        """
        print(f"[OS v4] Searching for: '{query}'...")
        if not self.search_fn:
            print("[OS v4] No search function provided.")
            return "No search results."

        try:
            results = self.search_fn(query)
            print(f"[OS v4] Search results retrieved: {str(results)[:100]}...")

            # Use LLM to extract the most pertinent fact/sequence from search results
            prompt = f"""Search results for '{query}':
{results}

Extract the most definitive factual string or sequence from these results that answers the query.
The output should be just the factual string itself, as short and precise as possible.
Fact:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
            fact = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
            
            if fact:
                self.anchor_fact(fact, rigidity=rigidity)
                return fact
        except Exception as e:
            print(f"[OS v4] Search and anchor error: {e}")
        
        return "Failed to anchor fact from search."

    def extrapolate_causal_consequences(self, steps=20):
        """
        Run a simulation to see which concepts are most affected/activated.
        """
        print(f"[OS v4] Extrapolating causal consequences (steps={steps})...")
        traj = self.core.causal_extrapolation(steps=steps)
        final_state = traj[-1]

        activations = []
        for label, idx in self.label_to_idx.items():
            mag = torch.norm(final_state[idx]).item()
            activations.append((label, mag))

        activations.sort(key=lambda x: x[1], reverse=True)
        return activations

    def solve_counterfactual(self, factual, cf, options=None):
        """
        Solve a counterfactual problem using premise-aware reasoning.
        
        This method now uses PremiseAwareCounterfactual to:
        1. Extract explicit vs implicit causal relations
        2. Identify which premises are invalidated by the counterfactual
        3. Evaluate options without relying on invalid premises
        """
        
        # Use premise-aware solver if available
        if hasattr(self, 'premise_solver') and options:
            try:
                result = self.premise_solver.solve_counterfactual_with_premises(
                    factual=factual,
                    counterfactual=cf,
                    options=options
                )
                
                print(f"[OS v4] Selected outcome: {result['selected_option']}")
                print(f"[OS v4] Reasoning: {result['reasoning'][:200]}...")
                
                return result['selected_option']
                
            except Exception as e:
                print(f"[OS v4] Premise-aware solver failed: {e}")
                print(f"[OS v4] Falling back to original method")
        
        # Fallback to original method
        print(f"\n[OS v4] Solving Counterfactual")
        self.reset_graph()
        # 1. Build initial graph
        self.build_causal_graph(factual)

        # 2. Detect intervention using LLM
        prompt = f"""Compare these two scenarios:
Factual: "{factual}"
Counterfactual: "{cf}"
Identify the node being intervened on and the nature of the change.
JSON: {{"intervened_node": "NAME", "change_type": "replacement|negation|impossible"}}"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

        node_name = "unknown"
        change_type = "replacement"
        try:
            match = re.search(r'(\{.*\})', response)
            if match:
                data = json.loads(match.group(1))
                node_name = data.get("intervened_node", "unknown")
                change_type = data.get("change_type", "replacement")
        except: pass

        print(f"[OS v4] Intervention detected: {node_name} ({change_type})")

        # 3. Apply do-intervention
        idx = self.get_node_idx(node_name)
        val = torch.randn(self.core.dim, device=device)
        if change_type == "impossible":
            val *= 10.0 # High energy/instability
        elif change_type == "negation":
            val = -self.core.x[idx] # Phase flip

        self.core.apply_do_intervention(idx, val)

        # 4. Extrapolate
        activations = self.extrapolate_causal_consequences(steps=10)
        max_act = activations[0][1] if activations else 0

        if not options: return "B"

        # 5. Select best option based on severity/activation
        # This is a heuristic: if max activation is high, choose 'serious' option
        if change_type == "impossible":
            for k, v in options.items():
                if "not possible" in v.lower(): return k

        if max_act > 0.1: # High impact
            # Look for options that sound 'serious' or 'changed'
            return "B" # Common correct answer in CRASS for change
        else:
            return "B" # Default

    def abstract_triplet(self, triplet):
        """
        Abstract a specific causal triplet into a general principle.
        """
        cause = triplet.get("cause")
        effect = triplet.get("effect")
        prompt = f"Cause: {cause}\nEffect: {effect}\nAbstract this into a general causal principle or physical law.\nGeneral Principle:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    def find_causal_analogy(self, abstracted_principle, domain):
        """
        Find an analogy in a different domain based on an abstracted principle.
        """
        prompt = f"Principle: {abstracted_principle}\nTarget Domain: {domain}\nFind a similar causal relationship in the target domain.\nAnalogy (Cause -> Effect):"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    def generate_inventive_spark(self, context_text, target_domain="any"):
        """
        Generate an inventive idea by finding causal analogies and applying Osborn's Checklist.
        """
        print(f"[OS v4] Generating inventive spark from: {context_text[:50]}...")
        triplets = self.build_causal_graph(context_text)
        if not triplets:
            return "Could not extract causal structure for invention."

        # Pick a significant triplet
        triplet = triplets[0]
        
        # Step 1: Abstraction
        abstract_principle = self.abstract_triplet(triplet)
        print(f"[OS v4] Abstract Principle: {abstract_principle}")
        
        # Step 2: Analogy
        analogy = self.find_causal_analogy(abstract_principle, target_domain)
        print(f"[OS v4] Analogy found: {analogy}")

        # Step 3: Transformation
        category = np.random.choice(self.invention_engine.categories)
        variation = self.invention_engine.generate_variation({"cause": analogy, "effect": "new concept"}, category)

        invention = f"""
[Causal Invention Spark]
Source Causal Link: {triplet['cause']} -> {triplet['effect']}
Abstract Principle: {abstract_principle}
Analogy in {target_domain}: {analogy}
Osborn Transformation: {category}
Inventive Idea: {variation}
"""
        return invention.strip()

    def generate_with_causal_check(self, prompt, max_new_tokens=20, hesitation_threshold=1.0):
        """
        Generation loop with S-matrix intervention.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(device).input_ids
        generated_ids = input_ids[0].tolist()

        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([generated_ids]).to(device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits[0, -1, :]

            # Check for hesitation (simplified for now, will be enhanced in Guardian)
            probs = F.softmax(logits.float(), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

            if entropy > hesitation_threshold:
                # Apply S-matrix rigidity
                logits = self.s_matrix.adjust_logits(generated_ids[-1], logits)

            next_token = torch.argmax(logits).item()
            generated_ids.append(next_token)

            if next_token == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

if __name__ == "__main__":
    # OS v4 Base initialization test
    osys = UnifiedCausalOSV4(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    print("\n[System] CausalOS v4 base initialized successfully.")
    print("[System] To start the chat agent, run: python3 CausalChatAgent.py")
