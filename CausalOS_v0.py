import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re, json, os
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from causal_node_extractor import CausalNodeExtractor

# ==========================================================
# DEVICE
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 1) LLM 観測層：剛性・Phi・CII の抽出
# ==========================================================
class LLMCausalObserver:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto"
        )
        self.expected_rigidity = defaultdict(float)
        self.known_tokens = {}
        self.phi_history = []

    def reset(self):
        self.phi_history = []

    # ---- 事実から「期待剛性」を学習 ----
    def train_topology(self, texts):
        print("[Observer] Learning expected rigidity from facts...")
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            ids = inputs.input_ids[0].tolist()

            with torch.no_grad():
                outputs = self.model(**inputs)

            for k in range(len(ids)-1):
                p = F.softmax(outputs.logits[0, k, :].float(), dim=-1)
                top_p, _ = torch.topk(p, 50)
                rig = 1.0 / (torch.var(top_p).item() + 1e-6)

                token_id = ids[k+1]
                self.expected_rigidity[token_id] = max(
                    self.expected_rigidity[token_id], rig
                )
                self.known_tokens[token_id] = self.tokenizer.decode([token_id])

    # ---- Phi（位相） ----
    def calculate_phi(self, logits):
        logits_f = logits.float()
        probs = F.softmax(logits_f, dim=-1)
        top_v, _ = torch.topk(probs, 50)
        phi = 1.0 / (torch.var(top_v).item() + 1e-6)
        return phi

    # ---- 迷いスコア：エントロピー ----
    def calculate_entropy(self, logits):
        logits_f = logits.float()
        probs = F.softmax(logits_f, dim=-1)
        log_probs = F.log_softmax(logits_f, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.item()

    # ---- 迷いスコア：Top-1 確率 ----
    def calculate_top1_prob(self, logits):
        logits_f = logits.float()
        probs = F.softmax(logits_f, dim=-1)
        return torch.max(probs).item()

    # ---- 迷いスコア：尖度 (Kurtosis) ----
    def calculate_kurtosis(self, logits):
        logits_f = logits.float()
        probs = F.softmax(logits_f, dim=-1)
        mean = torch.mean(probs)
        std = torch.std(probs)
        kurt = torch.mean(((probs - mean) / (std + 1e-10))**4)
        return kurt.item()

    # ---- CII（位相加速度） ----
    def calculate_cii(self):
        if len(self.phi_history) < 3:
            return 0.0
        d2_phi = (
            self.phi_history[-1]
            - 2 * self.phi_history[-2]
            + self.phi_history[-3]
        )
        return d2_phi ** 2

    # ---- LLMを1ステップ動かし、各種指標を取得 ----
    def step(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]

        phi = self.calculate_phi(logits)
        self.phi_history.append(phi)
        cii = self.calculate_cii()
        
        entropy = self.calculate_entropy(logits)
        top1_prob = self.calculate_top1_prob(logits)
        kurtosis = self.calculate_kurtosis(logits)

        next_token = torch.argmax(logits).item()
        
        metrics = {
            "phi": phi,
            "cii": cii,
            "entropy": entropy,
            "top1_prob": top1_prob,
            "kurtosis": kurtosis
        }
        
        return next_token, metrics, logits


# ==========================================================
# 2) 因果ダイナミクス：ISAExecutor + CausalCore
# ==========================================================
class ISAExecutor:
    """
    Atomic Intervention (ISA) Graph Logic
    """
    @staticmethod
    def execute(S, r_scale, phi_shift, omega_eff, sequence, label_map):
        for op_data in sequence:
            if not isinstance(op_data, dict): continue
            op = op_data.get("op")
            args = op_data.get("args", [])
            
            if op == "DELETE":
                src_l, dst_l = args
                for i in label_map.get(src_l, []):
                    for j in label_map.get(dst_l, []):
                        S[i, j] = 0.0
            elif op == "INSERT":
                src_l, dst_l = args
                for i in label_map.get(src_l, []):
                    for j in label_map.get(dst_l, []):
                        S[i, j] = 1.0
            elif op == "SCALE_GAIN":
                label, alpha = args
                for i in label_map.get(label, []):
                    r_scale[:, i] *= float(alpha)
            elif op == "DELAY":
                label, dt = args
                for i in label_map.get(label, []):
                    phi_shift[:, i] += float(dt)
            elif op == "PERMUTE":
                p = torch.randperm(S.shape[0])
                S.copy_(S[p][:, p] * 1.5)
        return S, r_scale, phi_shift, omega_eff

class CausalCore(nn.Module):
    def __init__(self, n_vars=10, d_model=128, K=2):
        super().__init__()
        self.n_vars = n_vars
        self.K = K

        # S行列（因果構造）
        self.raw_S = nn.Parameter(torch.randn(n_vars, n_vars) * 0.5 + 0.5)

        # エッジ位相
        self.raw_phase = nn.Parameter(torch.randn(n_vars, n_vars) * 0.1)

        # モード生成器（attention-like）
        self.mode_gen = nn.Sequential(
            nn.Linear(n_vars, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Linear(d_model, K * 2)
        )

        self.register_buffer("phi_base", torch.tensor([0.0, np.pi / 2]))
        # 角周波数（時制・周波数）
        self.register_buffer("omega", torch.tensor(0.1))

    def _get_S_core(self, epoch):
        S = torch.tanh(self.raw_S)
        tau = max(0, (epoch - 1500) / 5000) if epoch > 1500 else 0
        mask = (torch.abs(S) > tau).float()
        S = S * mask
        diag = torch.eye(self.n_vars, device=S.device)
        return S * (1 - diag) + diag * 0.95

    def forward(self, x_2ch, history_flat, epoch=2500, do_mask=None, atomic_interventions=None, label_map=None, t=0):
        """
        x_{j}(t+1) = sigma( sum_i S_ij * r_ij * e^{i(phi_ij + omega*t)} * x_i(t) )
        """
        B = x_2ch.shape[0]
        params = self.mode_gen(history_flat).view(B, self.K, 2)

        phi_curr = (
            self.phi_base.unsqueeze(0)
            + torch.tanh(params[:, :, 0]) * (np.pi / 12)
        )

        r_mode = torch.stack([
            1.5 * torch.sigmoid(params[:, 0, 1]),
            0.8 * torch.sigmoid(params[:, 1, 1])
        ], dim=1)

        S = self._get_S_core(epoch)
        
        # --- 原子操作 (ISA) の適用 ---
        r_scale = torch.ones_like(S).unsqueeze(0).repeat(B, 1, 1)
        phi_shift = torch.zeros_like(S).unsqueeze(0).repeat(B, 1, 1)
        omega_eff = self.omega.clone()
        
        if atomic_interventions and label_map:
            S, r_scale, phi_shift, omega_eff = ISAExecutor.execute(
                S, r_scale, phi_shift, omega_eff, atomic_interventions, label_map
            )

        self_loop_mask = torch.eye(self.n_vars, device=S.device)

        x_real = x_2ch[:, :, 0].unsqueeze(1)
        x_imag = x_2ch[:, :, 1].unsqueeze(1)

        out_real, out_imag = 0, 0

        for k in range(self.K):
            w_k = (1.0 - self_loop_mask) if k == 0 else self_loop_mask
            theta = (self.raw_phase.unsqueeze(0) 
                     + phi_curr[:, k].view(B, 1, 1) 
                     + phi_shift
                     + omega_eff * t)
            
            A = S.unsqueeze(0) * w_k * r_mode[:, k].view(B, 1, 1) * r_scale

            if do_mask is not None:
                A = A * do_mask.unsqueeze(2)

            out_real += torch.sum(
                A * (torch.cos(theta)*x_real - torch.sin(theta)*x_imag), dim=2
            )
            out_imag += torch.sum(
                A * (torch.sin(theta)*x_real + torch.cos(theta)*x_imag), dim=2
            )

        x_next = torch.stack([torch.tanh(out_real), torch.tanh(out_imag)], dim=-1)
        return x_next, phi_curr


# ==========================================================
# 3) 評価層：CSI, CII, CII'
# ==========================================================
class CausalMetrics:
    @staticmethod
    def compute_CSI(phi_traj):
        phi_mean = torch.mean(phi_traj, dim=1, keepdim=True)
        var = torch.mean((phi_traj - phi_mean) ** 2, dim=1)
        return torch.mean(var).item()

    @staticmethod
    def compute_CII(phi_traj):
        d2 = phi_traj[2:] - 2 * phi_traj[1:-1] + phi_traj[:-2]
        return torch.mean(d2**2).item()

    @staticmethod
    def compute_CII_prime(phi_edge, phi_node, alpha=0.5):
        d2_edge = phi_edge[2:] - 2*phi_edge[1:-1] + phi_edge[:-2]
        d2_node = phi_node[2:] - 2*phi_node[1:-1] + phi_node[:-2]

        return alpha * torch.mean(torch.abs(d2_edge)).item() + \
               (1-alpha) * torch.mean(torch.abs(d2_node)).item()

    @staticmethod
    def compute_responsiveness(phi_traj):
        """
        介入後のノードが『動いているか』を測る。
        変化（速度）がないものは、反実仮想の帰結として不適切。
        """
        d_phi = phi_traj[1:] - phi_traj[:-1]
        return torch.mean(torch.abs(d_phi)).item()

    @staticmethod
    def compute_alignment(intervention_isa, option_isa):
        """
        介入の意図(DELETE/NEGATION)とオプションのISAの論理的整合性を簡易チェック
        """
        def get_ops(isa):
            ops = []
            for item in isa:
                if isinstance(item, dict) and 'op' in item:
                    ops.append(item['op'])
            return ops

        int_ops = get_ops(intervention_isa)
        opt_ops = get_ops(option_isa)
        
        # 介入が削除系(DELETE)で、オプションが挿入系(INSERT)なら、補償行動として高評価
        if ("DELETE" in int_ops or "SCALE_GAIN" in int_ops) and "INSERT" in opt_ops:
            return 2.0 
        # 介入が不可能事象で、オプションがPERMUTEなら高評価
        if "PERMUTE" in int_ops and "PERMUTE" in opt_ops:
            return 3.0
        return 1.0


# ==========================================================
# 4) S行列エンジン：事実の剛性誘導
# ==========================================================
class SMatrixEngine:
    def __init__(self):
        self.matrix = defaultdict(lambda: defaultdict(float))

    def register_sequence(self, token_ids, rigidity=50.0):
        for i in range(len(token_ids) - 1):
            curr = token_ids[i]
            nxt = token_ids[i+1]
            self.matrix[curr][nxt] = max(self.matrix[curr][nxt], rigidity)

    def adjust_logits(self, last_token_id, logits, lambda_val=2.0):
        if last_token_id in self.matrix:
            for nxt_id, rig in self.matrix[last_token_id].items():
                logits[..., nxt_id] += lambda_val * rig
        return logits


# 6) 統合OS：Unified CausalOS
# ==========================================================
class UnifiedCausalOS:
    def __init__(self, n_vars=5):
        self.observer = LLMCausalObserver()
        self.causal_core = CausalCore(n_vars=n_vars).to(device)
        self.metrics = CausalMetrics()
        self.s_matrix = SMatrixEngine()
        self.node_extractor = CausalNodeExtractor(
            model=self.observer.model, 
            tokenizer=self.observer.tokenizer
        )
        self.causal_core.eval()
        self.label_cache = {}

    def reset(self):
        self.observer.reset()

    def get_semantic_stability(self, text, steps=5):
        self.observer.reset()
        input_ids = self.observer.tokenizer(text, return_tensors="pt").to(device).input_ids
        ents, ciis = [], []
        
        next_tok, metrics, _ = self.observer.step(input_ids)
        ents.append(metrics['entropy'])
        ciis.append(metrics['cii'])
        
        curr_ids = input_ids
        for _ in range(steps):
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_tok]]).to(device)], dim=-1)
            next_tok, metrics, _ = self.observer.step(curr_ids)
            ents.append(metrics['entropy'])
            ciis.append(metrics['cii'])
        return np.mean(ents), np.mean(ciis)

    def detect_intervention_from_llm(self, prompt, horizon=30):
        self.observer.reset()
        input_ids = self.observer.tokenizer(prompt, return_tensors="pt").to(device).input_ids
        phi_traj, cii_traj = [], []

        for t in range(horizon):
            next_tok, metrics, _ = self.observer.step(input_ids)
            phi_traj.append(metrics["phi"])
            cii_traj.append(metrics["cii"])
            input_ids = torch.cat([input_ids, torch.tensor([[next_tok]]).to(device)], dim=-1)

        return int(np.argmax(np.abs(cii_traj)))

    def rollout_with_intervention(self, initial_energy=0.5, atomic_interventions=None, label_map=None, horizon=10):
        x = torch.zeros(1, self.causal_core.n_vars, 2, device=device)
        x[0, :, 0] = initial_energy 
        hist = torch.zeros(1, self.causal_core.n_vars, device=device)
        phi_history = []

        for t in range(horizon):
            with torch.no_grad():
                x, phi = self.causal_core(x, hist, atomic_interventions=atomic_interventions, label_map=label_map, t=t)
            phi_history.append(phi.mean())
            hist = x[:, :, 0]
        return torch.stack(phi_history)

    def generate_intervention_isa(self, factual, counterfactual, label_graph):
        """
        Generate ISA for the counterfactual intervention part.
        Uses 2-stage reasoning for universality.
        """
        labels = [l['name'] for l in label_graph.get("labels", [])]
        if not labels:
            labels = ["something"]

        # ============================
        # Stage 1: Intervention logic explanation
        # ============================
        explain_prompt = f"""
Factual world: {factual}
Counterfactual condition: {counterfactual}

Explain how the counterfactual condition breaks or modifies the factual causal chain.
Identify which entity or action is suppressed, replaced, or added.
Answer in 1-2 sentences.
"""
        explanation = self.generate_short(explain_prompt, max_new_tokens=80)

        # ============================
        # Stage 2: ISA construction
        # ============================
        isa_prompt = f"""
Causal logic: {explanation}
Available labels: {labels}

Task: Translate this intervention into ISA operations.
Ops: DELETE(src, dst), INSERT(src, dst), SCALE_GAIN(label, alpha), DELAY(label, dt), PERMUTE()

Return ONLY a JSON list.
JSON:
"""
        raw = self.generate_short(isa_prompt, max_new_tokens=150)

        try:
            match = re.search(r'(\[.*\])', raw, re.DOTALL)
            if match:
                isa = json.loads(match.group(1))
                if isinstance(isa, list) and len(isa) > 0:
                    return isa
        except Exception:
            pass

        # Deterministic fallback
        t = counterfactual.lower()
        if " not " in t or "n't " in t or "didn't" in t:
            return [{"op": "SCALE_GAIN", "args": [labels[0], 0.0]}]
        
        return [{"op": "SCALE_GAIN", "args": [labels[-1], 0.5]}]

    def generate_option_isa(self, factual, counterfactual, option_text, label_graph):
        """
        Generate ISA (atomic intervention sequence) for EACH option.
        Always returns a non-empty ISA list.
        """

        labels = [l['name'] for l in label_graph.get("labels", [])]
        if not labels:
            labels = ["something"]

        # ============================
        # Stage 1: causal explanation
        # ============================
        explain_prompt = f"""
Factual world:
{factual}

Counterfactual condition:
{counterfactual}

Candidate outcome:
{option_text}

Explain the CAUSAL reasoning that connects the counterfactual condition
to the candidate outcome.

Focus on:
- What FAILED or changed?
- Does this outcome introduce a NEW action or event?
- Is this action a compensation or follow-up?
- Is the effect stronger, weaker, delayed, or impossible?

Answer in 1–3 sentences.
"""
        explanation = self.generate_short(explain_prompt, max_new_tokens=120)

        # ============================
        # Stage 2: ISA construction
        # ============================
        isa_prompt = f"""
Causal explanation:
{explanation}

Available labels (existing entities or actions):
{labels}

Task:
Translate the causal consequence into ATOMIC OPERATIONS.

Atomic operations:
- DELETE(src, dst): remove causal influence
- INSERT(src, dst): introduce a NEW action or event
- SCALE_GAIN(label, alpha): change intensity (0.0 = suppressed, >1.0 = stronger)
- DELAY(label, dt): temporal delay
- PERMUTE(): logical or category impossibility

Rules:
- If a NEW action/event appears, you MUST use INSERT.
- Compensation after failure should INCREASE intensity.
- If outcome states "nothing happens", suppress influence.
- If outcome is impossible or nonsensical, use PERMUTE.
- Return ONLY a JSON list.

JSON:
"""
        raw = self.generate_short(isa_prompt, max_new_tokens=200)

        # ============================
        # JSON extraction
        # ============================
        try:
            match = re.search(r'(\[.*\])', raw, re.DOTALL)
            if match:
                isa = json.loads(match.group(1))
                if isinstance(isa, list) and len(isa) > 0:
                    return isa
        except Exception:
            pass

        # ============================
        # Deterministic fallback (ALWAYS returns ISA)
        # ============================
        t = option_text.lower()

        # --- Impossibility ---
        if any(w in t for w in ["not possible", "impossible", "nonsense", "cannot", "category error"]):
            return [{"op": "PERMUTE", "args": []}]

        # --- Negation / nothing happens ---
        if any(w in t for w in ["nothing", "no effect", "did not happen", "never"]):
            return [{"op": "SCALE_GAIN", "args": [labels[0], 0.0]}]

        # --- Compensation / follow-up action ---
        if any(w in t for w in ["need to", "had to", "would have to", "study", "prepare", "practice"]):
            # introduce a new action node implicitly
            return [
                {"op": "INSERT", "args": [labels[0], "compensatory_action"]},
                {"op": "SCALE_GAIN", "args": ["compensatory_action", 1.5]}
            ]

        # --- Intensity change ---
        if any(w in t for w in ["more", "stronger", "harder", "faster"]):
            return [{"op": "SCALE_GAIN", "args": [labels[0], 1.5]}]

        if any(w in t for w in ["less", "weaker", "slower"]):
            return [{"op": "SCALE_GAIN", "args": [labels[0], 0.5]}]

        # --- Default: weak but valid continuation ---
        return [{"op": "SCALE_GAIN", "args": [labels[0], 1.0]}]

    def classify_phenomenology(self, factual, cf):
        t = cf.lower()
        # 強力なキーワードベースの分類（LLMの不安定さを補う）
        if any(w in t for w in [" not ", "n't ", " never ", " failed to "]):
            return "NEGATION"
        if any(w in t for w in [" more ", " less ", " faster ", " slower ", " heavier ", " weaker "]):
            return "INTENSITY"
        if any(w in t for w in [" earlier ", " before ", " later ", " after "]):
            return "TENSE"
        
        prompt = f"""Classify the type of counterfactual intervention.
Factual: {factual}
Counterfactual: {cf}
Categories: SUBSTITUTION, NEGATION, INTENSITY, TENSE, IMPOSSIBILITY
Result:"""
        category = self.generate_short(prompt, max_new_tokens=5).strip().upper()
        for cat in ["SUBSTITUTION", "NEGATION", "INTENSITY", "TENSE", "IMPOSSIBILITY"]:
            if cat in category: return cat
        return "SUBSTITUTION"

    def generate_short(self, prompt, max_new_tokens=10):
        inputs = self.observer.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.observer.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=self.observer.tokenizer.eos_token_id
            )
        return self.observer.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    def generate_with_causal_check(self, prompt, max_new_tokens=20, hesitation_threshold=1.0):
        input_ids = self.observer.tokenizer(prompt, return_tensors="pt").to(device).input_ids
        generated_ids = input_ids[0].tolist()
        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([generated_ids]).to(device)
            next_token, metrics, logits = self.observer.step(input_tensor)
            if metrics["entropy"] > 2.0 or abs(metrics["cii"]) > hesitation_threshold:
                logits = self.s_matrix.adjust_logits(generated_ids[-1], logits)
                next_token = torch.argmax(logits).item()
            generated_ids.append(next_token)
            if next_token == self.observer.tokenizer.eos_token_id: break
        return self.observer.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def solve_counterfactual(self, factual, cf, options=None):
        print("\n=== Unified CausalOS (ISA Graph Mode) ===")
        
        # 1. Label Extraction (with Caching)
        if factual in self.label_cache:
            label_graph = self.label_cache[factual]
        else:
            label_graph = self.node_extractor.extract_label_graph(factual)
            self.label_cache[factual] = label_graph

        if not label_graph.get("labels"):
            label_graph = {"labels": [{"name": "something", "nodes": 2}], "edges": []}
            
        # 2. Map Labels to Node Indices and Ground Topology
        label_map = {}
        curr_idx = 0
        for label in label_graph["labels"]:
            n = label.get("nodes", 2)
            label_map[label["name"]] = list(range(curr_idx, min(self.causal_core.n_vars, curr_idx + n)))
            curr_idx += n
        
        # Reserve remaining nodes for virtual labels like "compensatory_action"
        if curr_idx < self.causal_core.n_vars:
            label_map["compensatory_action"] = list(range(curr_idx, self.causal_core.n_vars))
            
        # Grounding: Initialize topology based on extracted edges
        with torch.no_grad():
            self.causal_core.raw_S.zero_()
            for edge in label_graph.get("edges", []):
                src_nodes = label_map.get(edge["src"], [])
                dst_nodes = label_map.get(edge["dst"], [])
                for s in src_nodes:
                    for d in dst_nodes:
                        if s < self.causal_core.n_vars and d < self.causal_core.n_vars:
                            self.causal_core.raw_S[s, d] = 1.0

        # 3. Generate Intervention ISA
        intervention_isa = self.generate_intervention_isa(factual, cf, label_graph)
        if not intervention_isa:
            print("[Error] Could not represent intervention as ISA combination.")
            return "B"
            
        # 4. Evaluation
        if not options: return "B"
        
        scores = {}
        for key, text in options.items():
            option_isa = self.generate_option_isa(factual, cf, text, label_graph)
            combined_isa = intervention_isa + option_isa
            
            # Simulate
            self.observer.reset()
            phi_traj = self.rollout_with_intervention(
                initial_energy=0.5, 
                atomic_interventions=combined_isa, 
                label_map=label_map
            )
            
            # --- 新しいスコアリングロジック ---
            # 1. 反応性: 介入によってどれだけ「意味のある変化」が起きたか
            resp = self.metrics.compute_responsiveness(phi_traj)
            
            # 2. 論理整合性: 介入とオプションのISAの組み合わせ評価
            align = self.metrics.compute_alignment(intervention_isa, option_isa)
            
            # 3. 安定性 (CSI): 依然として必要だが、これだけで決めない
            csi = self.metrics.compute_CSI(phi_traj.unsqueeze(0))
            stability = 1.0 / (csi + 1e-4) # ゼロ除算回避

            # 合計スコア: 「変化の大きさ」と「論理の正しさ」を重視
            score = resp * align * (stability ** 0.5)
            scores[key] = score
            print(f"Option {key}: Resp={resp:.4f}, Align={align}, Score={score:.2f}")
            
        return max(scores, key=scores.get)


# ==========================================================
# DEMO
# ==========================================================
if __name__ == "__main__":
    osys = UnifiedCausalOS()
    answer = osys.solve_counterfactual(
        "A man walks on a street.",
        "What would have happened if a man had walked on a bed?"
    )
    print(f"\nFinal Answer: <Answer>{answer}</Answer>")
