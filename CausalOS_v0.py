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
# 2) 因果ダイナミクス：CausalCore（複素位相＋attention-like伝播）
# ==========================================================
class CausalCore(nn.Module):
    def __init__(self, n_vars=5, d_model=128, K=2):
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

    def forward(self, x_2ch, history_flat, epoch=2500, do_mask=None, atomic_interventions=None, t=0):
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

        # --- 原子操作 (Atomic Interventions) の適用 ---
        r_scale = torch.ones_like(S)
        phi_shift = torch.zeros_like(S)
        omega_eff = self.omega.clone()

        if atomic_interventions:
            # 1) Delete/Insert (Structure)
            if 'delete' in atomic_interventions:
                for i, j in atomic_interventions['delete']:
                    S[i, j] = 0.0
            if 'insert' in atomic_interventions:
                for i, j in atomic_interventions['insert']:
                    S[i, j] = 1.0

            # 2) Scale (Intensity)
            if 'scale' in atomic_interventions:
                for i, j, alpha in atomic_interventions['scale']:
                    r_scale[i, j] *= alpha

            # 3) Invert (Polarity)
            if 'invert' in atomic_interventions:
                for i, j in atomic_interventions['invert']:
                    phi_shift[i, j] += np.pi

            # 4) Delay (Tense/Frequency)
            if 'delay_omega' in atomic_interventions:
                omega_eff += atomic_interventions['delay_omega']

            # 5) Permute (Logical Structure / Non-Hermitian)
            if 'permute' in atomic_interventions:
                for i, j in atomic_interventions['permute']:
                    tmp = S[i, j].clone()
                    S[i, j] = S[j, i]
                    S[j, i] = tmp * 2.0 # 発散を誘発

        self_loop_mask = torch.eye(self.n_vars, device=S.device)

        x_real = x_2ch[:, :, 0].unsqueeze(1)
        x_imag = x_2ch[:, :, 1].unsqueeze(1)

        out_real, out_imag = 0, 0

        for k in range(self.K):
            w_k = (1.0 - self_loop_mask) if k == 0 else self_loop_mask
            theta = (self.raw_phase.unsqueeze(0)
                     + phi_curr[:, k].view(B, 1, 1)
                     + phi_shift.unsqueeze(0)
                     + omega_eff * t)

            A = S.unsqueeze(0) * w_k * r_mode[:, k].view(B, 1, 1) * r_scale.unsqueeze(0)

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


# ==========================================================
# 5) 追加ユーティリティ：Causal Primitive Encoder
# ==========================================================
class CausalPrimitiveEncoder:
    """
    Convert a natural language outcome into a causal primitive vector
    """
    @staticmethod
    def encode(text: str):
        t = text.lower()
        prim = {
            "substitution": 0.0,
            "negation": 0.0,
            "intensity": 0.0,
            "tense": 0.0,
            "impossibility": 0.0
        }
        if any(w in t for w in ["instead", "replace", "changed into"]):
            prim["substitution"] = 1.0
        if any(w in t for w in ["not", "never", "didn't", "failed to"]):
            prim["negation"] = 1.0
        if any(w in t for w in ["more", "faster", "heavier", "stronger"]):
            prim["intensity"] = 1.0
        elif any(w in t for w in ["less", "slower", "weaker"]):
            prim["intensity"] = -1.0
        if any(w in t for w in ["earlier", "before"]):
            prim["tense"] = -1.0
        elif any(w in t for w in ["later", "after"]):
            prim["tense"] = 1.0
        if any(w in t for w in ["impossible", "cannot", "category error", "nonsense", "not possible", "不可能"]):
            prim["impossibility"] = 1.0
        return prim

def expected_primitives(phenom, csi, cii, impossible):
    p = {
        "substitution": 0.0,
        "negation": 0.0,
        "intensity": 0.0,
        "tense": 0.0,
        "impossibility": 0.0
    }
    if phenom == "SUBSTITUTION":
        p["substitution"] = 1.0
    elif phenom == "NEGATION":
        p["negation"] = 1.0
    elif phenom == "INTENSITY":
        p["intensity"] = 1.0
    elif phenom == "TENSE":
        p["tense"] = 1.0
    elif phenom == "IMPOSSIBILITY":
        p["impossibility"] = 1.0

    if impossible or cii > 1e3:
        p["impossibility"] = 1.0
    if csi < 0.01:
        p["negation"] *= 0.5
    return p

def primitive_distance(p1, p2):
    v1 = np.array(list(p1.values()), dtype=np.float32)
    v2 = np.array(list(p2.values()), dtype=np.float32)
    return np.linalg.norm(v1 - v2)


# ==========================================================
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

        self.mapping = {
            "man": 0,
            "walk": 1,
            "street": 2,
            "bed": 3,
            "destination": 4
        }

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

    def rollout_with_intervention(self, initial_energy=0.5, atomic_interventions=None, horizon=20):
        x = torch.zeros(1, self.causal_core.n_vars, 2, device=device)
        x[0, :, 0] = initial_energy
        hist = torch.zeros(1, self.causal_core.n_vars, device=device)
        phi_history = []

        for t in range(horizon):
            with torch.no_grad():
                x, phi = self.causal_core(x, hist, atomic_interventions=atomic_interventions, t=t)
            phi_history.append(phi.mean())
            hist = x[:, :, 0]
        return torch.stack(phi_history)

    def classify_phenomenology(self, factual, cf):
        prompt = f"""Classify the type of counterfactual intervention between these two sentences.
Factual: {factual}
Counterfactual: {cf}

Categories:
1. SUBSTITUTION: A is replaced by B
2. NEGATION: A is negated (e.g., 'not')
3. INTENSITY: Degree or speed changes (e.g., 'more', 'fast')
4. TENSE: Time or order changes (e.g., 'earlier', 'after')
5. IMPOSSIBILITY: Logic error or paradox

Return only the category name.
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
        print("\n=== Unified CausalOS ===")
        phenom = self.classify_phenomenology(factual, cf)
        print(f"[CausalOS] Phenomenology: {phenom}")

        nodes_f = self.node_extractor.extract_nodes(factual)
        nodes_c = self.node_extractor.extract_nodes(cf)
        self.mapping = {node: i for i, node in enumerate(nodes_f[:self.causal_core.n_vars])}

        removed = [n for n in nodes_f if n.lower() not in [nc.lower() for nc in nodes_c]]
        orig_node = removed[0] if removed else nodes_f[-1]
        do_idx = self.mapping.get(orig_node, 0)

        atomic_int = {}
        if phenom == "SUBSTITUTION":
            atomic_int["delete"] = [(i, do_idx) for i in range(self.causal_core.n_vars)]
        elif phenom == "NEGATION":
            atomic_int["invert"] = [(i, do_idx) for i in range(self.causal_core.n_vars)]
        elif phenom == "INTENSITY":
            atomic_int["scale"] = [(i, do_idx, 2.0) for i in range(self.causal_core.n_vars)]
        elif phenom == "TENSE":
            atomic_int["delay_omega"] = 0.5
        elif phenom == "IMPOSSIBILITY":
            atomic_int["permute"] = [(do_idx, (do_idx + 1) % self.causal_core.n_vars)]

        self.observer.reset()
        _, f_metrics, _ = self.observer.step(self.observer.tokenizer(factual, return_tensors="pt").to(device).input_ids)
        initial_energy = min(1.0, f_metrics["phi"] / 50.0)
        phi_traj = self.rollout_with_intervention(initial_energy=initial_energy, atomic_interventions=atomic_int)
        csi = self.metrics.compute_CSI(phi_traj.unsqueeze(0))
        cii = self.metrics.compute_CII(phi_traj)
        impossible = torch.isnan(phi_traj).any() or cii > 1e4 or phenom == "IMPOSSIBILITY"
        print(f"Causal Rollout: CSI={csi:.4e}, CII={cii:.4e}, Impossible={impossible}")

        if not options: return "B"

        P_expected = expected_primitives(phenom, csi, cii, impossible)
        scores = {}
        for key, outcome in options.items():
            P_opt = CausalPrimitiveEncoder.encode(outcome)
            dist = primitive_distance(P_opt, P_expected)
            scores[key] = -dist
            print(f"Option {key}: primitives={P_opt}, score={-dist:.4f}")

        return max(scores.items(), key=lambda x: x[1])[0]


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
