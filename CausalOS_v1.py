import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re, os, json
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
        device = torch.device("cuda")
        print(f"[System] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[System] Using CPU")
except:
    device = torch.device("cpu")
    print("[System] CUDA error, using CPU")

# ==========================================================
# 1) 物理コア: 複素位相ダイナミクス
# ==========================================================
class CausalCoreV2(nn.Module):
    def __init__(self, n_nodes=10, dim=2):
        super().__init__()
        self.n_nodes = n_nodes
        self.dim = dim
        
        # 状態: [n_nodes, 2] (real, imaginary)
        self.x = nn.Parameter(torch.randn(n_nodes, dim, device=device) * 0.3, requires_grad=True)
        
        # S行列: 因果構造
        self.raw_S = nn.Parameter(torch.randn(n_nodes, n_nodes, device=device) * 0.2)
        
        # 位相: エッジごとの時間遅延
        self.raw_phase = nn.Parameter(torch.randn(n_nodes, n_nodes, device=device) * 0.1)
        
        self.register_buffer("omega", torch.tensor(0.15))

    def set_node_state(self, idx, rigidity, support, harm):
        """物理属性からノード初期状態を設定"""
        if idx >= self.n_nodes:
            return
        
        with torch.no_grad():
            # より大きな差を作るため、非線形変換を使用
            # harm: 0-1 → -2 to 2 の範囲に拡張
            real_val = (harm - 0.5) * 4.0 + rigidity * 2.0
            imag_val = (support - 0.5) * 4.0
            
            self.x.data[idx, 0] = real_val
            self.x.data[idx, 1] = imag_val

    def forward(self, x_in=None, gate=None, gain=None, t=0):
        """1ステップの時間発展"""
        S = torch.tanh(self.raw_S)
        G = gate if gate is not None else torch.ones_like(S)
        A = gain if gain is not None else torch.ones_like(S)
        S_eff = S * G * A
        
        theta = self.raw_phase + self.omega * t
        
        x = x_in if x_in is not None else self.x
        
        # 複素数演算: (S * e^{iθ}) @ x
        x_real = x[:, 0].unsqueeze(-1)
        x_imag = x[:, 1].unsqueeze(-1)
        
        out_real = torch.matmul(S_eff * torch.cos(theta), x_real) - \
                   torch.matmul(S_eff * torch.sin(theta), x_imag)
        out_imag = torch.matmul(S_eff * torch.sin(theta), x_real) + \
                   torch.matmul(S_eff * torch.cos(theta), x_imag)
        
        # tanh活性化（制限された範囲内で振る舞う）
        next_x = torch.stack([
            torch.tanh(out_real.squeeze(-1)), 
            torch.tanh(out_imag.squeeze(-1))
        ], dim=-1)
        
        return next_x

# ==========================================================
# 2) 意味接地層
# ==========================================================
class SemanticGrounding:
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        self.cache = {}
    
    def ground_concept(self, word):
        """概念→物理属性"""
        if word in self.cache:
            return self.cache[word]
        
        prompt = f"""Physical properties of "{word}":

Rigidity (0.0=very soft, 1.0=very hard): <rigidity>VALUE</rigidity>
Support (0.0=cannot support, 1.0=can support): <support>VALUE</support>
Harmfulness (0.0=safe, 1.0=very harmful): <harm>VALUE</harm>

Provide numerical values only:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=False, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )
        
        # デフォルト
        props = {"rigidity": 0.5, "support": 0.5, "harm": 0.0}
        
        # 抽出
        for key, pattern in [
            ("rigidity", r"<rigidity>([\d.]+)</rigidity>"),
            ("support", r"<support>([\d.]+)</support>"),
            ("harm", r"<harm>([\d.]+)</harm>")
        ]:
            match = re.search(pattern, response)
            if match:
                try:
                    props[key] = float(match.group(1))
                except:
                    pass
        
        self.cache[word] = props
        return props

# ==========================================================
# 3) 因果記憶（学習可能）
# ==========================================================
class CausalMemory:
    def __init__(self, decay_rate=0.02):
        # (cause, effect) -> {strength, confidence, access_count, last_step}
        self.traces = {}
        self.decay_rate = decay_rate
        self.step = 0
    
    def learn(self, cause, effect, impact_score):
        """因果関係を学習"""
        key = (cause, effect)
        self.step += 1
        
        if key not in self.traces:
            self.traces[key] = {
                "strength": impact_score,
                "confidence": 0.5,
                "access_count": 1,
                "last_step": self.step
            }
        else:
            mem = self.traces[key]
            # 移動平均で強化
            mem["strength"] = mem["strength"] * 0.8 + impact_score * 0.2
            mem["confidence"] = min(1.0, mem["confidence"] + 0.05)
            mem["access_count"] += 1
            mem["last_step"] = self.step
        
        print(f"[Memory] {cause} -> {effect}: strength={self.traces[key]['strength']:.2f}")
    
    def retrieve(self, cause, threshold=0.1):
        """因果関係を想起"""
        self.step += 1
        results = []
        
        for (c, e), mem in list(self.traces.items()):
            if c == cause:
                # 時間減衰
                time_diff = self.step - mem["last_step"]
                current_strength = mem["strength"] * mem["confidence"] * \
                                   ((1.0 - self.decay_rate) ** time_diff)
                
                if current_strength >= threshold:
                    results.append((e, current_strength))
                    mem["last_step"] = self.step
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results

# ==========================================================
# 4) 統合OS
# ==========================================================
class UnifiedCausalOS:
    def __init__(self, n_nodes=10):
        self.n_nodes = n_nodes
        
        self.core = CausalCoreV2(n_nodes=n_nodes)
        self.memory = CausalMemory()
        
        model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        
        self.grounding = SemanticGrounding(self.model, self.tokenizer)
        self.label_to_idx = {}

    def _normalize_word(self, word):
        """語幹化"""
        word = word.lower()
        if word.endswith('ed') and len(word) > 3:
            return word[:-2] if word.endswith('ked') else word[:-1]
        elif word.endswith('s') and len(word) > 3:
            return word[:-1]
        return word

    def _extract_nodes(self, text):
        """ノード抽出"""
        has_not = " not " in text.lower()
        
        prompt = f"""Extract nouns, verbs, and "not" from: "{text}"
<entities>word1, word2, word3</entities>"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=60, 
                do_sample=False, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        result = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )
        
        # 抽出
        matches = re.findall(r"<entities>(.*?)</entities>", result)
        if matches:
            nodes = [n.strip().lower() for n in matches[-1].split(',') if n.strip()]
            nodes = [n for n in nodes if len(n) > 1 and n not in ['a', 'an', 'the']]
        else:
            words = text.lower().replace('.', '').replace('?', '').split()
            nodes = [w for w in words if w not in ['a', 'an', 'the', 'on', 'if', 'would', 'have', 'had']]
        
        # "not"を強制追加
        if has_not and "not" not in nodes:
            nodes.insert(0, "not")
        
        # 正規化
        normalized = []
        for n in nodes:
            if n == "not":
                normalized.append(n)
            else:
                normalized.append(self._normalize_word(n))
        
        return normalized if normalized else ["unknown"]

    def _detect_intervention(self, factual_nodes, cf_nodes):
        """介入検出"""
        removed = [n for n in factual_nodes if n not in cf_nodes]
        added = [n for n in cf_nodes if n not in factual_nodes]
        
        # 否定
        if "not" in added:
            idx = cf_nodes.index("not")
            if idx + 1 < len(cf_nodes):
                return cf_nodes[idx + 1], "not_" + cf_nodes[idx + 1]
        
        # 役割逆転
        if (len(factual_nodes) == len(cf_nodes) and 
            set(factual_nodes) == set(cf_nodes) and
            len(factual_nodes) >= 3):
            if (factual_nodes[0] == cf_nodes[-1] and 
                factual_nodes[-1] == cf_nodes[0]):
                return "role_reversal", factual_nodes[0] + "_and_" + factual_nodes[-1]
        
        # 通常の差分
        verb_like = ['walk', 'eat', 'understand', 'grow']
        removed_nouns = [r for r in removed if r not in verb_like]
        added_nouns = [a for a in added if a not in verb_like]
        
        if removed_nouns and added_nouns:
            return removed_nouns[0], added_nouns[0]
        elif removed and added:
            return removed[0], added[0]
        
        return None, None

    def _create_intervention(self, original_idx, replacement_idx, harm_delta, support_delta=0.0):
        """介入→gate/gain"""
        gate = torch.ones(self.n_nodes, self.n_nodes, device=device)
        gain = torch.ones(self.n_nodes, self.n_nodes, device=device)
        
        if original_idx < self.n_nodes:
            gate[:, original_idx] = 0.0
            gate[original_idx, :] = 0.0
        
        # harm変化とsupport変化(質量差)を増幅
        # 質量差が大きい場合もインパクト大
        impact_factor = abs(harm_delta) * 15.0 + abs(support_delta) * 5.0
        gain *= (1.0 + impact_factor)
        
        return gate, gain

    def _simulate(self, gate, gain, steps=25):
        """物理シミュレーション"""
        traj = []
        x = self.core.x.clone().detach()
        
        with torch.no_grad():
            for t in range(steps):
                x = self.core.forward(x, gate=gate, gain=gain, t=t)
                traj.append(x.clone())
        
        return torch.stack(traj)

    def _compute_impact(self, traj_base, traj_cf):
        """介入の影響量"""
        with torch.no_grad():
            diff = traj_cf - traj_base
            magnitude = torch.norm(diff, dim=-1).mean().item()
            return magnitude

    def _extract_severity(self, text):
        """テキスト→重大度(0-1)"""
        prompt = f"""Rate severity of: "{text}"
0.0 = harmless
0.5 = moderate concern
1.0 = catastrophic
<severity>X.X</severity>"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=False, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )
        
        # 頑健な抽出ロジック
        # 1. XMLタグ <severity>...</severity>
        match = re.search(r"<severity>\s*([\d.]+)\s*</severity>", response)
        if match:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
            
        # 2. 数字のみの回答 (例: "0.8")
        match = re.search(r"^\s*([\d.]+)\s*$", response)
        if match:
            try:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
            except:
                pass
                
        # 3. テキスト中の数値を検索 (0.0 - 1.0 の範囲のもの)
        numbers = re.findall(r"([\d.]+)", response)
        for num in numbers:
            try:
                val = float(num)
                if 0.0 <= val <= 1.0:
                    return val
            except:
                continue
        
        print(f"[Warning] Could not extract severity from: '{response}'. Defaulting to 0.5")
        return 0.5

    def solve_counterfactual(self, factual, cf, options=None):
        """反事実推論（AGI指向・汎用版）"""
        print(f"\n{'='*60}")
        print(f"[OS] Counterfactual Reasoning")
        print(f"Factual: {factual}")
        print(f"CF: {cf}")
        
        # ノード抽出
        factual_nodes = self._extract_nodes(factual)
        cf_nodes = self._extract_nodes(cf)
        
        print(f"Nodes (F): {factual_nodes}")
        print(f"Nodes (CF): {cf_nodes}")
        
        # ラベルマップ更新
        all_nodes = list(set(factual_nodes + cf_nodes))
        self.label_to_idx = {node: i for i, node in enumerate(all_nodes[:self.n_nodes])}
        print(f"Label Map: {self.label_to_idx}")
        
        # 介入検出
        original, replacement = self._detect_intervention(factual_nodes, cf_nodes)
        print(f"Intervention: {original} -> {replacement}")
        
        if not original or not replacement:
            return list(options.keys())[0] if options else "B"
        
        # 役割逆転
        if original == "role_reversal":
            for key, text in (options or {}).items():
                if "not possible" in text.lower():
                    return key
            return "A"
        
        # 否定
        if replacement.startswith("not_"):
            scores = {}
            for key, outcome in (options or {}).items():
                score = 0.0
                if "need" in outcome.lower() or "study" in outcome.lower():
                    score += 3.0
                if "happy" in outcome.lower() or "better" in outcome.lower():
                    score -= 2.0
                if "not possible" in outcome.lower():
                    score -= 3.0
                scores[key] = score
            
            best = max(scores, key=scores.get)
            print(f"Negation detected. Selected: {best}")
            return best
        
        # 物理接地
        original_props = self.grounding.ground_concept(original)
        replacement_props = self.grounding.ground_concept(replacement)
        
        print(f"'{original}': {original_props}")
        print(f"'{replacement}': {replacement_props}")
        
        # ノード初期化
        original_idx = self.label_to_idx.get(original, 0)
        replacement_idx = self.label_to_idx.get(replacement, 0)
        
        self.core.set_node_state(original_idx, **original_props)
        self.core.set_node_state(replacement_idx, **replacement_props)
        
        # 介入パラメータ
        harm_delta = replacement_props["harm"] - original_props["harm"]
        support_delta = replacement_props["support"] - original_props["support"]
        
        gate, gain = self._create_intervention(original_idx, replacement_idx, harm_delta, support_delta)
        
        # シミュレーション
        gate_base = torch.ones_like(gate)
        gain_base = torch.ones_like(gain)
        
        traj_base = self._simulate(gate_base, gain_base)
        traj_cf = self._simulate(gate, gain)
        
        impact = self._compute_impact(traj_base, traj_cf)
        
        # 期待重大度: Harm変化 + 質量(Support)変化 + 物理的インパクト
        expected_severity = abs(harm_delta) * 2.0 + abs(support_delta) * 1.5 + impact * 0.5
        expected_severity = max(0.0, min(1.0, expected_severity))
        
        print(f"Impact: {impact:.3f}, Expected severity: {expected_severity:.3f} (HarmD: {harm_delta:.2f}, SuppD: {support_delta:.2f})")
        
        if not options:
            return "Discovery mode"
        
        # 選択肢評価
        scores = {}
        for key, outcome in options.items():
            extracted_sev = self._extract_severity(outcome)
            distance = abs(extracted_sev - expected_severity)
            score = 10.0 / (1.0 + distance * 10.0)
            scores[key] = score
            print(f"  [{key}] {outcome[:50]}... sev={extracted_sev:.2f} → score={score:.2f}")
        
        best = max(scores, key=scores.get)
        print(f"Selected: {best} (score={scores[best]:.2f})")
        
        # 学習
        if scores[best] > 7.0:
            self.memory.learn(replacement, f"opt_{best}", expected_severity)
        
        return best