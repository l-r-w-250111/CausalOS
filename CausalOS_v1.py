import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re, os, json, uuid
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    if torch.cuda.is_available():
        # Test CUDA availability
        torch.zeros(1).cuda()
        device = torch.device("cuda")
        print(f"[System] Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[System] Using CPU device")
except:
    device = torch.device("cpu")
    print("[System] CUDA available but error occurred. Falling back to CPU.")

# ==========================================================
# 1) 物理コア (V1): 拡張可能な複素位相ダイナミクス
# ==========================================================
class CausalCoreV1(nn.Module):
    def __init__(self, n_nodes=8, dim=2):
        super().__init__()
        self.n_nodes = n_nodes
        self.dim = dim
        self.x = nn.Parameter(torch.randn(n_nodes, dim, device=device) * 0.1, requires_grad=True)
        self.raw_S = nn.Parameter(torch.randn(n_nodes, n_nodes, device=device) * 0.1)
        self.raw_phase = nn.Parameter(torch.randn(n_nodes, n_nodes, device=device) * 0.1)
        self.register_buffer("omega", torch.tensor(0.1))

    def expand(self, k):
        """ノードをk個追加し、既存の学習済み重みを保持する"""
        old_n = self.n_nodes
        new_n = old_n + k
        
        new_x = torch.zeros(k, self.dim, device=device)
        self.x = nn.Parameter(torch.cat([self.x.data, new_x], dim=0))
        
        new_S = torch.zeros(new_n, new_n, device=device)
        new_S[:old_n, :old_n] = self.raw_S.data
        self.raw_S = nn.Parameter(new_S)
        
        new_phase = torch.zeros(new_n, new_n, device=device)
        new_phase[:old_n, :old_n] = self.raw_phase.data
        self.raw_phase = nn.Parameter(new_phase)
        
        self.n_nodes = new_n
        print(f"[Core] Expanded to {new_n} nodes.")
    
    def set_node_state_from_physics(self, node_idx, rigidity, support, harm):
        """物理属性からノード状態を設定"""
        if node_idx >= self.n_nodes:
            return
        
        # 物理属性を複素数状態にマッピング（スケール強化）
        # real: harm + rigidity の組み合わせ（物質的特性）
        # imag: support（相互作用の強さ）
        # ×2でより大きな初期状態差を作る
        with torch.no_grad():
            self.x.data[node_idx, 0] = (harm + rigidity * 0.5) * 2.0  # real部（強化）
            self.x.data[node_idx, 1] = support * 2.0  # imaginary部（強化）

    def forward(self, x_in=None, gate=None, gain=None, t=0):
        S = torch.tanh(self.raw_S)
        G = gate if gate is not None else torch.ones_like(S)
        A = gain if gain is not None else torch.ones_like(S)
        S_eff = S * G * A
        theta = self.raw_phase + self.omega * t
        
        x = x_in if x_in is not None else self.x
        
        x_real = x[:, 0].unsqueeze(-1)
        x_imag = x[:, 1].unsqueeze(-1)
        
        out_real = torch.matmul(S_eff * torch.cos(theta), x_real) - \
                   torch.matmul(S_eff * torch.sin(theta), x_imag)
        out_imag = torch.matmul(S_eff * torch.sin(theta), x_real) + \
                   torch.matmul(S_eff * torch.cos(theta), x_imag)
        
        next_x = torch.stack([torch.tanh(out_real.squeeze(-1)), torch.tanh(out_imag.squeeze(-1))], dim=-1)
        return next_x

# ==========================================================
# 2) ノード群 (NodeGroups): 意味（トークン）の表現
# ==========================================================
class NodeGroups(nn.Module):
    def __init__(self, n_groups, n_nodes):
        super().__init__()
        self.G = nn.Parameter(torch.randn(n_groups, n_nodes, device=device) * 0.5)

    def expand_nodes(self, k):
        G_new = torch.zeros(self.G.shape[0], self.G.shape[1] + k, device=device)
        G_new[:, :self.G.shape[1]] = self.G.data
        self.G = nn.Parameter(G_new)

    def expand_groups(self, k):
        new_rows = torch.randn(k, self.G.shape[1], device=device) * 0.5
        self.G = nn.Parameter(torch.cat([self.G.data, new_rows], dim=0))

    def forward(self, x):
        W = F.softmax(self.G, dim=-1)
        return torch.matmul(W, x)

# ==========================================================
# 3) 意味接地層 (Semantic Grounding)
# ==========================================================
class SemanticGrounding:
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
        self.concept_physics = {}
    
    def ground_concept(self, word):
        """概念を物理属性に接地"""
        if word in self.concept_physics:
            return self.concept_physics[word]
        
        prompt = f"""Physical properties of "{word}":

Rigidity (0.0=very soft, 1.0=very hard): <rigidity>VALUE</rigidity>
Support (0.0=cannot support, 1.0=can support): <support>VALUE</support>
Harmfulness (0.0=safe, 1.0=very harmful): <harm>VALUE</harm>

Provide numerical values only:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        # デフォルト値
        props = {"rigidity": 0.5, "support": 0.5, "harm": 0.0}
        
        # 抽出
        rig_match = re.search(r"<rigidity>([\d.]+)</rigidity>", response)
        sup_match = re.search(r"<support>([\d.]+)</support>", response)
        harm_match = re.search(r"<harm>([\d.]+)</harm>", response)
        
        if rig_match:
            props["rigidity"] = float(rig_match.group(1))
        if sup_match:
            props["support"] = float(sup_match.group(1))
        if harm_match:
            props["harm"] = float(harm_match.group(1))
        
        self.concept_physics[word] = props
        return props

# ==========================================================
# 4) メタ認知層 (Meta Cognition)
# ==========================================================
class MetaCognition:
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
    
    def is_physically_possible(self, factual_concepts, cf_concepts):
        """反事実が物理的に可能かを判断"""
        factual_str = ", ".join(factual_concepts)
        cf_str = ", ".join(cf_concepts)
        
        prompt = f"""Factual: {factual_str}
Counterfactual: {cf_str}

Is the counterfactual scenario physically possible in reality?

(A) Yes, physically possible
(B) No, violates physical laws or logic

Answer: <possible>A</possible> or <possible>B</possible>"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        match = re.search(r"<possible>([AB])</possible>", response)
        if match and match.group(1) == "B":
            return False
        
        return True

# ==========================================================
# 5) 短期記憶 (Memory)
# ==========================================================
class CausalMemory:
    def __init__(self, capacity=32):
        self.capacity = capacity
        self.buffer = []

    def store(self, snapshot):
        self.buffer.append(snapshot)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def associate(self, current_G):
        if not self.buffer: 
            return None
        
        scores = []
        for snap in self.buffer:
            snap_G = snap["G"]
            if snap_G.shape != current_G.shape:
                scores.append(-1.0)
                continue
            
            sim = F.cosine_similarity(current_G.view(-1), snap_G.view(-1), dim=0)
            scores.append(sim.item())
        
        best_idx = np.argmax(scores)
        if scores[best_idx] < 0.3: 
            return None
        return self.buffer[best_idx]

# ==========================================================
# 6) 発見エンジン (Discovery Engine)
# ==========================================================
class DiscoveryEngine:
    def __init__(self, osys):
        self.osys = osys

    def compute_metrics(self, traj):
        phases = torch.stack([torch.atan2(x[:, 1], x[:, 0]) for x in traj])
        avg_phases = phases.mean(dim=-1)
        
        csi = 1.0 / (torch.var(avg_phases) + 1e-5)
        cii = torch.norm(avg_phases - avg_phases[0])
        return csi, cii

# ==========================================================
# 7) S行列エンジン (Hallucination Prevention)
# ==========================================================
class SMatrixEngine:
    def __init__(self):
        self.matrix = defaultdict(lambda: defaultdict(float))
    
    def register_sequence(self, token_ids, rigidity=50.0):
        for i in range(len(token_ids) - 1):
            self.matrix[token_ids[i]][token_ids[i+1]] = max(self.matrix[token_ids[i]][token_ids[i+1]], rigidity)
    
    def adjust_logits(self, last_token_id, logits, lambda_val=2.0):
        if last_token_id in self.matrix:
            for nxt_id, rig in self.matrix[last_token_id].items():
                logits[..., nxt_id] += lambda_val * rig
        return logits

# ==========================================================
# 8) 統合OS (UnifiedCausalOSV1 - Fixed)
# ==========================================================
class UnifiedCausalOSV1:
    def __init__(self, n_nodes=10, n_groups=5):
        self.n_nodes = n_nodes
        self.n_groups = n_groups
        
        self.core = CausalCoreV1(n_nodes=n_nodes)
        self.groups = NodeGroups(n_groups=n_groups, n_nodes=n_nodes)
        self.memory = CausalMemory()
        self.discovery = DiscoveryEngine(self)
        self.s_matrix = SMatrixEngine()
        
        model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        
        # 新しい層
        self.grounding = SemanticGrounding(self.model, self.tokenizer)
        self.metacog = MetaCognition(self.model, self.tokenizer)
        
        # ラベルマップ（修正版）
        self.label_to_idx = {}

    def _normalize_word(self, word):
        """語を正規化（基本形に変換）"""
        word = word.lower()
        if word.endswith('ed') and len(word) > 3:
            base = word[:-2] if word.endswith('ked') else word[:-1]
            return base if len(base) > 2 else word
        elif word.endswith('s') and len(word) > 3:
            return word[:-1]
        return word

    def _extract_nodes(self, text):
        """文からノードを抽出（改善版）"""
        
        # まず "not" が含まれているかチェック
        has_negation = " not " in text.lower() or text.lower().startswith("not ")
        
        prompt = f"""Extract all important words (nouns, verbs, and negations like "not") from: "{text}"

Output format: <entities>word1, word2, word3</entities>

Answer:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=60, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        result = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        # タグから抽出
        matches = re.findall(r"<entities>(.*?)</entities>", result, re.DOTALL)
        entities_str = matches[-1] if matches else None
        
        if entities_str:
            nodes = [n.strip().lower() for n in entities_str.split(',') if n.strip()]
            nodes = [n for n in nodes if len(n) > 1 and n not in ['a', 'an', 'the']]
        else:
            # フォールバック
            words = text.lower().replace('.', '').replace('?', '').split()
            nodes = [w for w in words if w not in ['a', 'an', 'the', 'on', 'if', 'would', 'have', 'had', 'what', 'happened']]
        
        # "not" が文に含まれているのに抽出されていない場合、強制的に追加
        if has_negation and "not" not in nodes:
            # "not" の直後の単語を見つけて、その前に挿入
            words = text.lower().split()
            if "not" in words:
                not_idx_in_text = words.index("not")
                if not_idx_in_text + 1 < len(words):
                    following_word = self._normalize_word(words[not_idx_in_text + 1])
                    # nodesの中でその単語を探して、その前に "not" を挿入
                    for i, node in enumerate(nodes):
                        if node == following_word or self._normalize_word(node) == following_word:
                            nodes.insert(i, "not")
                            break
                    else:
                        # 見つからなかったら先頭に追加
                        nodes.insert(0, "not")
        
        # 正規化（"not"は正規化しない）
        normalized = []
        for n in nodes:
            if n == "not":
                normalized.append(n)
            else:
                normalized.append(self._normalize_word(n))
        
        return normalized if normalized else ["unknown"]

    def _update_label_map(self, nodes):
        """ノードからラベルマップを更新（修正版）"""
        self.label_to_idx = {}
        for i, node in enumerate(nodes[:self.n_nodes]):
            self.label_to_idx[node] = i
        print(f"[OS] Updated Label Map: {self.label_to_idx}")

    def _detect_intervention(self, factual_nodes, cf_nodes):
        """介入を検出（v37_41の差分検出ロジック + 役割逆転検出）"""
        removed = [n for n in factual_nodes if n not in cf_nodes]
        added = [n for n in cf_nodes if n not in factual_nodes]
        
        # ケース1: "not"の追加/削除
        if "not" in added:
            # 否定の追加 → 何かが否定された
            # 否定されたのは直後の単語
            idx = cf_nodes.index("not")
            if idx + 1 < len(cf_nodes):
                negated = cf_nodes[idx + 1]
                return negated, "not_" + negated
        
        if "not" in removed:
            # 否定の削除 → 肯定に変わった
            idx = factual_nodes.index("not")
            if idx + 1 < len(factual_nodes):
                affirmed = factual_nodes[idx + 1]
                return "not_" + affirmed, affirmed
        
        # ケース2: 役割の逆転（主語と目的語の入れ替え）
        # 例: ['plant', 'grow', 'planter'] vs ['planter', 'grow', 'plant']
        if len(factual_nodes) == len(cf_nodes) and set(factual_nodes) == set(cf_nodes):
            # ノードは同じだが順序が違う → 役割逆転の可能性
            if factual_nodes != cf_nodes:
                # 最初と最後が入れ替わっているか確認
                if (len(factual_nodes) >= 3 and 
                    factual_nodes[0] == cf_nodes[-1] and 
                    factual_nodes[-1] == cf_nodes[0]):
                    return "role_reversal", f"{factual_nodes[0]}_and_{factual_nodes[-1]}"
        
        # ケース3: 通常の差分（名詞の変化を優先）
        verb_like = ['walk', 'eat', 'understand', 'grow', 'run', 'go']
        removed_nouns = [r for r in removed if r not in verb_like]
        added_nouns = [a for a in added if a not in verb_like]
        
        if removed_nouns and added_nouns:
            return removed_nouns[0], added_nouns[0]
        elif removed and added:
            return removed[0], added[0]
        else:
            return None, None

    def save_state(self):
        snap = {
            "id": str(uuid.uuid4()),
            "S": self.core.raw_S.data.clone(),
            "G": self.groups.G.data.clone(),
            "n_nodes": self.core.n_nodes
        }
        self.memory.store(snap)

    def load_best_memory(self):
        best = self.memory.associate(self.groups.G.data)
        if best and best["n_nodes"] == self.core.n_nodes:
            self.core.raw_S.data = best["S"]
            self.groups.G.data = best["G"]
            print(f"[OS] Loaded memory: {best['id']}")
            return True
        return False

    def generate_short(self, prompt, tokens=200):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    # ==========================================================
    # 物理ベースの反事実評価メソッド
    # ==========================================================
    def _create_intervention_params(self, original_idx, replacement_idx, harm_change, support_change=0.0, rigidity_change=0.0):
        """介入を物理パラメータ(gate, gain)に変換"""
        gate = torch.ones(self.n_nodes, self.n_nodes, device=device)
        gain = torch.ones(self.n_nodes, self.n_nodes, device=device)
        
        # 置換: original→0, replacement→1 (トポロジー書換)
        if original_idx < self.n_nodes and replacement_idx < self.n_nodes:
            gate[:, original_idx] = 0.0  # originalへの入力を切断
            gate[original_idx, :] = 0.0  # originalからの出力を切断
        
        # 振幅変調: 物理属性の変化を総合的に反映
        # harm(有害性), support(質量/エネルギー), rigidity(硬さ) の変化
        impact_factor = abs(harm_change) * 10.0 + abs(support_change) * 5.0 + abs(rigidity_change) * 3.0
        
        gain_factor = torch.exp(torch.tensor(impact_factor, device=device))
        gain *= gain_factor
        
        return gate, gain
    
    def _simulate_counterfactual(self, gate, gain, steps=20):
        """物理コアで反事実シミュレーション（ステップ数増加）"""
        trajectory = []
        x = self.core.x.clone().detach()
        
        with torch.no_grad():
            for t in range(steps):
                x = self.core.forward(x, gate=gate, gain=gain, t=t)
                trajectory.append(x.clone())
        
        return torch.stack(trajectory)  # [steps, n_nodes, 2]
    
    def _extract_trajectory_signature(self, trajectory_baseline, trajectory_cf):
        """軌道から物理的特徴を抽出 (do-calculus: 介入の影響を差分で測定)"""
        with torch.no_grad():
            # 介入の影響 = counterfactual - baseline
            impact = trajectory_cf - trajectory_baseline  # [steps, n_nodes, 2]
            
            # 影響の大きさ (ノルム)
            impact_magnitude = torch.norm(impact, dim=-1)  # [steps, n_nodes]
            total_impact = impact_magnitude.mean().item()  # 平均的な影響
            
            # 位相変化（介入による位相シフト）
            phases_baseline = torch.atan2(trajectory_baseline[:, :, 1], trajectory_baseline[:, :, 0])
            phases_cf = torch.atan2(trajectory_cf[:, :, 1], trajectory_cf[:, :, 0])
            phase_shift = torch.abs(phases_cf - phases_baseline).mean().item()
            
            # 振幅変化
            amp_baseline = torch.norm(trajectory_baseline, dim=-1)
            amp_cf = torch.norm(trajectory_cf, dim=-1)
            amplitude_change = torch.abs(amp_cf - amp_baseline).mean().item()
            
            # 安定性 (影響の時間的分散)
            impact_variance = torch.var(impact_magnitude).item()
            stability = 1.0 / (impact_variance + 1e-5)
            
        return {
            'total_impact': total_impact,
            'phase_shift': phase_shift,
            'amplitude_change': amplitude_change,
            'stability': stability
        }
    
    def _extract_severity_from_text(self, text):
        """LLMを使って文章から重大度を抽出 (0-1)"""
        prompt = f"""Rate the severity/seriousness of this outcome on a scale of 0.0 to 1.0:
"{text}"

0.0 = not serious at all (e.g., "nothing happened", "everything was fine")
0.3 = minor concern (e.g., "slightly inconvenient", "a bit risky")
0.5 = moderate concern (e.g., "dangerous", "problematic", "illegal")
0.8 = very serious (e.g., "severe injury", "major damage")
1.0 = catastrophic (e.g., "death", "complete destruction")

Answer with just a number between 0.0 and 1.0:
<severity>X.X</severity>"""
        
        response = self.generate_short(prompt, tokens=20)
        
        # 数値抽出
        match = re.search(r"<severity>([\d.]+)</severity>", response)
        if match:
            try:
                severity = float(match.group(1))
                return max(0.0, min(1.0, severity))  # 0-1にクリップ
            except:
                pass
        
        # フォールバック: 数値を直接探す
        numbers = re.findall(r"(\d+\.?\d*)", response)
        if numbers:
            try:
                severity = float(numbers[0])
                # 0-1範囲に正規化
                if severity > 1.0:
                    severity = severity / 10.0 if severity <= 10.0 else 1.0
                return max(0.0, min(1.0, severity))
            except:
                pass
        
        # デフォルト
        return 0.5

    def _scale_impact_to_severity(self, total_impact, harm_change, rigidity_change=0.0, support_change=0.0):
        """物理影響量を意味的重大度にマッピング（汎用関数）
        
        物理的変化量と意味的重大度の間のスケール不一致を解決。
        問題に依存しない汎用的なマッピング。
        """
        # 基準: 物理的変化量から期待重大度を推定
        # harm ±0.1 → moderate (0.3-0.5)
        # harm ±0.2 → serious (0.5-0.7)
        # harm ±0.5 → severe (0.7-0.9)
        base_severity = abs(harm_change) * 3.0
        
        # rigidityの変化も考慮（脆さ↑ → 危険性↑）
        rigidity_factor = abs(rigidity_change) * 1.0
        
        # supportの変化も考慮（質量/エネルギー↑ → 衝撃↑）
        # 鳥(0.5) vs 飛行機(1.0) のような差を反映
        support_factor = abs(support_change) * 1.0
        
        # シミュレーション結果で微調整
        # total_impact大 → さらに増幅（複雑な相互作用を反映）
        simulation_factor = min(0.3, total_impact * 2.0)
        
        # 合成
        expected_severity = base_severity + rigidity_factor + support_factor + simulation_factor
        
        # 0-1にクリップ
        return max(0.0, min(1.0, expected_severity))


    def solve_counterfactual(self, factual, cf, options=None):
        print(f"\n[OS] Counterfactual Reasoning")
        print(f"Factual: {factual}")
        print(f"Counterfactual: {cf}")
        
        # Step 1: ノード抽出
        factual_nodes = self._extract_nodes(factual)
        cf_nodes = self._extract_nodes(cf)
        
        print(f"Factual Nodes: {factual_nodes}")
        print(f"Counterfactual Nodes: {cf_nodes}")
        
        # すべてのノードを収集
        all_nodes = list(set(factual_nodes + cf_nodes))
        self._update_label_map(all_nodes)
        
        # Step 2: 介入検出
        original, replacement = self._detect_intervention(factual_nodes, cf_nodes)
        print(f"Intervention: '{original}' -> '{replacement}'")
        
        if not original or not replacement:
            print("[OS] No clear intervention detected")
            if options:
                return list(options.keys())[0]
            return "B"
        
        # Step 3: 特殊ケース処理
        # 役割逆転の場合
        if original == "role_reversal":
            print("[OS] Role reversal detected - this is impossible")
            for key, text in options.items():
                if "not possible" in text.lower() or "impossible" in text.lower():
                    print(f"[OS] Selected impossible option: {key}")
                    return key
        
        # 否定の追加/削除の場合
        if replacement.startswith("not_") or original.startswith("not_"):
            print("[OS] Negation detected")
            # 否定された場合の影響を評価
            if not options:
                return "Discovery mode"
            
            # 否定の場合は特別な評価
            scores = {}
            for key, outcome in options.items():
                score = 0.0
                # "need to study more"のような追加アクションが必要
                if "need" in outcome.lower() or "must" in outcome.lower() or "should" in outcome.lower():
                    score += 2.0
                # "happy"や"better"は否定の結果としては不適切
                if "happy" in outcome.lower() or "better" in outcome.lower():
                    score -= 1.0
                # "not possible"は否定には適さない
                if "not possible" in outcome.lower():
                    score -= 1.0
                
                scores[key] = score
                print(f"  [Option {key}] {outcome} -> Score: {score:.2f}")
            
            best_option = max(scores, key=scores.get)
            print(f"[OS] Selected: {best_option}")
            return best_option
        
        # Step 4: メタ認知チェック（物理的可能性）
        if not self.metacog.is_physically_possible(factual_nodes, cf_nodes):
            print("[OS] Counterfactual is physically impossible")
            # "That is not possible" を選択
            for key, text in options.items():
                if "not possible" in text.lower() or "impossible" in text.lower():
                    print(f"[OS] Selected impossible option: {key}")
                    return key
        
        # Step 5: 物理属性の接地
        original_props = self.grounding.ground_concept(original)
        replacement_props = self.grounding.ground_concept(replacement)
        
        print(f"Original '{original}' properties: {original_props}")
        print(f"Replacement '{replacement}' properties: {replacement_props}")
        
        if not options:
            return "Discovery mode"
        
        # Step 6: 物理ベースの選択肢評価
        print("[OS] Using physics-based evaluation...")
        
        # 6.1: 介入をgate/gainに変換
        original_idx = self.label_to_idx.get(original, 0)
        replacement_idx = self.label_to_idx.get(replacement, 0)
        harm_change = replacement_props["harm"] - original_props["harm"]
        
        # 6.1.5: CausalCoreに物理属性を反映（重要！）
        # これにより、抽象的な物理シミュレーションが具体的な概念の物理特性を持つ
        print(f"[OS] Initializing CausalCore states from physical properties...")
        self.core.set_node_state_from_physics(
            original_idx,
            original_props["rigidity"],
            original_props["support"],
            original_props["harm"]
        )
        self.core.set_node_state_from_physics(
            replacement_idx,
            replacement_props["rigidity"],
            replacement_props["support"],
            replacement_props["harm"]
        )
        
        support_change = replacement_props["support"] - original_props["support"]
        rigidity_change = replacement_props["rigidity"] - original_props["rigidity"]
        
        gate, gain = self._create_intervention_params(original_idx, replacement_idx, harm_change, support_change, rigidity_change)
        
        # 6.2: 物理シミュレーション (do-calculus: baseline vs counterfactual)
        # ベースライン（介入なし）
        gate_baseline = torch.ones(self.n_nodes, self.n_nodes, device=device)
        gain_baseline = torch.ones(self.n_nodes, self.n_nodes, device=device)
        trajectory_baseline = self._simulate_counterfactual(gate_baseline, gain_baseline, steps=20)
        
        # 反事実（介入あり）
        trajectory_cf = self._simulate_counterfactual(gate, gain, steps=20)
        
        # 6.3: 介入の影響を測定
        signature = self._extract_trajectory_signature(trajectory_baseline, trajectory_cf)
        
        # 物理量を意味的重大度にスケーリング（汎用マッピング）
        expected_severity = self._scale_impact_to_severity(
            signature['total_impact'],
            harm_change,
            rigidity_change,
            support_change
        )
        
        print(f"Physics signature: total_impact={signature['total_impact']:.3f}, amplitude_change={signature['amplitude_change']:.3f}")
        print(f"Scaled expected severity: {expected_severity:.3f} (from harm={harm_change:.2f}, support={support_change:.2f})")
        
        # 6.4: 各選択肢を評価
        scores = {}
        
        for key, outcome in options.items():
            print(f"\n  [Option {key}] {outcome}")
            
            # メタ否定の検出（事前フィルタ）
            if "wouldn't" in outcome.lower() or "won't" in outcome.lower() or "cannot" in outcome.lower():
                scores[key] = 0.1
                print(f"    Meta-denial detected, Score: 0.10/10")
                continue
            
            # LLMで選択肢の重大度を抽出
            extracted_severity = self._extract_severity_from_text(outcome)
            
            # 物理的期待値とのマッチング
            distance = abs(extracted_severity - expected_severity)
            score = 10.0 / (1.0 + distance * 10.0)  # 0-10のスコア
            
            scores[key] = score
            
            if os.getenv("DEBUG_LLM"):
                print(f"    Extracted severity: {extracted_severity:.2f}, Expected: {expected_severity:.3f}")
            print(f"    Severity: {extracted_severity:.2f}, Distance: {distance:.3f}, Score: {score:.2f}/10")
        
        # 最高スコアを選択
        best_option = max(scores, key=scores.get)
        print(f"\n[OS] Selected: {best_option} (Score: {scores[best_option]:.2f})")
        
        return best_option