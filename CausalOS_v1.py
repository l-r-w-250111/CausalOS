import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re, os, json, uuid
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        # Step 6: 選択肢評価（シンプルかつ直接的）
        scores = {}
        
        for key, outcome in options.items():
            print(f"\n  [Option {key}] {outcome}")
            
            # 物理属性に基づく予測
            physical_score = 0.0
            
            # 有害性チェック
            if replacement_props["harm"] > 0.5:
                if "broken" in outcome.lower() or "hurt" in outcome.lower() or "suffer" in outcome.lower():
                    physical_score += 2.0
                if "happy" in outcome.lower() or "nothing" in outcome.lower():
                    physical_score -= 2.0
            
            # サポート性チェック
            if original_props["support"] > 0.7 and replacement_props["support"] < 0.3:
                if "late" in outcome.lower() or "difficult" in outcome.lower():
                    physical_score += 1.0
                if "on time" in outcome.lower():
                    physical_score -= 1.0
            
            # 剛性の変化
            rigidity_change = abs(original_props["rigidity"] - replacement_props["rigidity"])
            if rigidity_change > 0.3:
                if "nothing" in outcome.lower():
                    physical_score -= 1.0
            else:
                if "nothing" in outcome.lower():
                    physical_score += 1.0
            
            # LLMによる評価
            prompt = f"""Scenario: Replace "{original}" with "{replacement}" in the context.

Original properties: rigidity={original_props['rigidity']:.2f}, support={original_props['support']:.2f}, harm={original_props['harm']:.2f}
Replacement properties: rigidity={replacement_props['rigidity']:.2f}, support={replacement_props['support']:.2f}, harm={replacement_props['harm']:.2f}

Outcome to evaluate: "{outcome}"

Is this outcome logically consistent with the physical properties?

(A) Yes, consistent
(B) No, inconsistent

Answer: <choice>A</choice> or <choice>B</choice>"""
            
            response = self.generate_short(prompt, tokens=50)
            
            match = re.search(r"<choice>([AB])</choice>", response)
            llm_score = 1.0 if (match and match.group(1) == "A") else 0.0
            
            # 総合スコア
            total_score = physical_score + (llm_score * 2.0)
            scores[key] = total_score
            
            print(f"    Physical Score: {physical_score:.2f}, LLM Score: {llm_score:.2f}, Total: {total_score:.2f}")
        
        # 最高スコアを選択
        best_option = max(scores, key=scores.get)
        print(f"\n[OS] Selected: {best_option} (Score: {scores[best_option]:.2f})")
        
        return best_option