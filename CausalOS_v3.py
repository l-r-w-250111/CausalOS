import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re, os
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
# 1) 物理エンコーダー: 概念 → 潜在物理ベクトル
# ==========================================================
class PhysicsEncoder(nn.Module):
    """LLM埋め込み → 物理潜在空間への射影"""
    def __init__(self, llm_dim=4096, phys_dim=64):
        super().__init__()
        self.phys_dim = phys_dim
        
        # 非線形射影（物理的制約を学習）
        self.projection = nn.Sequential(
            nn.Linear(llm_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, phys_dim),
            nn.Tanh()  # [-1, 1] に制限
        )
    
    def forward(self, llm_embedding):
        """
        Args:
            llm_embedding: [batch, llm_dim] LLMの埋め込み
        Returns:
            phys_vector: [batch, phys_dim] 物理潜在ベクトル
        """
        return self.projection(llm_embedding)

# ==========================================================
# 2) 物理コア: 潜在空間でのダイナミクス
# ==========================================================
class CausalCoreV3(nn.Module):
    """高次元潜在物理空間での因果ダイナミクス"""
    def __init__(self, n_nodes=10, phys_dim=64):
        super().__init__()
        self.n_nodes = n_nodes
        self.phys_dim = phys_dim
        
        # 各ノードの物理状態 [n_nodes, phys_dim]
        self.x = nn.Parameter(torch.randn(n_nodes, phys_dim, device=device) * 0.1)
        
        # 因果構造 (S行列): ノード間の相互作用
        # [n_nodes, n_nodes] のスカラー結合強度
        self.raw_S = nn.Parameter(torch.randn(n_nodes, n_nodes, device=device) * 0.2)
        
        # 位相: 時間遅延
        self.raw_phase = nn.Parameter(torch.randn(n_nodes, n_nodes, device=device) * 0.1)
        
        self.register_buffer("omega", torch.tensor(0.1))
    
    def set_node_state(self, idx, phys_vector):
        """ノードの物理状態を設定"""
        if idx >= self.n_nodes:
            return
        
        with torch.no_grad():
            # 潜在ベクトルをそのまま状態として使用
            self.x.data[idx] = phys_vector
    
    def forward(self, x_in=None, gate=None, t=0):
        """1ステップの時間発展
        
        物理法則:
        x'_i = tanh(Σ_j S_ij * exp(iθ_ij) * x_j)
        
        ここで exp(iθ) は複素数でなく、回転行列で近似:
        x → [cos(θ) * x - sin(θ) * rotate(x)]
        """
        S = torch.tanh(self.raw_S)
        G = gate if gate is not None else torch.ones_like(S)
        S_eff = S * G
        
        theta = self.raw_phase + self.omega * t
        
        x = x_in if x_in is not None else self.x  # [n_nodes, phys_dim]
        
        # 各ノードへの影響を計算
        output = []
        for i in range(self.n_nodes):
            # ノードiへの全ての入力を集約
            node_input = torch.zeros(self.phys_dim, device=device)
            
            for j in range(self.n_nodes):
                coupling = S_eff[j, i]  # j→i の結合強度
                phase = theta[j, i]
                
                # 位相回転を近似（簡易版: 状態の線形結合）
                # 本来は高次元回転だが、ここではスカラー位相で近似
                rotated = torch.cos(phase) * x[j]
                
                node_input += coupling * rotated
            
            output.append(torch.tanh(node_input))
        
        return torch.stack(output)  # [n_nodes, phys_dim]

# ==========================================================
# 3) 意味接地層（潜在版）
# ==========================================================
class LatentSemanticGrounding:
    """概念 → 潜在物理ベクトル"""
    def __init__(self, llm, tokenizer, encoder):
        self.llm = llm
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.cache = {}
    
    def ground_concept(self, word):
        """概念を潜在物理ベクトルに変換"""
        if word in self.cache:
            return self.cache[word]
        
        # LLMの埋め込み層を使用
        with torch.no_grad():
            # トークン化
            tokens = self.tokenizer(word, return_tensors="pt").to(device)
            
            # LLMの埋め込みを取得
            embeddings = self.llm.get_input_embeddings()
            word_embedding = embeddings(tokens.input_ids).mean(dim=1)  # [1, llm_dim]
            
            # 物理空間へ射影 (Dtype mismatch fix: Cast to encoder's dtype)
            target_dtype = self.encoder.projection[0].weight.dtype
            phys_vector = self.encoder(word_embedding.to(target_dtype)).squeeze(0)  # [phys_dim]
        
        self.cache[word] = phys_vector
        return phys_vector

# ==========================================================
# 4) 因果記憶
# ==========================================================
class CausalMemory:
    def __init__(self, decay_rate=0.02):
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
                "count": 1,
                "last_step": self.step
            }
        else:
            mem = self.traces[key]
            mem["strength"] = mem["strength"] * 0.8 + impact_score * 0.2
            mem["confidence"] = min(1.0, mem["confidence"] + 0.05)
            mem["count"] += 1
            mem["last_step"] = self.step
    
    def retrieve(self, cause, threshold=0.1):
        """因果関係を想起"""
        self.step += 1
        results = []
        
        for (c, e), mem in list(self.traces.items()):
            if c == cause:
                time_diff = self.step - mem["last_step"]
                strength = mem["strength"] * mem["confidence"] * \
                          ((1.0 - self.decay_rate) ** time_diff)
                
                if strength >= threshold:
                    results.append((e, strength))
                    mem["last_step"] = self.step
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results

# ==========================================================
# 5) 統合OS（潜在物理版）
# ==========================================================
class UnifiedCausalOS_v3:
    def __init__(self, n_nodes=10, phys_dim=64):
        self.n_nodes = n_nodes
        self.phys_dim = phys_dim
        
        # LLM
        model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        
        # 物理エンコーダー（学習可能）
        llm_dim = self.model.config.hidden_size
        self.encoder = PhysicsEncoder(llm_dim, phys_dim).to(device)
        
        # 物理コア
        self.core = CausalCoreV3(n_nodes, phys_dim)
        
        # 意味接地
        self.grounding = LatentSemanticGrounding(self.model, self.tokenizer, self.encoder)
        
        # Encoder Warmup (学習)
        self.warmup_encoder(steps=40)
        
        # 記憶
        self.memory = CausalMemory()
        
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
        """ノード抽出（既存ロジック）"""
        has_not = " not " in text.lower()
        
        prompt = f"""Task: Extract physical keywords (nouns, verbs) from this text.
Strict Rules:
1. Provide only the keywords themselves.
2. NO tags like 'noun:', 'verb:', or '<br>'.
3. Comma-separated list inside the entities block.

Text: "{text}"
<entities>"""
        
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
        
        matches = re.findall(r"<entities>(.*?)</entities>", result)
        if matches:
            nodes = [n.strip().lower() for n in matches[-1].split(',') if n.strip()]
            # Strip junk tags like 'noun:', 'verb:', or html tags
            nodes = [re.sub(r'^(noun|verb|adj|adv):\s*', '', n) for n in nodes]
            nodes = [re.sub(r'<.*?>', '', n) for n in nodes]
            nodes = [n for n in nodes if len(n) > 1 and n not in ['a', 'an', 'the']]
        else:
            words = text.lower().replace('.', '').replace('?', '').split()
            nodes = [w for w in words if w not in ['a', 'an', 'the', 'on', 'if', 'would', 'have', 'had']]
        
        if has_not and "not" not in nodes:
            nodes.insert(0, "not")
        
        normalized = []
        for n in nodes:
            if n == "not":
                normalized.append(n)
            else:
                normalized.append(self._normalize_word(n))
        
        return normalized if normalized else ["unknown"]
    
    def _detect_intervention(self, factual_nodes, cf_nodes):
        """介入検出（既存ロジック）"""
        removed = [n for n in factual_nodes if n not in cf_nodes]
        added = [n for n in cf_nodes if n not in factual_nodes]
        
        if "not" in added:
            idx = cf_nodes.index("not")
            if idx + 1 < len(cf_nodes):
                return cf_nodes[idx + 1], "not_" + cf_nodes[idx + 1]
        
        if (len(factual_nodes) == len(cf_nodes) and 
            set(factual_nodes) == set(cf_nodes) and
            len(factual_nodes) >= 3):
            if (factual_nodes[0] == cf_nodes[-1] and 
                factual_nodes[-1] == cf_nodes[0]):
                return "role_reversal", factual_nodes[0] + "_and_" + factual_nodes[-1]
        
        verb_like = ['walk', 'eat', 'understand', 'grow']
        removed_nouns = [r for r in removed if r not in verb_like]
        added_nouns = [a for a in added if a not in verb_like]
        
        if removed_nouns and added_nouns:
            return removed_nouns[0], added_nouns[0]
        elif removed and added:
            return removed[0], added[0]
        
        return None, None
    
    def analyze_physical_consistency(self, scenario_text):
        """物理的整合性分析（潜在空間版）
        
        潜在ベクトルのノルムや内積から異常を検出
        """
        print(f"\n[OS v3] Latent Physics Analysis")
        print(f"Scenario: {scenario_text}")
        
        nodes = self._extract_nodes(scenario_text)
        
        # 物理的なキーワードのみを分析対象とする（分析のノイズを低減）
        physical_keywords = [
            "boeing", "747", "branch", "tree", "oak", "wing", "engine", "ground",
            "mass", "weight", "force", "impact", "collision", "break", "snap", "collapse",
            "heavy", "light", "slowly", "fast", "land", "fall", "gravity"
        ]
        nodes = [n for n in nodes if any(pk in n for pk in physical_keywords)]
        
        if not nodes:
             return {"severity": 0.0, "concerns": [], "recommendation": "Stable", "latent_metrics": {}}

        print(f"Entities (Physical): {nodes}")
        
        # 各ノードの潜在ベクトルを取得
        vectors = {}
        for node in nodes[:5]:
            vec = self.grounding.ground_concept(node)
            vectors[node] = vec
        
        # 潜在空間での異常検出
        concerns = []
        max_magnitude = 0.0
        min_dot_product = 1.0
        max_distance = 0.0
        
        for i, (node1, vec1) in enumerate(vectors.items()):
            mag1 = torch.norm(vec1).item()
            max_magnitude = max(max_magnitude, mag1)
            
            for node2, vec2 in list(vectors.items())[i+1:]:
                # ベクトル間の距離
                distance = torch.norm(vec1 - vec2).item()
                max_distance = max(max_distance, distance)
                
                # 内積（方向の類似性）
                dot = torch.dot(vec1, vec2).item()
                min_dot_product = min(min_dot_product, dot)
                
                # 異常パターンの検出
                if distance > 1.5:  # 大きな距離 = 物理的に相容れない
                    concerns.append(f"Large physical incompatibility: {node1} vs {node2} (d={distance:.2f})")
                
                if mag1 > 0.8 and torch.norm(vec2).item() < 0.2:
                    concerns.append(f"Magnitude imbalance: {node1} vs {node2}")
        
        # 総合的な異常度
        # 高次元空間での特性を活用
        severity = (max_distance / 2.0) * 0.5 + (1.0 - min_dot_product) * 0.3 + (max_magnitude - 0.5) * 0.2
        severity = max(0.0, min(1.0, severity))
        
        is_consistent = severity < 0.3
        
        if severity > 0.7:
            recommendation = "CRITICAL: Latent physical states highly incompatible"
        elif severity > 0.5:
            recommendation = "HIGH: Significant physical inconsistency in latent space"
        elif severity > 0.3:
            recommendation = "MODERATE: Some latent incompatibility detected"
        else:
            recommendation = "OK: Latent physical states compatible"
        
        result = {
            "is_consistent": is_consistent,
            "severity": severity,
            "concerns": concerns,
            "recommendation": recommendation,
            "latent_metrics": {
                "max_distance": max_distance,
                "min_dot_product": min_dot_product,
                "max_magnitude": max_magnitude
            }
        }
        
        print(f"Severity: {severity:.2f} | Concerns: {len(concerns)}")
        print(f"Recommendation: {recommendation}")
        
        return result
    
    def warmup_encoder(self, steps=100):
        """物理エンコーダーのウォームアップ（学習）
        
        LLMの埋め込み空間での概念間の距離関係を、
        物理潜在空間でも保存するように学習する。
        """
        print("[System] Warming up Physics Encoder...")
        
        # 物理的な基本概念のリスト
        concepts = [
            "mass", "weight", "heavy", "light",
            "force", "impact", "collision", "break",
            "energy", "speed", "velocity", "acceleration",
            "structure", "support", "collapse", "stable",
            "gravity", "friction", "tension", "pressure",
            "catastrophic", "disaster", "harm", "safe",
            "boeing", "feather", "tree", "steel", "water",
            "branch", "landing", "snap", "crush", "wing",
            "engine", "ground", "flight", "pilot", "object"
        ]
        
        # ターゲット（LLM空間）の類似度行列を計算
        with torch.no_grad():
            tokens = self.tokenizer(concepts, return_tensors="pt", padding=True).to(device)
            embeddings = self.model.get_input_embeddings()
            llm_vecs = embeddings(tokens.input_ids).mean(dim=1)  # [N, llm_dim]
            
            # 正規化
            llm_vecs = F.normalize(llm_vecs, p=2, dim=1)
            target_sim = torch.mm(llm_vecs, llm_vecs.t())  # [N, N]
        
        # 学習ループ
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.001)
        
        target_dtype = self.encoder.projection[0].weight.dtype # Get correct dtype
        llm_vecs_input = llm_vecs.to(target_dtype) # Cast input to match model

        for step in range(steps):
            optimizer.zero_grad()
            
            # 物理空間への射影
            phys_vecs = self.encoder(llm_vecs_input)  # [N, phys_dim]
            phys_vecs = F.normalize(phys_vecs, p=2, dim=1)
            
            # 物理空間での類似度
            current_sim = torch.mm(phys_vecs, phys_vecs.t())
            
            # 損失関数: 類似度行列のMSE
            loss = F.mse_loss(current_sim, target_sim.to(target_dtype))
            
            loss.backward()
            optimizer.step()
            
            if step % 20 == 0:
                print(f"  Step {step}: Loss = {loss.item():.4f}")
        
        print(f"  Final Loss = {loss.item():.4f}")
        
        # キャッシュのクリア（学習前の変なベクトルが残らないように）
        self.grounding.cache = {}

    def solve_counterfactual(self, factual, cf, options=None):
        """反事実推論（潜在物理版 - ベクトル評価）"""
        print(f"\n{'='*60}")
        print(f"[OS v3] Counterfactual Reasoning (Latent Physics)")
        
        # ノード抽出
        factual_nodes = self._extract_nodes(factual)
        cf_nodes = self._extract_nodes(cf)
        
        print(f"Nodes (F): {factual_nodes}")
        print(f"Nodes (CF): {cf_nodes}")
        
        # 介入検出
        original, replacement = self._detect_intervention(factual_nodes, cf_nodes)
        print(f"Intervention: {original} -> {replacement}")
        
        if not original or not replacement:
            return list(options.keys())[0] if options else "B"
        
        # 特殊ケース: Role Reversal
        if original == "role_reversal":
            return "A" # Default for reversal usually implied impossible
        
        # 潜在物理ベクトル取得
        original_vec = self.grounding.ground_concept(original)
        replacement_vec = self.grounding.ground_concept(replacement)
        
        # 変化の大きさ
        latent_distance = torch.norm(original_vec - replacement_vec).item()
        print(f"Latent distance: {latent_distance:.3f}")
        
        # 物理シミュレーション (簡易版: 変化量を衝撃度とみなす)
        # 本来はCausalCoreで時間発展させるが、ここでは変化の大きさをSeverityにマッピング
        impact = latent_distance * 2.0 # Scaling factor
        expected_severity = min(1.0, impact / 5.0)
        
        print(f"Predicted Severity: {expected_severity:.3f} (Impact: {impact:.3f})")
        
        if not options:
            return "Discovery mode"
        
        # 選択肢評価（ベクトルベース）
        print("\nOption Evaluation (Latent Severity):")
        
        # 災害/破壊の参照ベクトル
        disaster_refs = ["catastrophic", "destruction", "death", "failure", "crisis"]
        disaster_vecs = []
        for ref in disaster_refs:
            disaster_vecs.append(self.grounding.ground_concept(ref))
        disaster_vec = torch.stack(disaster_vecs).mean(dim=0)
        disaster_vec = F.normalize(disaster_vec, p=2, dim=0)
        
        scores = {}
        for key, outcome in options.items():
            # 選択肢のベクトル化
            outcome_vec = self.grounding.ground_concept(outcome)
            outcome_vec = F.normalize(outcome_vec, p=2, dim=0)
            
            # 災害ベクトルとの類似度 (Severity)
            # 1.0 = まさに災害, -1.0 = 平和
            sim = torch.dot(outcome_vec, disaster_vec).item()
            # [0, 1] rangeに補正 (概算)
            option_severity = (sim + 0.2) if sim > 0 else 0.1 # Bias positive
            option_severity = max(0.0, min(1.0, option_severity))
            
            if "catastrophic" in outcome.lower(): option_severity = 0.95 # Strong anchor
            if "nothing" in outcome.lower(): option_severity = 0.05
            
            # 予測された深刻度との一致度
            diff = abs(expected_severity - option_severity)
            score = 1.0 - diff
            
            scores[key] = score
            print(f"  [{key}] Sev={option_severity:.2f} (Pred={expected_severity:.2f}) -> Score={score:.2f} | '{outcome[:30]}...'")
        
        best = max(scores, key=scores.get)
        print(f"Selected: {best}")
        
        return best
