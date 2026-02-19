# CausalOS v5.3_full 技術レポート（A+B統合 / 単一Markdown）

> 対象ファイル: `CausalOS_v5_3_full.py`（添付ファイル）  
> BUILD_ID: `2026-02-18-v5.3_full+robustpack_v8plus_v11r4(cf_anchor+opts_debug+label_fix)`  
> 作成日時: 2026-02-19 12:21:10

---

## 0. TL;DR（要点だけ）

- **目的**: LLMに「反事実推論（counterfactual）」と「ハルシネーション抑制（迷い検知＋誘導）」を付与する実験的フレームワーク。
- **v5.3_full の核**:
  - 反事実の推論は **(Frame抽出→差分IR→do介入→局所化(Omega)→スコアリング)** のパイプラインで実行。
  - 多肢選択は 2系統のスコアラ：
    - **B2**: embeddingコントラスト `Sim(option, CF) - Sim(option, F)`
    - **B11**: Yes/No ログオッズ差分 + CFアンカー + 汎用性ペナルティ + relevanceスケーリング
  - 迷いが大きい/拮抗しているときに **Query B** が起動し、**prior（因果メモリ）**を自動提案→注入して再推論。
---

## 1. 設計思想（Design Principles）

### 1.1 ADD-ONLY（削除しない）
本実装は「概念・構成要素を削除しない」前提で、代わりに `inactive` や `disabled_prior` といった**無効化フラグ**を使って挙動を制御する。

### 1.2 定数スキーマ／定数基準（Task-agnostic）
v5.3_full は「キーワードに依存した意味分類」を避け、**固定スキーマと固定基準**で処理（Frame schema, Scoring criterion, QueryB gate）を回す。

---

## 2. モジュール構成（クラスの役割マップ）

このファイルは概ね次の層に分かれる（クラス名は実ファイルの定義に対応）。

### 2.1 I/O / 応答パケット
- `AnswerPacket`: 最終の回答テキスト、確信度、追加質問、トレースをまとめるデータクラス。

### 2.2 LLM関連（抽出・スコア）
- `FrameExtractorLLM`: 文章→Frame（entities/events/states/constraints）をJSONで抽出。
- `CausalTripletExtractor`: 文→因果triplet（cause/effect/magnitude）をLLMで抽出。
- `LikelyYesNoScorer_B11`: 多肢選択を Yes/No ログオッズで評価。
- `OptionScorer_B2`: embeddingコントラストで多肢選択を評価。

### 2.3 因果コア（数値モデル）
- `CausalCoreV5`: 2次元複素状態（real/imag）を持つ因果力学コア。`raw_S`/`raw_r`/`raw_phase` とマスクで伝播。
- `WorkspaceGate`: 推論で扱うノード集合（ワークスペース）を一時的に活性化する。
- `OmegaLocalizer`: 目標ノードへの影響が大きい部分グラフ（Omega）を局所化する。
- `ImpossibilityController`: 発散・スペクトル半径・飽和などから「不可能度 u」を推定する。

### 2.4 介入・差分・再構成
- `InterventionIR_B2`: factual/counterfactual Frame差分から介入操作列（IR）を生成。
- `AtomicMapper_B2`: IRを do演算（cut-in / clamp）へ変換してコアに適用。
- `ReconstructionChecker`: IRが counterfactual Frame を再現できているか（整合スコア）を算出。

### 2.5 記憶・正規化・根拠
- `ConceptBank`: ラベル→概念ID/ノードスロット割当（埋め込み類似でalias統合）。
- `VarNormalizer`: state.var の表記揺れを埋め込みで正規化。
- `GroundingChecker`: Frame要素が元テキストに根拠（重なり＋embedding）を持つか評価。
- `EdgeBank`: strong/prior のエッジ（因果メモリ）を保持。priorは QueryB で注入される。
- `ScaffoldProjector`: Frameから軽い構造（scaffold）をコアへ投影（任意で無効化可能）。

### 2.6 統合OS
- `UnifiedCausalOSV5_3Full`: 上記すべてを束ね、`answer_counterfactual_B2()` で反事実QAを実行する本体。

> 参考: 定義されているクラス一覧（抽出）

```
AnswerPacket, Retriever, Verifier, NullRetriever, NullVerifier, KnowledgePolicy, ConceptBank, VarNormalizer, GroundingChecker, EdgeBank, CausalCoreV5, WorkspaceGate, OmegaLocalizer, ImpossibilityController, CausalTripletExtractor, FrameExtractorLLM, InterventionIR_B2, AtomicMapper_B2, ScaffoldProjector, ReconstructionChecker, OptionScorer_B2, LikelyYesNoScorer_B11, PriorCandidateGenerator, UnifiedCausalOSV5_3Full
```

---

## 3. 全体フロー（answer_counterfactual_B2 パイプライン）

### 3.1 処理の流れ（概要）

```mermaid
flowchart TD
  A[Input: factual + counterfactual + options] --> B[extract_grounded(factual)]
  A --> C[extract_grounded(counterfactual)]
  B --> D[scaffold.project(f_frame)]
  C --> E[scaffold.project(c_frame)]
  D --> F[IR diff: InterventionIR_B2.diff_frames]
  E --> F
  F --> G[ReconstructionChecker: apply_ir + score]
  F --> H[Workspace nodes: collect entities/events/states]
  H --> I[OmegaLocalizer: factual rollout -> x_f]
  F --> J[AtomicMapper_B2.apply -> do interventions]
  J --> K[OmegaLocalizer: counterfactual rollout -> x_c]
  K --> L[ImpossibilityController: u_div/u_rho/u_cst -> u]
  I --> M[predicted_f states]
  K --> N[predicted_cf states]
  M --> O[Option scoring]
  N --> O
  O --> P{margin / IDS / generic / rel}
  P -->|OK| Q[Compose AnswerPacket]
  P -->|Trigger| R[Query B: propose priors -> inject -> rerun]
  R --> Q
```

### 3.2 パイプラインの “確定的な順序”
特にv5.3_fullでは、Frameの「抽出→強制→重複排除→採点」の順序が強制されている（ヘッダで明示）。

```
0001: # -*- coding: utf-8 -*-
0002: """
0003: CausalOS_v5_3_full.py (robustpack_v8 FULL)
0004: - Contrast option scoring (task-agnostic, constant criterion): Sim(option, CF) - Sim(option, F)
0005: - Query B trigger uses constant margin gate OR IDS: (margin < M_THR) OR (IDS >= IDS_THR)
0006: - prior_mask wiring: A_eff_mask = clamp(A_mask + prior_mask)
0007: - enforce restored: extract -> enforce -> dedup(inclusion) -> dedup(embedding) -> score(content-only)
0008: - ADD-ONLY philosophy: do not delete; use inactive flags, disabled_prior flags
0009: - No keyword-based semantic classification; everything uses fixed numeric criteria and constant schemas.
0010: """
0011: 
0012: from __future__ import annotations
```

---

## 4. Frame抽出とGrounding（根拠付け）

### 4.1 Frameスキーマ
`FrameExtractorLLM` はJSON形式で以下の構造を返す（entities/events/states/constraints/notes）。
- event: predicate（短い句）、args（role/value）、order、polarity（pos/neg）、modality
- state: var、subject、value、polarity、modality

この構造をベースに、以降の処理（差分IR、do介入、スコアリング）が進む。

### 4.2 Placeholder / schema-leak guard
`CAUSALOS_PLACEHOLDER_GUARD=1` のとき、`...` や `pos
neg` のようなテンプレ語を「不正」として扱い、Frameから除去/置換する。

### 4.3 GroundingChecker（content-only）
`GroundingChecker` は Frame要素を元テキストと照合し、
- 文字列包含（完全一致）
- トークン/バイグラムのoverlap
- embedding cosine
を混合して 0〜1 のスコアを作る。

`CAUSALOS_GROUND_CONTENT_ONLY=1` の場合、roleなどを除いた「内容語」中心で平均/最小を評価する。

### 4.4 Enforce（根拠が弱い要素の“差し替え”）
`_enforce_grounded_frame()` は、根拠スコアが閾値未満の場合に、
- atomic predicate を抽出
- span探索で元テキストから近い断片を選ぶ
- 最終的に元文へフォールバック
という段階的手当を行う。

```
2318: 
2319:         return best if (best_score > 0.08 and best) else None
2320: 
2321:     def _enforce_grounded_frame(self, frame: Dict[str, Any], source: str, kind: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
2322:         if os.environ.get("CAUSALOS_ENFORCE_GROUND", "1") != "1":
2323:             return frame, {"changed": 0, "details": []}
2324: 
2325:         thr = float(os.environ.get("CAUSALOS_ENFORCE_THR", "0.55"))
2326:         fr = copy.deepcopy(frame)
2327:         details = []
2328:         changed = 0
2329: 
2330:         def _act(d: Dict[str, Any]) -> bool:
2331:             if os.environ.get("CAUSALOS_IGNORE_INACTIVE", "1") == "1":
2332:                 return not bool(d.get("inactive", False))
2333:             return True
2334: 
2335:         for idx, e in enumerate(fr.get("events", []) or []):
2336:             if not (isinstance(e, dict) and _act(e)):
2337:                 continue
2338:             pred = _normalize_text(e.get("predicate", ""))
2339:             if not pred:
2340:                 continue
2341:             s = self.ground.score_item(pred, source)
2342:             if s >= thr:
2343:                 continue
2344:             ap = None
2345:             if os.environ.get("CAUSALOS_DEFALLBACK_ATOMIC", "1") == "1":
2346:                 ap = self.frames._extract_atomic_predicate(source, kind=kind)
```

---

## 5. Dedup（inactive化による重複排除）

### 5.1 Inclusion dedup
短い表現が長い表現に含まれる場合、短い側を `inactive=True` にする（inclusion）。

### 5.2 Embedding dedup
埋め込みcosineが `CAUSALOS_DEDUP_SIM_THR` 以上なら重複とみなし `inactive=True`。

```
2385: 
2386:     # ---------- dedup ----------
2387:     def _inactive_dedup_inclusion(self, frame: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
2388:         if os.environ.get("CAUSALOS_INACTIVE_DEDUP", "1") != "1":
2389:             return frame, {"changed": 0, "events": 0, "states": 0}
2390: 
2391:         fr = copy.deepcopy(frame)
2392:         changed = 0; de = 0; ds = 0
2393: 
2394:         def _act(d: Dict[str, Any]) -> bool:
2395:             return not bool(d.get("inactive", False))
2396: 
2397:         evs = [e for e in (fr.get("events", []) or []) if isinstance(e, dict)]
2398:         preds = [(i, _normalize_text(e.get("predicate", "")), _norm_label(e.get("predicate", ""))) for i, e in enumerate(evs) if _act(e)]
2399:         for i, pi, pli in preds:
2400:             if not pi:
2401:                 continue
2402:             for j, pj, plj in preds:
2403:                 if i == j or not pj:
2404:                     continue
2405:                 if pli and plj and pli in plj and len(pi) < len(pj):
2406:                     if _act(fr["events"][i]):
2407:                         fr["events"][i]["inactive"] = True
2408:                         fr["events"][i]["modality"] = (_normalize_text(fr["events"][i].get("modality", "")) + "|inactive_inclusion").strip()
2409:                         changed += 1; de += 1
2410: 
2411:         sts = [s for s in (fr.get("states", []) or []) if isinstance(s, dict)]
2412:         vals = []
2413:         for i, s in enumerate(sts):
2414:             if not _act(s):
2415:                 continue
```

---

## 6. 因果コア（CausalCoreV5）と prior_mask

### 6.1 状態と伝播
- 各ノード状態は2次元（real/imag）
- 有効結合は `S`（tanh(raw_S)）× `r`（sigmoid(raw_r)）× `A_mask`（接続マスク）× `G_gate`
- 位相 `raw_phase` と `omega*t` で回転を含む複素伝播

### 6.2 prior_mask の導入（A_maskへの加算）
prior（記憶）に基づくエッジ候補を `prior_mask` として作り、
**A_maskに足して clamp** することで、コアに「接続可能性」を与える。

```
0701: 
0702:         Aeff = self.A_mask
0703:         if prior_mask is not None:
0704:             Aeff = torch.clamp(Aeff + prior_mask, 0.0, 1.0)
0705: 
0706:         Aamp = Aeff * self.G_gate * S * r
```

### 6.3 prior_mask 生成ロジック
`_build_prior_mask(S_prior)` で
- |S_prior| >= `CAUSALOS_PRIOR_ABS_THR` の候補を抽出
- 上位 `CAUSALOS_PRIOR_TOPK` を選択
- その位置に prior_mask=1 を立てる

```
2128:     # ---------- prior_mask ----------
2129:     def _build_prior_mask(self, S_prior: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Dict[str, int]]:
2130:         if S_prior is None:
2131:             return None, {"nonzero": 0, "topk": 0, "added_to_A": 0}
2132: 
2133:         abs_thr = float(os.environ.get("CAUSALOS_PRIOR_ABS_THR", "0.01"))
2134:         topk = int(os.environ.get("CAUSALOS_PRIOR_TOPK", "64"))
2135:         abs_thr = float(max(0.0, abs_thr))
2136:         topk = int(max(0, topk))
2137: 
2138:         A = self.core.A_mask.detach()
2139:         Sp = S_prior.detach()
2140:         absSp = Sp.abs()
2141:         n = Sp.shape[0]
2142: 
2143:         mask_cand = (absSp >= abs_thr)
2144:         diag = torch.eye(n, device=Sp.device, dtype=torch.bool)
2145:         mask_cand = mask_cand & (~diag)
2146: 
2147:         idx = torch.nonzero(mask_cand, as_tuple=False)
2148:         nonzero = int(idx.shape[0])
2149:         if nonzero == 0 or topk == 0:
2150:             return None, {"nonzero": nonzero, "topk": 0, "added_to_A": 0}
2151: 
2152:         vals = absSp[mask_cand]
2153:         k = min(topk, vals.numel())
2154:         top_vals, top_pos = torch.topk(vals.view(-1), k=k)
2155: 
2156:         idx_list = idx.tolist()
2157:         chosen = [idx_list[p] for p in top_pos.tolist()]
2158: 
2159:         prior_mask = torch.zeros_like(Sp)
2160:         for j, i in chosen:
2161:             prior_mask[j, i] = 1.0
2162: 
2163:         added_to_A = int((prior_mask.bool() & (A == 0.0).bool()).sum().item())
2164:         return prior_mask, {"nonzero": nonzero, "topk": k, "added_to_A": added_to_A}
2165: 
2166:     # ---------- ingest_context ----------
```

---

## 7. OmegaLocalizer と ImpossibilityController

### 7.1 OmegaLocalizer（局所化）
`OmegaLocalizer.localize()` は
- rollout して target ノードの損失を定義
- raw_S 勾配や寄与、到達可能性を混ぜて重要エッジを抽出
- OmegaA_nodes（関与ノード集合）と Omega_edges を返す

### 7.2 Impossibility（不可能度 u）
counterfactual rollout から
- 局所発散（energy増加）
- Omega局所のスペクトル半径リスク
- 飽和/NaN/inf
を評価し `u in [0,1]` を作る。

この `u` は確信度推定（confidence）にも組み込まれる。

---

## 8. 介入の表現：IR（差分）→ do演算

### 8.1 InterventionIR_B2（Frame差分）
Frameの差分から以下の操作列を作る：
- `SET_STATE` / `UNSET_STATE`
- `ADD_EVENT` / `REMOVE_EVENT`
- `MODALITY`
- 何もなければ `NOOP`

### 8.2 AtomicMapper_B2（do介入）
IRを
- `apply_do_cut_in(node)`（流入遮断）
- `apply_do_value(node, v_real, v_imag)`（値をクランプ）
へ落とし込む。

state.value は LLM embedding → 2D射影（W）→ tanh で2次元値に変換される。

---

## 9. 多肢選択スコアリング（Option scoring）

v5.3_fullでは2系統。

### 9.1 OptionScorer_B2（embeddingコントラスト）
- predicted_cf / predicted_f を要約→埋め込み化
- option文をFrame化→埋め込み化
- **Sim(CF, option) - Sim(F, option)** を基本スコアとする
- さらに scenario relevance と ops signature alignment を乗算で調整

### 9.2 LikelyYesNoScorer_B11（Yes/No ログオッズ）
コメントで式が明示されている：

```
1704: # ==========================================================
1705: # LikelyYesNoScorer_B11 (task-agnostic, constant criterion)
1706: # - Score(option) = Lik(CF, option) - Lik(F, option) - λ * max(0, Lik(EMPTY, option))
1707: # - Lik(world, option) = logP(Yes|prompt) - logP(No|prompt)
1708: # - Relevance scaling: score *= clamp((1-w)+w*Rel, floor, 1)
1709: # - Prior signature appended to WORLD so QueryB priors can affect scoring
1710: # ==========================================================
```

実装上はさらに
- **CFアンカー**: `score += cf_w * lik_cf`
- **汎用性ペナルティ**: EMPTY world に対して Yes に寄りすぎる option を減点
- **relevanceスケール**: `score *= clamp((1-w)+w*Rel, floor, 1)`
を含む。

### 9.3 OPTS ログの意味（あなたが見ている行）
ログ `OPTS:` は、各選択肢ラベルについて以下を1行で出す：
- `sc`: 最終スコア
- `rel`: relevance（シナリオ関連度）
- `gen`: generic-positiveness（汎用性ペナルティの正の部分）
- `lik_cf`: counterfactual worldでのYes/Noログオッズ
- `lik_f`: factual worldでのYes/Noログオッズ
- `cfT`: CFアンカー項（= cf_w * lik_cf）

```
2954:                     lik_f = float(pr.get('lik_f', 0.0))
2955:                     genp = float(pr.get('gen_pos', 0.0))
2956:                     relv = float(pr.get('rel', 0.0))
2957:                     cfterm = float(pr.get('cf_term', 0.0))
2958:                     items.append(f"{lab}:sc={sc:.3f},rel={relv:.2f},gen={genp:.2f},lik_cf={lik_cf:.2f},lik_f={lik_f:.2f},cfT={cfterm:.2f}")
2959:                 lines.append('OPTS: ' + ' | '.join(items)[:900])
```

---

## 10. Query B（prior自動提案→注入→再推論）

### 10.1 Trigger条件
Query B は、**定数ゲート**として
- marginが小さい（拮抗）
- IDSが大きい（不確実）
- genericが強い
- relevanceが弱い
などで起動する。

```
2833:         )
2834:         qb_info = {"triggered": False, "ids": ids, "added": 0, "edges": [], "margin_now": margin_now, "m_thr": m_thr}
2835: 
2836:         if enable_qb and budget > 0 and (margin_now < m_thr or ids >= ids_thr or best_genpos_now > gen_thr or best_rel_now < rel_thr):
2837:             if beta <= 0.0:
2838:                 beta = beta_min
2839:             def active_event_texts(fr):
```

### 10.2 IDS（Inconsistency / Instability Diagnostic Score）
`_compute_ids()` は
- margin
- grounding min
- density
- coverage
- impossibility u
を固定重みで合成し 0〜1 に正規化する。

### 10.3 PriorCandidateGenerator → inject
Query B は、候補ノード集合（cause/effect candidates）を作ってLLMに
「もっともらしい prior edges」を提案させ、
`_inject_prior_edges()` で `EdgeBank.prior` に重み付きで注入する。

注入後、`S_prior` と `prior_mask` を更新して再推論し、
`0.6*margin + 0.4*confidence` の固定基準で良い方を採用する。

---

## 11. 確信度（confidence）の作り方（要点）

`_confidence()` は
- 1-u（安定性）
- targetベクトルの大きさ（反応の強さ）
- recon_score（IRがCF Frameを再現できているか）
- grounding平均
- option margin（あれば）
- placeholder_ratio / density
を掛け合わせて 0〜1 にクリップする。

このため、
- **uが高い（不可能っぽい）**
- **groundingが弱い**
- **Frameが薄い（density低い）**
- **選択肢が拮抗（margin小さい）**
ほど確信度は下がる。

---

## 12. 設定（環境変数）リファレンス

> このセクションは「run_demo.shで触っていたつまみ」が、コード内のどこに効くかを追えるようにするための索引です。

### Frame / Grounding
- `CAUSALOS_NO_LLM_FRAME` (used around L1239)- `CAUSALOS_FRAME_STRICT_MAX` (used around L1672)- `CAUSALOS_GROUND_RETRY` (used around L2553)- `CAUSALOS_GROUND_THR` (used around L2552)- `CAUSALOS_GROUND_TOKEN_OVERLAP` (used around L514)- `CAUSALOS_GROUND_CONTENT_ONLY` (used around L574)- `CAUSALOS_PLACEHOLDER_GUARD` (used around L196)- `CAUSALOS_STATE_FALLBACK` (used around L1192)- `CAUSALOS_DEFALLBACK_ATOMIC` (used around L1269)- `CAUSALOS_TARGET_FALLBACK` (used around L2665)### Enforce / Span selection
- `CAUSALOS_ENFORCE_GROUND` (used around L2322)- `CAUSALOS_ENFORCE_THR` (used around L2325)- `CAUSALOS_SPAN_MIN_TOK` (used around L2284)- `CAUSALOS_SPAN_MAX_TOK` (used around L2285)- `CAUSALOS_SPAN_SPECIFICITY` (used around L2253)### Dedup / Inactive management
- `CAUSALOS_INACTIVE_DEDUP` (used around L2388)- `CAUSALOS_IGNORE_INACTIVE` (used around L208)- `CAUSALOS_DEDUP_SIM_THR` (used around L2441)### Option scoring (contrast embedding)
- `CAUSALOS_OPT_MODE` (used around L1646)- `CAUSALOS_OPT_SCENARIO_REL` (used around L1662)- `CAUSALOS_OPT_SCENARIO_W` (used around L1663)- `CAUSALOS_OPT_SCENARIO_EMB` (used around L1664)- `CAUSALOS_OPT_OPS_ALIGN` (used around L1667)- `CAUSALOS_OPT_OPS_W` (used around L1668)- `CAUSALOS_OPT_MIN_MARGIN` (used around L2555)- `CAUSALOS_REL_COMB` (used around L1628)- `CAUSALOS_REL_EMB_W` (used around L1631)### Option scoring (Likely Yes/No)
- `CAUSALOS_OPT_SCORER` (used around L2732)- `CAUSALOS_ENTAIL_YES` (used around L1804)- `CAUSALOS_ENTAIL_NO` (used around L1805)- `CAUSALOS_LIKELY_CF_W` (used around L1880)- `CAUSALOS_GENERIC_PENALTY` (used around L1853)- `CAUSALOS_GENERIC_LAMBDA` (used around L1854)- `CAUSALOS_LIKELY_REL` (used around L1857)- `CAUSALOS_LIKELY_REL_W` (used around L1858)- `CAUSALOS_LIKELY_REL_FLOOR` (used around L1859)- `CAUSALOS_PRIOR_SIG_MAX` (used around L1834)### Query B / IDS
- `CAUSALOS_ENABLE_QUERY_B` (used around L2818)- `CAUSALOS_QUERY_B_BUDGET` (used around L2817)- `CAUSALOS_QB_MARGIN_THR` (used around L2819)- `CAUSALOS_IDS_THR` (used around L2816)- `CAUSALOS_IDS_MARGIN_REF` (used around L2494)- `CAUSALOS_QB_GEN_THR` (used around L2820)- `CAUSALOS_QB_REL_THR` (used around L2821)- `CAUSALOS_QB_BETA_MIN` (used around L2822)### Priors / prior_mask
- `CAUSALOS_PRIOR_ABS_THR` (used around L2133)- `CAUSALOS_PRIOR_TOPK` (used around L2134)- `CAUSALOS_PRIOR_BASE_W` (used around L2508)- `CAUSALOS_PRIOR_W_MAX` (used around L2509)- `CAUSALOS_DISABLE_SCAFFOLD` (used around L1444)### Debug / Trace
- `CAUSALOS_DEBUG_FRAME_RAW` (used around L1065)- `CAUSALOS_TRACE_FRAMES` (used around L2997)### Modes
- `CAUSALOS_ENABLE_FACT_MODE` (used around L299)- `CAUSALOS_NO_LLM_GRAPH` (used around L1006)- `CAUSALOS_LATENT_OPT` (used around L2715)

---

## 13. ログとデバッグの読み方

### 13.1 grepしやすい出力
- `OPTS:` 1行に候補別の内訳を出す（score/relevance/generic/lik_cf/lik_f/cfT）。
- `PriorMask:` nonzero/topk/added_to_A を出す。
- `IDS` と `QB` の発火・追加本数を出す。

### 13.2 Frameのhead trace
`CAUSALOS_TRACE_FRAMES=1` のとき、AnswerPacket.trace に `frames_head` が入り、
entities/events/statesの要約を見られる。

---

## 14. 研究・評価のための観点（提案）

1. **スコアラ比較**: `CAUSALOS_OPT_SCORER=likely_yesno` と `contrast` を切り替え、
   - OPTSの分離（margin）
   - 誤答の型（generic勝ち/関連語に引っ張られる）
   を比較。
2. **Query B の寄与**: `CAUSALOS_ENABLE_QUERY_B=0/1` で
   - margin改善
   - IDS改善
   - prior_mask added_to_A の増加
   を追う。
3. **Grounding強化の副作用**: `CAUSALOS_ENFORCE_THR` を上下させ、
   - 無理なspan置換による意味崩れ
   - placeholder_ratio の変化
   を観察。
4. **prior_mask の上位K**: `CAUSALOS_PRIOR_TOPK` を変えて
   - 収束性（u）
   - 推論の安定性
   を確認。

---

## 付録A. 重要スニペット集（原文）

### A.1 ヘッダ（設計要点の宣言）
```
0001: # -*- coding: utf-8 -*-
0002: """
0003: CausalOS_v5_3_full.py (robustpack_v8 FULL)
0004: - Contrast option scoring (task-agnostic, constant criterion): Sim(option, CF) - Sim(option, F)
0005: - Query B trigger uses constant margin gate OR IDS: (margin < M_THR) OR (IDS >= IDS_THR)
0006: - prior_mask wiring: A_eff_mask = clamp(A_mask + prior_mask)
0007: - enforce restored: extract -> enforce -> dedup(inclusion) -> dedup(embedding) -> score(content-only)
0008: - ADD-ONLY philosophy: do not delete; use inactive flags, disabled_prior flags
0009: - No keyword-based semantic classification; everything uses fixed numeric criteria and constant schemas.
0010: """
0011: 
0012: from __future__ import annotations
```

### A.2 prior_mask のA_mask加算
```
0701: 
0702:         Aeff = self.A_mask
0703:         if prior_mask is not None:
0704:             Aeff = torch.clamp(Aeff + prior_mask, 0.0, 1.0)
0705: 
0706:         Aamp = Aeff * self.G_gate * S * r
```

### A.3 QueryBトリガ条件
```
2833:         )
2834:         qb_info = {"triggered": False, "ids": ids, "added": 0, "edges": [], "margin_now": margin_now, "m_thr": m_thr}
2835: 
2836:         if enable_qb and budget > 0 and (margin_now < m_thr or ids >= ids_thr or best_genpos_now > gen_thr or best_rel_now < rel_thr):
2837:             if beta <= 0.0:
2838:                 beta = beta_min
2839:             def active_event_texts(fr):
```

### A.4 OPTSログ生成部
```
2954:                     lik_f = float(pr.get('lik_f', 0.0))
2955:                     genp = float(pr.get('gen_pos', 0.0))
2956:                     relv = float(pr.get('rel', 0.0))
2957:                     cfterm = float(pr.get('cf_term', 0.0))
2958:                     items.append(f"{lab}:sc={sc:.3f},rel={relv:.2f},gen={genp:.2f},lik_cf={lik_cf:.2f},lik_f={lik_f:.2f},cfT={cfterm:.2f}")
2959:                 lines.append('OPTS: ' + ' | '.join(items)[:900])
```

---

## 付録B. “次に欲しい情報”

このレポートは `CausalOS_v5_3_full.py` 単体から復元している。
もしリポジトリ側の `CausalChatAgent.py` も共有できれば、
- 入力パース（factual/cf/options分解）
- バッチ実行（ベンチ）
- ログ整形
まで含めた **エンドツーエンドの運用レポート**に拡張できる。
