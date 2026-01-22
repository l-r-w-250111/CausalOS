# CausalOS - LLM 因果推論・ハルシネーション防止フレームワーク

CausalOSは、LLM（大規模言語モデル）に因果推論能力を付与し、ハルシネーションの防止、反実推論の精度向上、および動的な計画遂行を実現するための実験的フレームワークです。

## 主な機能

1.  **迷い（Hesitation）の検知**:
    - LLMの出力ロジットから、エントロピー、Top-1確率、尖度（Kurtosis）、位相（Phi）、慣性（CII）などの指標をリアルタイムで算出します。
    - これらの指標により、モデルが「統計的な確信度」を失っているタイミング（ハルシネーションが起きやすい瞬間）を特定します。

2.  **S行列（Scattering Matrix）による剛性誘導**:
    - 事実情報（論文名、固有名詞、定義など）を「剛性の高いトークン列」として登録します。
    - LLMに「迷い」が生じた際、S行列に基づいてロジット（logits）を調整し、正しい事実（因果のレール）へと出力を誘導します。

3.  **動的な因果ノード抽出と介入ポイントの特定**:
    - `CausalNodeExtractor` モジュールにより、テキストから主要なエンティティ（名詞）やアクション（動詞）を動的に抽出します。
    - 事実（Factual）と反事実（Counterfactual）の文章を比較し、どの単語が置き換わったか（介入ポイント）を自動的に特定します。

4.  **介入の現象分類と原子操作 (Causal Primitives)**:
    - 介入の種類（置換、否定、強度、時制、不可能）を検知し、物理パラメータ（$S, r, \phi, \omega$, インデックス置換）への原子操作にマッピングします。
    - 以下の数式に基づいた動的な因果推論を可能にします：
      $$x_{j}(t+1) = \sigma \left( \sum_{i} S_{ij} \cdot r_{ij} e^{i(\phi_{ij} + \omega t)} x_{i}(t) \right)$$

| 現象分類 (Phenomenology) | 原子操作 (Atomic) | 物理パラメータ | 物理的帰結 |
| :--- | :--- | :--- | :--- |
| 置換 (A → B) | Delete & Insert | $S_{ij} \to 0, S_{ik} \to 1$ | 接続トポロジーの書き換え |
| 否定 (Not A) | Invert | $\phi \to \phi + \pi$ | 干渉による信号の相殺 |
| 強度変化 (More/Less) | Scale | $r \to r \times \alpha$ | 伝播振幅の増減 |
| 時制・順序 (Before/After) | Delay / Shift | $\omega$ または $t$ | 因果連鎖の到達時間変容 |
| 不可能性 (Logic Error) | Permute | $S$ の非エルミート化 | エネルギー発散・不安定化 |

5.  **セマンティック論理チェックによるハイブリッド評価**:
    - 物理シミュレーションに加え、LLMによる論理判断（Yes/No）を統合。
    - 反事実シナリオにおける各選択肢の整合性をスコアリングし、最も妥当な回答を選択します。

6.  **数値的安定性の確保**:
    - `float16` 環境での計算誤差による `nan` 発生を防ぐため、内部的な指標計算（エントロピーや位相）は `float32` で実行されます。

## セットアップ

### 必要条件

- Python 3.10+
- PyTorch
- Transformers
- Accelerate
- NumPy

### インストール

```bash
pip install torch transformers accelerate numpy
```

## 使い方

### 1. ハルシネーション防止（論文名検索の例）

`test_paper_hallucination.py` を実行すると、S行列を用いて特定の論文名を正確に出力させるデモを確認できます。

```bash
python3 test_paper_hallucination.py
```

**仕組み**:
- 「Attention Is All You Need」というタイトルをS行列に登録。
- 生成中にモデルの迷い（CIIのスパイク等）を検知すると、S行列が介入し、正しいトークンを選択させます。

### 2. 反実推論ベンチマーク（CRASS）

`test_crass.py` を実行すると、動的なノード抽出と論理チェックを用いた反実推論のテストを行えます。

```bash
python3 test_crass.py
```

**新機能：セマンティック論理判定**
CRASSのような多肢選択式問題に対し、以下のステップで解答します：
1. `Factual` と `Counterfactual` からノードを抽出し、介入（何が何に変わったか）を特定。
2. 各選択肢（Option）について、「シナリオ：AがBに変わった場合、結果はCになる。これは論理的か？」というプロンプトをLLMに与え、その応答を評価。
3. すべてが否定された場合、"Nothing special" などの「変化なし」を優先的に選択。

## 主要クラス・メソッドの使い方

### SMatrixEngine (ハルシネーション防止)
特定の事実をモデルに「記憶」させ、誘導するために使用します。

```python
# 事実トークン列の登録
paper_title = "Attention Is All You Need"
token_ids = osys.observer.tokenizer.encode(paper_title, add_special_tokens=False)
osys.s_matrix.register_sequence(token_ids, rigidity=100.0)

# 因果チェック付き生成
# 迷いを検知すると自動的にS行列が適用されます
output = osys.generate_with_causal_check(prompt, max_new_tokens=20)
```

### solve_counterfactual (反実推論)
シナリオと選択肢を渡すと、因果的な妥当性を評価して最適な回答を返します。

```python
factual = "A girl eats an apple."
counterfactual = "What would have happened if the girl had eaten a stone?"
options = {
    "A": "She would have been happy.",
    "B": "She would have broken her teeth.",
    "C": "She would have felt full."
}

answer = osys.solve_counterfactual(factual, counterfactual, options=options)
# answer -> "B"
```

## ファイル構成

- `CausalOS_v0.py`: フレームワークの本体（Observer, Core, SMatrixEngine, UnifiedOS）。
- `causal_node_extractor.py`: LLMを用いた動的ノード抽出モジュール。
- `test_crass.py`: CRASSベンチマーク用テストスクリプト。
- `test_paper_hallucination.py`: S行列によるハルシネーション防止テストスクリプト。

## 詳細設定

`CausalOS_v0.py` 内の `LLMCausalObserver` で使用するモデル ID を変更できます（デフォルトは `Qwen/Qwen2.5-7B`）。ローカル環境のリソースに合わせて調整してください。

```python
# CausalOS_v0.py
class LLMCausalObserver:
    def __init__(self, model_id="Qwen/Qwen2.5-7B"):
        ...
```

## ライセンス

研究・実験目的のプロトタイプとして提供されています。
