# CausalOS v4 - Web Knowledge Integration

## 概要

CausalOS v4に**APIキー不要のWeb検索機能**を追加しました。これにより、リアルタイムの事実確認とハルシネーション防止が可能になります。

## 主な機能

### 1. Web検索（DuckDuckGo）
- APIキー不要
- 自動キャッシング
- 高速な検索結果取得

### 2. 事実検証
- LLMベースの検証
- キーワードベースのフォールバック
- 信頼度スコアリング

### 3. S行列への自動登録
- 検証済みの事実をS行列に固定
- ハルシネーション抑制

## インストール

### 必須依存関係

```bash
pip install duckduckgo-search
```

または：

```bash
pip install -r requirements_web_knowledge.txt
```

## 使い方

### 基本的な使用

```python
from CausalOS_v4 import UnifiedCausalOSV4

# Web知識統合を有効化
osys = UnifiedCausalOSV4(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    use_premise_aware=True,      # 前提条件管理
    use_web_knowledge=True       # Web知識統合
)
```

### 事実確認付き反事実推論

```python
result = osys.knowledge_augmented.solve_counterfactual_with_facts(
    factual="Python is a programming language created by Guido van Rossum",
    counterfactual="Python is a snake",
    options={
        "A": "It would be dangerous",
        "B": "The meaning would be different",
        "C": "Nothing would change"
    },
    verify_facts=True  # Web検索で事実を確認
)

print(f"Selected: {result['selected_option']}")
print(f"Verified facts: {len(result['verified_facts'])}")
```

### Web検索単体での使用

```python
from WebKnowledgeRetriever import WebKnowledgeRetriever

retriever = WebKnowledgeRetriever()

# Web検索
results = retriever.search("Python programming language", max_results=3)
for r in results:
    print(f"{r['title']}: {r['snippet'][:100]}...")

# 事実確認
is_verified, evidence, confidence = retriever.verify_fact(
    "Python was created by Guido van Rossum",
    use_llm=False  # キーワードベース
)
print(f"Verified: {is_verified}, Confidence: {confidence:.2f}")
```

### 生成時の事実チェック

```python
result = osys.knowledge_augmented.generate_with_fact_checking(
    prompt="Tell me about Python programming language",
    max_tokens=100,
    check_interval=10  # 10トークンごとに事実確認
)
```

## テスト

テストスクリプトを実行：

```bash
python test_web_knowledge.py
```

テスト内容：
1. 基本的なWeb検索
2. 事実確認機能
3. キャッシュ機構
4. CausalOS統合

## アーキテクチャ

```
WebKnowledgeRetriever
├── search()                  # DuckDuckGo検索
├── verify_fact()             # 事実検証
├── extract_factual_claims()  # 事実抽出
└── _update_cache()           # LRUキャッシュ

KnowledgeAugmentedCausalOS
├── solve_counterfactual_with_facts()  # 事実確認付き推論
├── generate_with_fact_checking()       # 生成時の事実チェック
└── get_fact_statistics()               # 統計情報

CausalOS_v4
└── __init__(use_web_knowledge=True)   # 統合
```

## 制限事項

1. **DuckDuckGo検索の制限**
   - レート制限あり（過度な検索は避ける）
   - 検索結果は3-5件に制限

2. **LLM使用時**
   - 事実検証にモデルのリソースを使用
   - 小さいモデルでは精度が低い可能性

3. **インターネット接続**
   - オフラインでは動作しない
   - キャッシュがあれば一部動作可能

## トラブルシューティング

### `duckduckgo-search not found`

```bash
pip install duckduckgo-search
```

### 検索が遅い

- キャッシュが有効か確認
- `max_results`を減らす（デフォルト: 3）

### 事実確認の精度が低い

- `use_llm=True`を使用（より正確だが遅い）
- より大きなモデルを使用

## 次のステップ

この実装を基に、以下の拡張が可能です：

1. **複数検索エンジンの統合** - Google、Bingなど
2. **Wikipedia API統合** - より信頼性の高いソース
3. **文献検索** - PubMed、arXivなど
4. **時系列データ** - ニュースAPI統合

## ライセンス

Apache 2.0 License
