# REMflow + CausalOS 統合フレームワーク（RAG / Web検索 / Fine-tuning / 多バックエンド推論）

このリポジトリは、**RAG（短期記憶）**・**Web検索**・**LoRA学習（CPT/SFT）**・**複数推論バックエンド**（Ollama / vLLM / Unsloth / CausalOS-Transformers）を、**Docker Compose + Streamlit UI**で一体運用する実験基盤です。

加えて **CausalOS（因果推論・ハルシネーション防止）** を統合し、**Verified/Exact（根拠優先/抽出優先）**、**URL捏造防止（post-check）**、**S行列（因果メモリ）**、**trust_remote_code トグル**等を備えます。

---
## 1. 目的
- **エンドツーエンド**：データ準備 → CPT/SFT（LoRA） → 変換（GGUF/AWQ） → 推論（チャット）までを1つのUIで扱う。
- **RAG + Web検索**：社内/個人ドキュメントと外部情報を切り替え、根拠を付けて応答する。
- **CausalOS統合**：因果メモリ（S行列）とガード（Verified/Exact, post-check）により、固有名詞/URL/論文等の捏造を抑制する。

---
## 2. 主要機能（概要）
### 2.1 Streamlit UI（統合コンソール）
- チャットUI：モデルと対話（RAG / Web検索 自動切替・手動切替）
- 学習ワークフロー：Step1〜4（DL → CPT → SFT → 変換/デプロイ）
- セッション管理：チャット履歴の保存と再利用（SFTデータ化にも利用）

### 2.2 推論バックエンド
- **Ollama**：GGUFモデルの推論・Embedding（GPU/CPU）
- **vLLM**：高スループット推論（AWQ / bitsandbytes 量子化対応）
- **Unsloth Server**：軽量サーバでの推論・学習系タスク用
- **CausalOS / Transformers（PyTorch）**：HFモデルを直接ロードし、quant（4bit/8bit/none）と trust_remote_code に対応

### 2.3 学習（CPT / SFT）
- **CPT**：生テキストで継続事前学習（LoRA）
- **SFT**：指示追従・対話形式を学習（LoRA）
- 生成される学習コマンドをターミナルで実行してUIフリーズを回避

### 2.4 変換・デプロイ
- **GGUF**：Ollama向けに変換（量子化方式指定）
- **AWQ**：vLLM向けに量子化（保存先指定）
- GGUFをOllamaへデプロイするUI機能

### 2.5 CausalOS（因果推論/ハルシネーション防止）
- **迷い（Hesitation）検知**：entropy / top1 / kurtosis / phi / CII 等（設計コンセプト）
- **S行列（Scattering Matrix）**：剛性の高いトークン列・因果レールにより出力を誘導（設計コンセプト）
- **Verified/Exact**：根拠優先・抽出優先の応答モード（実装）
- **post-check**：検索結果集合に存在しないURLの出力を除去（実装）
- **S行列ストア**：`./storage/s_matrix.json`（nodes/edges/groups/commits、複素重み雛形 {re,im}、maskメタ）

---
## 3. 現在の実装状態（会話で反映済みの内容）
### 3.1 追加UI/制御
- **Answer Mode**：Assist / Verified / Exact
- **Routing Policy**：Auto / WEB / RAG / NONE
- **Debug表示**：通常は内部を表示せず、debugオンでルーティング/ソースを表示

### 3.2 ルーティング改善
- 強制WEB：論文/著者/DOI/URL/根拠要求などを検知した場合にWeb検索を優先
- ルータ出力を **JSON固定**に寄せ、壊れた出力でも正規化して解釈

### 3.3 安全ガード
- Web検索結果を可能な限り構造化（title/url/snippet）してSOURCESとして扱う
- **post-check**：回答内URLが検索結果集合に無ければ `[UNVERIFIED_URL_REMOVED]` に置換
- Verified/Exact で根拠が弱い場合は **「不明（ソースから確定できません）」**へ落とす（ハルシネ抑制）
- 一部モデルが出す `<think>...</think>` 等をUI表示前に除去する **サニタイズ**（実装方針）

---
## 4. 現在の実装で使用される主なファイル（ファイル名一覧）
> 「実行時に参照/実行される」「UIから呼び出される」ものを中心に列挙します。

### 4.1 エントリーポイント / UI
- `app.py`：Streamlit UI本体（チャット、RAG管理、学習ワークフロー、各バックエンド制御）

### 4.2 RAG / Web検索
- `rag_handler.py`：RAGの初期化・ドキュメント追加・retriever提供
- `web_search.py`：DuckDuckGo（ddgs）検索（現状は整形文字列を返す）

### 4.3 CausalOS（Transformers）
- `CausalOS_v5_3_full.py`：`UnifiedCausalOSV5_3Full`（quant / trust_remote_code を含むロード対象）

### 4.4 学習/データ/変換（UI Step1-4 から呼び出し）
- `download_model.py`：Hugging FaceからベースモデルをDL
- `create_dataset.py`：CPT/SFTデータセット生成（chat履歴やファイルを加工）
- `train_lora.py`：LoRA学習（Unsloth側）
- `train_trl.py`：LoRA学習（TRL側）
- `convert_to_gguf.py`：GGUF変換（Ollama向け）
- `quantize_to_awq.py`：AWQ量子化（vLLM向け）
- `unsloth_server.py`：Unsloth推論サーバ（uvicorn）

### 4.5 生成物/永続化（実行時に生成・参照される）
- `chat_history.json`：チャット履歴（セッション保存）
- `./storage/s_matrix.json`：S行列ストア（nodes/edges/groups/commits）
- `./data/`：RAG投入ファイル、CPT/SFTデータ、生成データセット
- `./base_models/`：ベースモデル格納
- `./lora_models/`：LoRAアダプタ格納
- `./gguf_models/`：GGUF出力
- `./awq_models/`：AWQ出力
- `./datasets/`：生成データセット出力

### 4.6 コンテナ/設定
- `docker-compose.yml`（または同等のCompose設定）：ui / unsloth / vllm / ollama-gpu 等を起動
- `.env` / `.env.example`：環境設定（URLやポート等）
- `ocker-compose.mamba-builder.yml` ui にmamba-ssmを導入するためのwhl生成用  
---
## 5. セットアップ（クイック）
### 5.1 必要要件
- Docker / Docker Compose
- NVIDIA GPU + CUDA（推奨）

### 5.2 起動
```bash
cp .env.example .env   # 任意
docker compose up --build -d
```
UI: `http://localhost:8501`

---
## 6. 使い方（UIガイド）
### 6.1 推論（チャット）
1. サイドバーで Inference Engine を選択（Ollama / Unsloth / vLLM / CausalOS）
2. 必要に応じてモデル/サーバを起動
3. チャット入力
4. Answer Mode と Routing Policy を調整（根拠重視なら Verified/Exact + Auto/WEB 推奨）

### 6.2 学習（Step1-4）
- Step1: HFからベースモデルDL
- Step2: CPT（任意）
- Step3: SFT（任意）
- Step4: GGUF/AWQ変換・デプロイ

---
## 7. トラブルシュート（要点）
- **固有名詞/論文/URL**：Verified/Exactにすると“不明”になりやすいが、ハルシネ抑制として正常。ソースが必要。
- **trust_remote_code**：Qwen3等で必要になる場合あり。信頼できるモデルにのみON。
- **Web検索結果が一般論に寄る**：クエリ改善、あるいは `web_search.py` を構造化返却へ移行すると改善。

---
## 8. ロードマップ（次にやること）
- 発明ができるAIにする
-- 因果を扱えること
-- 自律成長できること
-- 発想の飛躍・視点転換ができること

