# Auto Agent Package

`auto_agent/` は、Gmail を起点にメールを監視し、AI でタスク化し、必要に応じて人間承認を挟んで実行するためのひな形です。

## 構成

```text
auto_agent/
├── __init__.py
├── api.py                  # 承認リンク / API 受付
├── config.py               # .env から設定を読み込む
├── main.py                 # Gmail ポーリングのエントリーポイント
├── models.py               # メール / タスク / 承認のデータモデル
├── orchestrator.py         # フロー全体の司令塔
├── clients/
│   ├── ai_analyzer.py      # OpenAI によるタスク案生成
│   ├── gmail_client.py     # Gmail API の読み取り
│   └── notifier.py         # Slack Webhook 通知
├── services/
│   ├── approval_service.py # 承認後の状態遷移と実行
│   └── executor.py         # 自動実行関数のレジストリ
└── storage/
    └── pending_store.py    # 保留タスクのローカル JSON 保存
```

## フロー

1. `python -m auto_agent.main` が Gmail を検索します。
2. `support@outlier.ai` などの条件に合うメールを取得します。
3. `ai_analyzer.py` がタスク案を JSON として返します。
4. 低リスクで確実なものは `executor.py` に渡して自動実行します。
5. 微妙なタスクは `pending_approvals.json` に保存し、Slack 通知を送ります。
6. 人間が承認リンクまたは API を叩くと `api.py` 経由で後続実行されます。

## 初期セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r auto_agent/requirements.txt
cp .env.example .env
```

Gmail 連携では、Google Cloud で OAuth クライアントを作成して `secrets/gmail_credentials.json` を配置してください。初回実行時に `secrets/gmail_token.json` が生成されます。

## 起動方法

監視を 1 回だけ実行:

```bash
python -m auto_agent.main
```

監視を継続実行:

```bash
python -m auto_agent.main --loop
```

承認 API:

```bash
uvicorn auto_agent.api:app --reload
```

## 実装メモ

- Slack は Incoming Webhook の Block Kit で通知します。
- ワンクリック承認は `APPROVAL_BASE_URL` と `APPROVAL_TOKEN` を使う簡易版です。
- 本番では Slack App の interactive endpoint や LINE Messaging API に差し替える前提です。
- 実行関数は `services/executor.py` の `handlers` に追加してください。
