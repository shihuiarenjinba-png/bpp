# Finance App Collection

複数の金融アプリを `Streamlit` で切り替えて使える構成です。`bpp` 単体でも、ルートの [`app.py`](/Users/mitsuyoshitsuha/Documents/New%20project/app.py) から各研究画面へ入れるようにしています。

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

左のナビゲーションから、各アプリを切り替えて使えます。

## Vercel

Vercel では Streamlit 自体を直接ホストする構成ではなく、依存の重い Python 実行環境を避けるため、静的な `index.html` を案内ページとして返すようにしています。
そのため、Vercel デプロイ先では各分析アプリの概要とコード導線を確認でき、実際の分析画面はローカルの `streamlit run app.py` で使う想定です。

## Apps

- `apps/accounting_workbench/app.py`
  - ファイルや表データから仕訳候補を整理し、予算編成と現預金設定付きの Excel を出力します。
- `apps/behavioral_gap_lab/app.py`
  - 伝統的ファイナンスと行動ファイナンスの価格形成の乖離を比較します。
- `apps/monetary_policy_lab/app.py`
  - 金利上昇によるインフレ抑制の確率シミュレーションを行います。
- `apps/factor_forecast_lab/app.py`
  - ローリングファクター、レジーム、フーリエ周期シグナルから次優位ファクターを予測します。

## Individual Run

```bash
streamlit run apps/accounting_workbench/app.py
streamlit run apps/behavioral_gap_lab/app.py
streamlit run apps/monetary_policy_lab/app.py
streamlit run apps/factor_forecast_lab/app.py
```

## Notes

- `apps/` 配下の各アプリは単独起動できます。
- 共有ロジックは `shared_finance/` にあります。
- `Factor Forecast Lab` のライブ因子取得は `shared_finance/factor_data_loader.py` に寄せてあるため、`app_stable/` が無くても動きます。
- IFRS 変換は、現時点では `Accounting Workbench` 内でヒント出力までです。
- `Factor Forecast Lab` はデモデータ、アップロードデータ、Ken French データの読み込みをサポートします。
