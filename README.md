# Finance App Collection

複数の金融アプリを `Streamlit` で切り替えて使える構成です。`bpp` 単体でも、ルートの [`app.py`](/Users/mitsuyoshitsuha/Documents/New%20project/app.py) から各研究画面へ入れるようにしています。

## Quick Start

```bash
pip install -r requirements-local.txt
streamlit run app.py
```

左のナビゲーションから、各アプリを切り替えて使えます。

## Vercel

Vercel では `Streamlit` をそのまま載せるのではなく、`shared_finance/` のエンジンを呼ぶ軽量な Python UI を `api/index.py` で返す構成にしています。
そのため、Vercel デプロイ先でも行動ファイナンス、ファクター予測、金融政策、会計整理の主要ロジックを実行できます。
一方で、ファイルアップロードや `Streamlit` 特有の編集 UI はローカル版のほうが充実しています。

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
- Vercel 用の軽量 UI は `requirements.txt`、ローカルの `Streamlit` UI は `requirements-local.txt` を使います。
