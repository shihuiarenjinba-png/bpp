# Finance App Collection

複数の独立した金融アプリを `Streamlit` で動かせる構成です。既存の `app_stable/` と `simulator/` はそのまま残しつつ、今夜の追加分は `apps/` と `shared_finance/` にまとめています。

## Apps

- `apps/accounting_workbench/app.py`
  - ファイルや表データから仕訳候補を整理し、予算編成と現預金設定付きの Excel を出力します。
- `apps/behavioral_gap_lab/app.py`
  - 伝統的ファイナンスと行動ファイナンスの価格形成の乖離を比較します。
- `apps/monetary_policy_lab/app.py`
  - 金利上昇によるインフレ抑制の確率シミュレーションを行います。
- `apps/factor_forecast_lab/app.py`
  - ローリングファクター、レジーム、フーリエ周期シグナルから次優位ファクターを予測します。

## Run

```bash
streamlit run apps/accounting_workbench/app.py
streamlit run apps/behavioral_gap_lab/app.py
streamlit run apps/monetary_policy_lab/app.py
streamlit run apps/factor_forecast_lab/app.py
```

## Notes

- `apps/` 配下の各アプリは単独起動できます。
- 共有ロジックは `shared_finance/` にあります。
- IFRS 変換は、現時点では `Accounting Workbench` 内でヒント出力までです。
- `Factor Forecast Lab` はデモデータ、アップロードデータ、Ken French データの読み込みをサポートします。

