---
name: ff5-macro-analysis
description: >
  FF5遅行ファクター研究スキル。Fama-French 5ファクターとマクロ経済変数の時系列的先行遅行関係を探索する修士論文分析ツール。
  Kenneth French Data LibraryおよびFRED（Federal Reserve Economic Data）から公式データを取得し、
  グレンジャー因果性検定（Bonferroni補正）・ランダムフォレスト・特徴量重要度分析を実行します。
  「遅行ファクター」「グレンジャー因果性」「FF5分析」「Fama-French」「マクロ変数」「RMW」「CPI先行」などで使用します。
---

# FF5遅行ファクター分析スキル
## 修士論文「遅行ファクターの探索的研究」支援ツール

---

## 概要

本スキルは以下の分析を自動実行します：
1. **データ取得**：FF5月次データ（Kenneth French Data Library）＋マクロ変数（FRED）
2. **前処理**：ADF検定による定常性確認・自動変換（v2以降）
3. **グレンジャー因果性検定**：35通り（7変数×5ファクター）、BICラグ選択、Bonferroni補正
4. **ランダムフォレスト分析**：ウォークフォワード検証、OOS R²、特徴量重要度
5. **Excelレポート出力**：p値マトリックス・結果表・チャート

---

## データソース（信頼性の高い公的機関のみ）

### FF5ファクターデータ
- **提供元**: Kenneth R. French - Data Library (Dartmouth University)
- **URL**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **ダウンロードURL**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip
- **形式**: 月次リターン（%）、1963年7月〜最新
- **変数**: Mkt-RF, SMB, HML, RMW, CMA, RF

### マクロ経済変数データ
- **提供元**: FRED - Federal Reserve Bank of St. Louis
- **URL**: https://fred.stlouisfed.org/
- **アクセス方法**: FRED APIキー（スクリプトにデフォルト設定済み）

| 変数 | FRED系列ID | 説明 | 周期 |
|------|-----------|------|------|
| CPI成長率 | CPIAUCSL | 都市部消費者物価指数 | 月次 |
| 鉱工業生産成長率 | INDPRO | 鉱工業生産指数 | 月次 |
| 10年国債 | GS10 | 10年物国債利回り | 月次 |
| 3ヶ月T-Bill | TB3MS | 3ヶ月財務省証券利回り | 月次 |
| BAA社債利回り | DBAA | Moody's BAA格社債 | 月次 |
| AAA社債利回り | DAAA | Moody's AAA格社債 | 月次 |
| 米ドル実効為替 | DTWEXBGS | 広義ドル指数（名目） | 日次→月次集計 |
| WTI原油価格 | DCOILWTICO | WTI原油スポット価格 | 日次→月次集計 |
| VIX | VIXCLS | CBOE Volatility Index | 日次→月次集計 |

> ⚠️ **重要**: 論文執筆時は必ずFREDの公式サイトでシリーズの定義・改定履歴を確認すること。
> FREDデータは遡及改定があるため、取得日時をメタデータとして必ず記録する。

---

## ファイル構成

```
ff5-macro-analysis/
├── SKILL.md                      # このファイル
├── scripts/
│   ├── 01_data_collection.py    # データ取得・前処理・ADF自動修正
│   ├── 02_granger_analysis.py   # グレンジャー因果性検定
│   ├── 03_random_forest.py      # ランダムフォレスト分析
│   └── 04_generate_report.py    # Excelレポート生成
└── references/
    └── data_sources.md          # データソース詳細・引用形式
```

---

## CSVファイル仕様（入出力フォーマット）

全スクリプトで以下のフォーマットに統一されています。

| ファイル | encoding | index | 備考 |
|---------|----------|-------|------|
| `data/merged_data.csv`（書き出し） | `utf-8-sig` | `index=True`（日付列） | 読み込み側は `index_col=0, parse_dates=True` |
| `data/merged_data_adf_results.csv` | `utf-8-sig` | `index=False` | ADF検定結果（変換前） |
| `data/merged_data_transform_log.csv` | `utf-8-sig` | `index=False` | ADF自動修正の変換ログ |
| `data/merged_data_metadata.json` | `utf-8` | — | 取得日時・データソース・変換内容 |
| `results/granger_results.csv` | `utf-8-sig` | `index=False` | 読み込み側は `index_col=False` |
| `results/rf_results.csv` | `utf-8-sig` | `index=False` | 読み込み側は `index_col=False` |
| `results/*_feature_importance.csv` | `utf-8-sig` | `index=False` | 読み込み側は `index_col=False` |
| `results/granger_results_pmatrix.csv` | `utf-8-sig` | `index=True`（マクロ変数名） | — |

> **注意**: Windowsで開く場合は `utf-8-sig`（BOM付きUTF-8）を使用しているため、
> Excelで直接開いても文字化けしません。

---

## ワークフロー

### Step 0: 必要パッケージのインストール
```bash
pip install pandas numpy scipy scikit-learn statsmodels openpyxl \
    pandas_datareader fredapi matplotlib seaborn requests zipfile36 \
    --break-system-packages -q
```

### Step 1: データ取得・前処理
```bash
python scripts/01_data_collection.py \
    --start-year 2006 \
    --end-year 2025 \
    --output data/merged_data.csv
```

> **FRED APIキーについて**: `--fred-api-key` はスクリプト内にデフォルト設定済みのため、
> 通常は指定不要です。別のキーを使用したい場合のみ明示指定してください。
> ```bash
> python scripts/01_data_collection.py --fred-api-key "別のAPIキー" ...
> ```

**出力ファイル**:
- `data/merged_data.csv` — 統合データ（月次、`utf-8-sig`、index=日付）
- `data/merged_data_adf_results.csv` — 変換前のADF検定結果
- `data/merged_data_transform_log.csv` — 自動変換ログ（下記参照）
- `data/merged_data_metadata.json` — 取得日時・変換内容の記録

---

### 【重要】ADF自動修正について（v2以降の新機能）

非定常と判定されたマクロ変数は自動的に変換が試みられます。

**変換の優先順位**（各段階でADF再検定し、定常化できた時点で停止）:

| Stage | 変換 | 適用条件 |
|-------|------|----------|
| 1 | 1階差分 `diff()` | 常に試みる |
| 2 | 対数1階差分 `log().diff()` | Stage 1 で非定常かつ全値が正の場合 |

**重要な注意事項**:
- FF5ファクター（Mkt-RF, SMB, HML, RMW, CMA, RF）は**変換対象外**（すでにリターン系列のため差分を取ると意味が変わる）
- 変換後の列名にはサフィックスが付与される: `VIX` → `VIX_d1`（差分）または `VIX_ld1`（対数差分）
- `nonlinear-granger-analysis` スキルを使う場合、列名変更を `transform_log.csv` で事前確認すること
- 変換後にADF再検定が自動実行され、定常性を再確認する
- 手動で変換を管理したい場合は `--no-auto-correct` フラグを使用:
  ```bash
  python scripts/01_data_collection.py --no-auto-correct ...
  ```

**論文記述例**: 「ADF検定で非定常と判定された[変数名]については1階差分を適用した（変換後 p < 0.05）。」

---

### Step 2: グレンジャー因果性検定
```bash
python scripts/02_granger_analysis.py \
    --data data/merged_data.csv \
    --max-lag 6 \
    --alpha 0.05 \
    --output results/granger_results.csv
```

**出力**:
- `results/granger_results.csv` — 全35通りのp値
- `results/granger_summary.csv` — 有意な結果のみ（Bonferroni補正後）

### Step 3: ランダムフォレスト分析
```bash
python scripts/03_random_forest.py \
    --data data/merged_data.csv \
    --train-ratio 0.7 \
    --n-estimators 500 \
    --output results/rf_results.csv
```

**出力**:
- `results/rf_results.csv` — 各ファクターのOOS R²・方向的一致率
- `results/feature_importance.csv` — 特徴量重要度（全ファクター）

### Step 4: Excelレポート生成
```bash
python scripts/04_generate_report.py \
    --granger results/granger_results.csv \
    --rf results/rf_results.csv \
    --importance results/feature_importance.csv \
    --output ../outputs/ff5_analysis_report.xlsx
```

---

## 呼び出し例

ユーザーから「グレンジャー検定を実行して」と言われた場合：
1. `data/merged_data.csv` が存在するか確認
2. なければ Step 1 を先に実行（APIキー指定不要）
3. Step 2 を実行してp値マトリックスを生成
4. 結果をユーザーに提示し、有意な結果（Bonferroni補正後）を強調

ユーザーから「最新データで再分析して」と言われた場合：
1. Step 1 を最新データで実行（`--end-year` を現在年に設定）
2. Step 2〜4 を順次実行
3. 前回との差分をレポート

---

## 結果の解釈ガイド

### グレンジャー因果性検定
- **p < 0.0014**（Bonferroni補正後）: 頑健な先行関係の証拠
- **p < 0.05（補正前）**: 参考値（論文では「補正前のみ有意」として報告）
- **ラグ次数（BIC選択）**: 選択されたラグが長い = より遅い情報織り込みを示唆

### ランダムフォレスト
- **OOS R² > 0**: ヒストリカル平均より優れた予測 → 予測可能性の存在
- **OOS R² ≤ 0**: 実用的予測改善なし（論文のRMW結果と整合）
- **特徴量重要度**: 長ラグ（6〜12ヶ月）の変数が上位 → 遅行的反応の示唆

---

## 論文執筆上の注意事項

1. **データの引用方法**:
   - FF5: `French, K.R. (2024). Data Library. Retrieved from https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html`
   - FRED: `Federal Reserve Bank of St. Louis (2024). FRED Economic Data. Retrieved from https://fred.stlouisfed.org/`

2. **データのバージョン管理**: 取得日を `merged_data_metadata.json` から確認し、論文に明記すること（FREDデータは遡及改定あり）

3. **再現可能性**: `numpy.random.seed()` と `sklearn` の `random_state` を固定すること

4. **多重検定**: 35通りの検定に対してBonferroni補正を必ず適用すること
   - 補正後有意水準 = 0.05 / 35 ≈ 0.0014

5. **定常性の確認**: ADF自動修正が適用された変数は `transform_log.csv` で変換内容を確認し、論文の前処理節に記載すること

---

## トラブルシューティング

| 問題 | 対処法 |
|------|--------|
| FREDデータが取得できない | `fredapi` への切り替えまたは手動CSVダウンロード（https://fred.stlouisfed.org/series/{ID}） |
| French Libraryが繋がらない | ミラーサイトまたは手動ダウンロード: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/ |
| メモリエラー | `n_estimators` を 100 に減らし、データ期間を短縮 |
| VIXのサンプル期間不足 | VIX分析は1990年以降のサブサンプルで実施（本論文設計通り） |
| ADF自動修正で列名が変わり02以降が動かない | `transform_log.csv` で変換後の列名を確認し、02のスクリプト引数を更新 |
