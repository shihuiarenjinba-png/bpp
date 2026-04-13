---
name: corporate-finance-analysis
description: 企業財務を多角的に分析する統合スキル。「財務比率を分析して」「複数企業を比較して」「キャッシュフローを予測して」「信用リスクを評価して」などの依頼に対応します。financial-ratio-analysis・comparative-financial-analysis・cashflow-forecast・credit-risk-analysisの4つのサブスキルを収録しており、CSVまたはExcelファイルを入力としてExcelレポートを出力します。
---

# 企業財務分析スキル (Corporate Finance Analysis)

企業の財務健全性を多角的に評価する統合スキルです。
財務比率分析、企業比較分析、キャッシュフロー予測、信用リスク分析の4つのサブスキルを収録しています。

> **重要**: 本スキルは財務分析の参考情報を提供するものであり、専門的な財務助言ではありません。
> 重要な意思決定の際は、公認会計士や財務アドバイザーにご相談ください。

---

## このスキルの使用場面（隣接スキルとの違い）

| 判断軸 | このスキル | securities-analysis | japan-stock-analyzer |
|--------|-----------|--------------------|--------------------|
| **目的** | 過去の財務健全性を診断 | 将来価値・投資判断 | データ収集・特徴量生成 |
| **入力** | CSVまたはExcelファイル | CSVまたはExcelファイル | 証券コード（自動取得） |
| **典型的な問い** | 「この企業は財務的に安全か？」 | 「この株は割安か？」 | 「〇〇（社名）を分析して」 |

**トリガーの使い分け**:
- 「**証券コードや会社名**が明示されている」→ `japan-stock-analyzer` を優先
- 「**CSVやExcelファイルがアップロードされている**」→ このスキルを使用
- 「**DCF・WACC・ポートフォリオ**」のキーワードがある → `securities-analysis` を使用

---

## サブスキル一覧

| サブスキル | フォルダ | 主なトリガー |
|-----------|----------|-------------|
| 財務比率分析 | `skills/financial-ratio-analysis/` | 「財務比率」「ROE」「デュポン分解」「財務分析レポート」 |
| 企業比較分析 | `skills/comparative-financial-analysis/` | 「企業を比較」「横並び比較」「競合比較」「コモンサイズ」 |
| キャッシュフロー予測 | `skills/cashflow-forecast/` | 「CF予測」「資金繰り」「キャッシュフロー予測」「シナリオ分析」 |
| 信用リスク分析 | `skills/credit-risk-analysis/` | 「信用リスク」「Z-Score」「F-Score」「倒産リスク」「信用格付け」 |

---

## サブスキルの詳細と使用方法

### 1. 財務比率分析 (`skills/financial-ratio-analysis/`)

**概要**: 財務諸表データから包括的な財務比率を計算し、収益性・安全性・効率性・市場評価の4カテゴリに分類したExcelレポートを出力します。デュポン分解、業界ベンチマーク比較、レーダーチャート可視化を搭載。

**スクリプト**: `skills/financial-ratio-analysis/scripts/ratio_calculator.py`
**参照**: `skills/financial-ratio-analysis/references/ratio_definitions.md`

**実行手順**:
```bash
SKILL_DIR="/home/ubuntu/skills/corporate-finance-analysis"
cp "${SKILL_DIR}/skills/financial-ratio-analysis/scripts/ratio_calculator.py" /home/ubuntu/
cp -r "${SKILL_DIR}/skills/financial-ratio-analysis/references/" /home/ubuntu/references_ratio/
pip install pandas numpy openpyxl matplotlib --break-system-packages -q

python3 /home/ubuntu/ratio_calculator.py \
    "/mnt/user-data/uploads/<ファイル名>" \
    /home/ubuntu/financial_ratio_report.xlsx
```

**トリガー例**:
- 「この決算データの財務比率を分析してレポートを作成してください」
- 「ROEとROAを計算してデュポン分解してください」

---

### 2. 企業比較分析 (`skills/comparative-financial-analysis/`)

**概要**: 複数企業の財務データを横並びで比較分析し、相対的な強みと弱みを可視化します。コモンサイズ分析（百分率損益計算書・貸借対照表）、成長率比較、指標ランキングを含むExcelレポートを出力。

**スクリプト**: `skills/comparative-financial-analysis/scripts/comparative_analyzer.py`
**参照**: `skills/comparative-financial-analysis/references/comparison_methods.md`

**実行手順**:
```bash
SKILL_DIR="/home/ubuntu/skills/corporate-finance-analysis"
cp "${SKILL_DIR}/skills/comparative-financial-analysis/scripts/comparative_analyzer.py" /home/ubuntu/
pip install pandas numpy openpyxl matplotlib --break-system-packages -q

python3 /home/ubuntu/comparative_analyzer.py \
    "/mnt/user-data/uploads/<ファイル名>" \
    /home/ubuntu/comparative_report.xlsx
```

**トリガー例**:
- 「A社、B社、C社の財務データを横並びで比較分析してください」
- 「競合他社との財務指標の比較レポートを作成してください」

---

### 3. キャッシュフロー予測 (`skills/cashflow-forecast/`)

**概要**: 企業の過去のキャッシュフロー実績データから多シナリオ予測・季節性分析・資金ショートリスク診断を実行。楽観/基本/悲観の3シナリオを含む12ヶ月間の詳細な資金繰り表を生成します。

**スクリプト**: `skills/cashflow-forecast/scripts/cashflow_forecaster.py`
**参照**: `skills/cashflow-forecast/references/forecasting_methods.md`

**実行手順**:
```bash
SKILL_DIR="/home/ubuntu/skills/corporate-finance-analysis"
cp "${SKILL_DIR}/skills/cashflow-forecast/scripts/cashflow_forecaster.py" /home/ubuntu/
pip install pandas numpy openpyxl matplotlib scipy --break-system-packages -q

python3 /home/ubuntu/cashflow_forecaster.py \
    "/mnt/user-data/uploads/<ファイル名>" \
    /home/ubuntu/cashflow_forecast_report.xlsx
```

**トリガー例**:
- 「過去3年のCFデータから今後12ヶ月のキャッシュフローを予測してください」
- 「資金繰り表を3シナリオで作成してください」

---

### 4. 信用リスク分析 (`skills/credit-risk-analysis/`)

**概要**: Altman Z-Score および Piotroski F-Score を用いた包括的な信用力評価。財務データから倒産リスク・デフォルトリスク・信用格付け（AAA〜D）を自動計算し、企業の財務健全性を多角的に分析します。

**スクリプト**: `skills/credit-risk-analysis/scripts/credit_risk_analyzer.py`
**参照**: `skills/credit-risk-analysis/references/credit_scoring_models.md`

**実行手順**:
```bash
SKILL_DIR="/home/ubuntu/skills/corporate-finance-analysis"
cp "${SKILL_DIR}/skills/credit-risk-analysis/scripts/credit_risk_analyzer.py" /home/ubuntu/
pip install pandas numpy openpyxl matplotlib scipy --break-system-packages -q

python3 /home/ubuntu/credit_risk_analyzer.py \
    "/mnt/user-data/uploads/<ファイル名>" \
    /home/ubuntu/credit_risk_report.xlsx
```

**トリガー例**:
- 「この企業の信用リスクをZ-ScoreとF-Scoreで評価してください」
- 「倒産リスクを分析して格付けしてください」

---

## 共通仕様

- **入力形式**: CSV または Excel (.xlsx / .xls)
- **出力形式**: Excel レポート (.xlsx)
- **依存ライブラリ**: pandas, numpy, openpyxl, matplotlib, scipy
- **文字コード**: UTF-8 / CP932 自動判定
