---
name: securities-analysis
description: 長期投資の視点から企業価値評価・ポートフォリオ管理・配当分析を行う統合スキル。「DCF分析して」「企業価値を算定して」「ポートフォリオを分析して」「配当を分析して」「DRIP（配当再投資）をシミュレーションして」などの依頼に対応します。dcf-valuation・portfolio-analytics・dividend-analysisの3つのサブスキルを収録しており、CSVまたはExcelファイルを入力としてExcelレポートを出力します。
---

# 証券分析スキル (Securities Analysis)

長期投資の視点から企業価値を評価し、ポートフォリオ管理を支援する統合スキルです。
DCFバリュエーション、ポートフォリオ分析、配当分析の3つのサブスキルを収録しています。

> **重要**: 本スキルは投資判断の参考情報を提供するものであり、投資助言ではありません。
> 投資判断はご自身の責任で行ってください。

---

## このスキルの使用場面（隣接スキルとの違い）

| 判断軸 | このスキル | corporate-finance-analysis | japan-stock-analyzer |
|--------|-----------|--------------------------|---------------------|
| **目的** | 将来価値の評価・投資判断支援 | 過去の財務健全性を診断 | データ収集・特徴量生成 |
| **入力** | CSVまたはExcelファイル | CSVまたはExcelファイル | 証券コード（自動取得） |
| **典型的な問い** | 「この株の内在価値はいくらか？」 | 「財務的に安全な企業か？」 | 「〇〇（社名）を分析して」 |

**トリガーの使い分け**:
- 「**DCF・WACC・内在価値・ポートフォリオ・配当再投資（DRIP）**」→ このスキルを使用
- 「**財務比率・信用リスク・Z-Score**」→ `corporate-finance-analysis` を使用
- 「**証券コードや会社名**が明示されている」→ `japan-stock-analyzer` を優先

---

## サブスキル一覧

| サブスキル | フォルダ | 主なトリガー |
|-----------|----------|-------------|
| DCFバリュエーション | `skills/dcf-valuation/` | 「DCF」「企業価値」「WACC」「FCF」「バリュエーション」「内在価値」 |
| ポートフォリオ分析 | `skills/portfolio-analytics/` | 「ポートフォリオ」「シャープレシオ」「最大ドローダウン」「アセットアロケーション」 |
| 配当分析 | `skills/dividend-analysis/` | 「配当」「配当利回り」「増配」「DRIP」「配当再投資」「配当割引モデル」 |

---

## サブスキルの詳細と使用方法

### 1. DCFバリュエーション (`skills/dcf-valuation/`)

**概要**: DCF法（割引キャッシュフロー法）による企業価値算定を自動化します。FCF予測・WACC計算・ターミナルバリュー・感度分析を含む詳細なExcelレポートを生成します。

**スクリプト**: `skills/dcf-valuation/scripts/dcf_calculator.py`
**参照**: `skills/dcf-valuation/references/dcf_formulas.md`

**実行手順**:
```bash
SKILL_DIR="/home/ubuntu/skills/securities-analysis"
cp "${SKILL_DIR}/skills/dcf-valuation/scripts/dcf_calculator.py" /home/ubuntu/
pip install pandas numpy openpyxl matplotlib --break-system-packages -q

python3 /home/ubuntu/dcf_calculator.py \
    "/mnt/user-data/uploads/<ファイル名>" \
    /home/ubuntu/dcf_valuation_report.xlsx
```

**トリガー例**:
- 「この企業の財務データでDCF分析をお願いします」
- 「WACCを計算して企業価値を算定してください」
- 「感度分析込みのバリュエーションレポートを作成してください」

---

### 2. ポートフォリオ分析 (`skills/portfolio-analytics/`)

**概要**: 保有銘柄のリスク・リターン分析、分散投資の効果測定、アセットアロケーション最適化、シャープレシオなどのリスク指標を自動計算し、可視化チャートを含むExcelレポートを生成します。

**スクリプト**: `skills/portfolio-analytics/scripts/portfolio_analyzer.py`
**参照**: `skills/portfolio-analytics/references/portfolio_metrics.md`
**サンプル**: `skills/portfolio-analytics/sample_portfolio.csv`

**実行手順**:
```bash
SKILL_DIR="/home/ubuntu/skills/securities-analysis"
cp "${SKILL_DIR}/skills/portfolio-analytics/scripts/portfolio_analyzer.py" /home/ubuntu/
pip install pandas numpy openpyxl matplotlib --break-system-packages -q

python3 /home/ubuntu/portfolio_analyzer.py \
    "/mnt/user-data/uploads/<ファイル名>" \
    /home/ubuntu/portfolio_report.xlsx
```

**トリガー例**:
- 「保有銘柄のポートフォリオ分析をお願いします」
- 「シャープレシオと最大ドローダウンを計算してください」
- 「セクター配分とリスク分散を可視化してください」

---

### 3. 配当分析 (`skills/dividend-analysis/`)

**概要**: 配当利回り・配当性向・連続増配年数・配当割引モデル（DGM）・配当再投資シミュレーション（DRIP）を含む包括的な配当分析レポートを生成します。

**スクリプト**: `skills/dividend-analysis/scripts/dividend_analyzer.py`
**参照**: `skills/dividend-analysis/references/dividend_models.md`

**実行手順**:
```bash
SKILL_DIR="/home/ubuntu/skills/securities-analysis"
cp "${SKILL_DIR}/skills/dividend-analysis/scripts/dividend_analyzer.py" /home/ubuntu/
pip install pandas numpy openpyxl matplotlib --break-system-packages -q

python3 /home/ubuntu/dividend_analyzer.py \
    "/mnt/user-data/uploads/<ファイル名>" \
    /home/ubuntu/dividend_report.xlsx
```

**トリガー例**:
- 「この銘柄の配当を分析して将来の配当収入をシミュレーションしてください」
- 「配当再投資（DRIP）の複利効果を計算してください」
- 「配当割引モデルで理論株価を算出してください」

---

## 共通仕様

- **入力形式**: CSV または Excel (.xlsx / .xls)
- **出力形式**: Excel レポート (.xlsx)
- **依存ライブラリ**: pandas, numpy, openpyxl, matplotlib
- **文字コード**: UTF-8 / CP932 自動判定
