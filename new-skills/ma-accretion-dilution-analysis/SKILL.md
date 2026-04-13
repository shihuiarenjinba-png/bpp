---
name: ma-accretion-dilution-analysis
description: M&AにおけるプロフォーマEPS（アクレション／ディリューション）を、調達構造・シナジー・税率を踏まえて定量化する。「アクレディル」「EPSインパクト」「株式交換」「デット調達での買収」など。入力はCSV（field,value列）、出力はExcel（Summary/Bridge/Assumptions/Sensitivity）。
---

# M&A アクレション／ディリューション分析スキル

買収側・対象の純利益と株式数、新株・デット・シナジー（税前または税後）を入力し、**プロフォーマEPS** と **アクレション／ディリューション率** をExcelに出力する。

> 教育・分析用の簡易モデルです。のれん・PPA・潜在株の詳細は未モデル化（Assumptionsシートに注記）。

---

## いつこのスキルを使うか（一文）

**買収後の「1株あたり利益が増えるか減るか」を、調達とシナジー込みで数値化したいとき**に使う。

## 他スキルとの使い分け

| 軸 | 本スキル | corporate-finance-analysis | dcf-valuation |
|----|----------|------------------------------|---------------|
| 問い | EPSの増減 | 財務健全性・比率 | 企業価値 |
| 入力 | `ma_inputs.csv` | 財務CSV/Excel | FCF・前提 |

---

## 4-Phase ワークフロー

### Phase 1 — 調査
- 取引構造（新株数・デット・持分率）とLTMのNI・株数を確認。
- `sample_inputs/ma_inputs.csv` をコピーし、`field` / `value` を埋める（`unit` / `notes` は任意）。

### Phase 2 — 実行
```bash
pip install pandas openpyxl -q
python3 scripts/ma_accretion_dilution.py sample_inputs/ma_inputs.csv ./ma_accretion_dilution_report.xlsx
```
（パスはスキルルートからの相対でも絶対でも可。）

### Phase 3 — 批判
- `Bridge` の符号とオーダーが経済合理性と合うか確認。
- `Sensitivity` でシナジー±20%・金利±100bpのブレを確認。
- 未モデル化項目は `Assumptions` の Limitations をユーザーに明示。

### Phase 4 — 最終化
- Summaryの3指標（Standalone EPS / Pro forma EPS / 変化率%）を要約し、必要ならDCFやデューデリへ誘導。

---

## 入力（固定スキーマ）

CSVは **`field`,`value` 列必須**（`unit`,`notes` 任意）。

| field | 説明 |
|-------|------|
| `ni_acquirer` | 買収側 当期純利益（LTM推奨） |
| `shares_acquirer` | 買収側 加重平均株数（基本EPS用） |
| `ni_target` | 対象 当期純利益 |
| `shares_target` | 対象 発行済株数（交換比率の参考用・計算は `consideration_shares` が主） |
| `ownership_pct` | 取得後持分（1.0=100%） |
| `consideration_cash` | 現金対価総額（簡易モデルでは株数に直接効かず、必要なら別途反映） |
| `consideration_shares` | **新規発行株数**（ストックスワップ） |
| `new_debt_principal` | 新規デット元本 |
| `debt_interest_rate` | 年利（例: 0.04） |
| `tax_rate` | 法人税率（例: 0.30） |
| `synergy_aftertax` | 税引後シナジー（年率）※優先 |
| `synergy_pretax` | 税前シナジー（`synergy_aftertax` が0のとき ×(1−税率) で使用） |
| `one_time_costs_aftertax` | 取引費用等（税引後、任意） |
| `preferred_dividend` | 優先配当（任意） |

**計算式（コア）**  
プロフォーマNI = NI_A + NI_T×ownership + 税引後シナジー − 税引後利払い − 一時費用税後 − 優先配当  
プロフォーマ株数 = 買収側株数 + `consideration_shares`  
Pro forma EPS = プロフォーマNI ÷ プロフォーマ株数  
変化率 = (Pro forma EPS − Standalone EPS) / |Standalone EPS|

---

## Python実装

- **スクリプト**: `scripts/ma_accretion_dilution.py`
- **依存**: `pandas`, `openpyxl`

---

## 出力Excel構成

| シート | 内容 |
|--------|------|
| `Summary` | Standalone / Pro forma EPS、変化率%、株数・プロフォーマNI |
| `Bridge` | NIの増減ブリッジ |
| `Assumptions` | 入力一覧 + 未モデル化の注記 |
| `Sensitivity` | シナジー±20% × 金利±100bp のグリッド |

---

## トリガー例

- 「アクレディル」「EPSインパクト」「M&A 希薄化」「ストックスワップ EPS」
- 「デットで買収した場合のEPS」

---

## 実行例（エージェント向け）

```bash
cd /path/to/ma-accretion-dilution-analysis
python3 scripts/ma_accretion_dilution.py sample_inputs/ma_inputs.csv /tmp/ma_report.xlsx
```

成功時、標準出力にプロフォーマEPSの一行サマリーが出る。
