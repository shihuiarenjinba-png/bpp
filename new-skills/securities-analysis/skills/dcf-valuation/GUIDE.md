---
name: dcf-valuation
type: independent-skill
description: DCF(割引キャッシュフロー法)に基づいた企業価値評価とバリュエーション分析を自動化するスキル。CSV/XLSXファイルから財務データを読み込み、FCF予測、WACC計算、感度分析を含む詳細なExcelレポートを生成します。
triggers:
  - DCF
  - 企業価値
  - バリュエーション
  - 割引キャッシュフロー
  - WACC
  - intrinsic value
  - 株価算定
version: 1.0.0
author: Claude Code
---

# DCFバリュエーション分析スキル

## 概要
このスキルは、割引キャッシュフロー(DCF)法を用いた企業価値評価を自動化します。財務データからフリーキャッシュフロー(FCF)、加重平均資本コスト(WACC)、企業価値、株価を計算し、感度分析を含む詳細なExcelレポートを出力します。

## 必要な入力データ

CSV形式またはExcel形式のファイルに以下の情報を含める必要があります:

### 必須項目
- **売上(Revenue)**: 過去3年から5年の実績値
- **営業費用(Operating Costs)**: 営業に関連する費用
- **税率(Tax Rate)**: 実効税率（小数点形式：0.30で30%）
- **減価償却費(Depreciation & Amortization)**: D&A額
- **資本支出(Capital Expenditure)**: CapEx額
- **運転資本変化(Change in Working Capital)**: ΔWC（オプション）

### WACC計算用項目
- **株式時価総額(Market Cap of Equity)**: E
- **債務時価総額(Market Value of Debt)**: D
- **株式の期待収益率(Cost of Equity)**: Re（またはリスクフリーレート、市場リスクプレミアム）
- **負債の利率(Cost of Debt)**: Rd
- **負債比率(Debt Ratio)**: D/(D+E)

### オプション項目
- **成長率(Terminal Growth Rate)**: 永続成長率（デフォルト: 2.5%）
- **予測期間(Forecast Period)**: 年数（デフォルト: 5年）
- **出口マルチプル(Exit Multiple)**: EBITDA倍数での評価

## 使用方法

### 1. 入力ファイルの準備
以下の形式のCSVまたはExcelファイルを用意してください:

```
Year,Revenue,Operating_Costs,D_and_A,CapEx,Tax_Rate,Market_Cap,Debt_Value,Cost_of_Equity,Cost_of_Debt
2022,1000000,600000,50000,80000,0.30,5000000,1000000,0.10,0.05
2023,1100000,650000,55000,85000,0.30,5500000,1000000,0.10,0.05
2024,1200000,700000,60000,90000,0.30,6000000,1000000,0.10,0.05
```

### 2. スクリプト実行
```bash
python scripts/dcf_calculator.py --input <入力ファイルパス> --output <出力ファイルパス>
```

例：
```bash
python scripts/dcf_calculator.py --input financial_data.xlsx --output dcf_valuation_report.xlsx
```

### 3. 出力ファイルの確認
出力Excelファイルには以下のシートが含まれます:

- **サマリー**: 主要な評価結果（企業価値、株価）
- **FCF予測**: 5年間のフリーキャッシュフロー予測
- **WACC計算**: 加重平均資本コスト計算の詳細
- **感度分析**: WACC vs 成長率のマトリックス分析

## 計算方法

### フリーキャッシュフロー(FCF)計算
```
NOPAT = 営業利益 × (1 - 税率)
FCF = NOPAT + D&A - CapEx - ΔWC
```

### WACC（加重平均資本コスト）
```
WACC = (E / (E + D)) × Re + (D / (E + D)) × Rd × (1 - Tc)
```
- E: 株式の時価総額
- D: 負債の時価総額
- Re: 株式の期待収益率
- Rd: 負債コスト
- Tc: 企業税率

### 企業価値（Enterprise Value）
```
EV = Σ(FCFt / (1 + WACC)^t) + TV / (1 + WACC)^n
```
- FCFt: t年目のフリーキャッシュフロー
- TV: ターミナルバリュー
- n: 予測期間

### ターミナルバリュー
ゴードン成長モデル:
```
TV = FCFn × (1 + g) / (WACC - g)
```
- g: 永続成長率
- FCFn: 最終年のFCF

### 株価計算
```
株価 = (企業価値 - 負債 + 現金) / 発行済株式数
```

## 感度分析
WACC（6%-12%）と成長率（0%-5%）のマトリックスで、異なるシナリオの企業価値を表示します。

## エラーハンドリング
スクリプトは以下のエラーに対応しています:

- **不足データ**: 必須項目が不足している場合はエラーメッセージを表示
- **負のFCF**: 負の値が検出された場合は警告を出力
- **無効なWACC**: WACC > 成長率の条件をチェック
- **ファイル形式**: CSV および Excel（.xlsx）形式に対応

## 参照資料
詳細な計算式と業界別パラメータについては、`references/dcf_formulas.md`を参照してください。

## トラブルシューティング

**Q: PCLがインストールされていないというエラーが出る**
A: `pip install openpyxl pandas numpy` を実行してください

**Q: ファイルが見つからないエラー**
A: ファイルパスが正しいか確認してください。相対パスと絶対パスの両方に対応しています

**Q: 負のWACC値が出ている**
A: 負債コストが株式コストより高い場合は調整が必要です。入力データを確認してください

## ライセンス
このスキルはMIT Licenseの下で提供されています。
