---
name: nonlinear-granger-analysis
description: >
  非線形グレンジャー因果性分析スキル。線形グレンジャー検定で「非有意」だったペアに対して、
  非線形・体制転換アプローチで隠れた因果関係を探索する修士論文支援ツール。
  MS-VAR（マルコフスイッチングVAR）、TAR（閾値自己回帰）、カーネルベースの非線形グレンジャー検定、
  危機期/平常期サブサンプル分析を実行し、Excelレポートを出力します。
  「非線形グレンジャー」「体制転換」「MS-VAR」「TAR」「構造変化」「サブサンプル分析」
  「危機期分析」「非線形因果性」「FF5非線形」などで積極的に使用してください。
  線形グレンジャー検定の補完・頑健性チェックとして必ず参照すること。
---

# 非線形グレンジャー因果性分析スキル
## 修士論文「遅行ファクターの探索的研究」補完分析

---

## CSVファイル仕様（入出力フォーマット）

全スクリプトで以下のフォーマットに統一されています。

| ファイル | encoding | index | 備考 |
|---------|----------|-------|------|
| `data/merged_data.csv`（入力） | `utf-8-sig` | `index_col=0, parse_dates=True` | ff5-macro-analysisスキルのStep1で生成 |
| `results/subsample_results.csv` | `utf-8-sig` | `index=False` | 読み込み側は `index_col=False` |
| `results/tar_results.csv` | `utf-8-sig` | `index=False` | 読み込み側は `index_col=False` |
| `results/kernel_results.csv` | `utf-8-sig` | `index=False` | 読み込み側は `index_col=False` |
| `results/rolling_results.csv` | `utf-8-sig` | `index=False` | 読み込み側は `index_col=False` |

> **注意**: `merged_data.csv` の日付インデックスは `parse_dates=True` で自動変換されます。
> Windowsの場合も `utf-8-sig`（BOM付き）なのでExcelで直接開いても文字化けしません。

---

## 概要と位置づけ

線形グレンジャー検定（スキル: ff5-macro-analysis）で「非有意」だった28ペアに対し、
以下の非線形・体制転換アプローチで再検討する。

**非有意 ≠ 因果関係なし** ── 非線形モデルで初めて可視化される関係が存在する。

---

## ⚠️ 実行前チェック（必須）

本スキルは `ff5-macro-analysis` スキルの出力ファイルに依存します。
**Step 1 を実行する前に、必ず以下を確認してください。**

```
【チェックリスト】
□ data/merged_data.csv が存在するか？
  → 存在する場合 : Step 1（サブサンプル分析）から実行できます
  → 存在しない場合: 先に ff5-macro-analysis スキルの Step 1 を実行してください

      python ff5-macro-analysis/scripts/01_data_collection.py \
          --start-year 2006 --end-year 2025 \
          --fred-api-key "YOUR_API_KEY" \
          --output data/merged_data.csv

□ data/merged_data.csv のインデックスは日付型か？
  → pd.read_csv("data/merged_data.csv", index_col=0, parse_dates=True) で確認

□ results/granger_results.csv が存在するか？（Step 5の統合レポートに必要）
  → 存在しない場合: ff5-macro-analysis の Step 2 も先に実行してください
```

> **なぜこのチェックが必要か**
> `merged_data.csv` は ff5-macro-analysis の Step 1 で生成される
> 月次パネルデータです。このファイルがない状態で本スキルを実行すると、
> 全ての分析スクリプトがファイル未検出エラーで停止します。
> また、ADF自動修正後に列名が変わっている可能性（例: `VIX` → `VIX_d1`）があるため、
> `merged_data_transform_log.csv` も確認しておくと安全です。

---

## 分析メニュー（4つのアプローチ）

| # | 手法 | 対象問題 | 実装 |
|---|------|---------|------|
| 1 | **サブサンプル分析** | 体制転換・構造変化 | scripts/01_subsample.py |
| 2 | **閾値グレンジャー検定 (TAR)** | 非線形閾値効果 | scripts/02_threshold_granger.py |
| 3 | **カーネルグレンジャー検定** | 一般的非線形関係 | scripts/03_kernel_granger.py |
| 4 | **ローリングウィンドウ分析** | 時変因果性 | scripts/04_rolling_granger.py |

---

## ワークフロー

### 前提
```
data/merged_data.csv が存在すること（ff5-macro-analysisスキルのStep 1で生成）
```

### Step 1: サブサンプル分析（推奨：まず実行）
```bash
python scripts/01_subsample.py \
    --data data/merged_data.csv \
    --output results/subsample_results.csv
```
**出力**: 危機期（2008-09, 2020）vs 平常期のグレンジャー検定比較表

### Step 2: 閾値グレンジャー検定
```bash
python scripts/02_threshold_granger.py \
    --data data/merged_data.csv \
    --threshold-var DEF_SPREAD \
    --output results/tar_results.csv
```
**出力**: TAR検定結果（体制別p値・閾値推定値）

### Step 3: カーネルグレンジャー検定
```bash
python scripts/03_kernel_granger.py \
    --data data/merged_data.csv \
    --n-permutations 499 \
    --output results/kernel_results.csv
```
**出力**: 非線形グレンジャー検定結果（置換検定p値）

### Step 4: ローリングウィンドウ分析
```bash
python scripts/04_rolling_granger.py \
    --data data/merged_data.csv \
    --window 60 \
    --output results/rolling_results.csv
```
**出力**: 時変p値系列・有意期間の特定

### Step 5: 統合Excelレポート生成
```bash
python scripts/05_generate_nonlinear_report.py \
    --subsample results/subsample_results.csv \
    --tar       results/tar_results.csv \
    --kernel    results/kernel_results.csv \
    --rolling   results/rolling_results.csv \
    --linear    results/granger_results.csv \
    --output    outputs/nonlinear_analysis_report.xlsx
```

---

## 各手法の解説と論文での使い方

### 1. サブサンプル分析

**理論的根拠**: Lucas critique — パラメータは体制（政策・経済環境）の変化で変わる。

**体制区分の推奨設定**:
```python
REGIMES = {
    "GFC危機期":   ("2008-09-01", "2009-06-30"),
    "COVID危機期": ("2020-02-01", "2020-09-30"),
    "平常期":      全期間からGFC・COVID除外,
    "低金利期":    ("2010-01-01", "2021-12-31"),
    "高金利期":    ("2022-01-01", 最新),
}
```

**論文記述例**:
> 「全サンプルでは非有意であった [変数X] → [ファクターY] の関係（p=0.xxx）は、
> GFC危機期サブサンプルでは有意であった（p=0.0xx, n=10ヶ月）。
> この結果は、危機時における信用リスクチャネルの活性化と整合的である。」

### 2. 閾値グレンジャー検定 (Threshold Autoregression)

**理論的根拠**: マクロ変数とFFファクターの関係は、特定の閾値（例: DEF_SPREAD > 2%）を
超えた時のみ有意になる可能性がある（Balke, 2000; Hansen, 2011）。

**閾値変数の選択基準**:
- DEF_SPREAD（信用スプレッド）→ 体制指標として最適
- VIX（恐怖指数）→ リスクオン/オフの体制区分
- TERM_SPREAD（タームスプレッド）→ 景気局面の代理変数

**実装方法**:
```python
# 閾値を探索グリッドで推定
thresholds = np.percentile(threshold_var, np.arange(15, 85, 5))
for q in thresholds:
    regime_high = data[threshold_var] > q   # 体制1（高リスク）
    regime_low  = data[threshold_var] <= q  # 体制2（低リスク）
    # 各体制でグレンジャー検定を実行
```

### 3. カーネルグレンジャー検定（非線形）

**理論的根拠**: Diks & Panchenko (2006) の非線形グレンジャー検定。
線形モデルが捉えられない非線形依存構造を置換検定（permutation test）で検出。

**特徴**:
- H0: xのラグがyの条件付き分布に影響しない（非線形版）
- 計算コスト高（n_permutations=499推奨、論文では999も）
- 線形検定で非有意でも、ここで有意 → 非線形関係の存在証拠

**論文での位置づけ**: 「頑健性チェック」として付録に記載するのが一般的。

### 4. ローリングウィンドウ分析（時変因果性）

**理論的根拠**: 因果関係の強度と方向は時間とともに変化する（Shi et al., 2020）。

**設定の推奨**:
```python
WINDOW = 60   # 5年（60ヶ月）ローリングウィンドウ
STEP   = 1    # 1ヶ月ステップ
```

**可視化**: 各ペアについてp値の時系列プロットを生成。
有意期間（p<0.05のゾーン）が危機時に集中 → 「危機駆動型因果性」と解釈。

---

## 結果の解釈マトリックス

| 線形GC | サブサンプル | TAR | カーネル | 解釈 |
|--------|-------------|-----|---------|------|
| 非有意 | 危機期のみ有意 | - | - | **危機駆動型因果性** |
| 非有意 | 低金利期のみ | - | - | **政策体制依存型** |
| 非有意 | 非有意 | 高体制のみ有意 | - | **閾値型非線形** |
| 非有意 | 非有意 | 非有意 | 有意 | **一般非線形関係** |
| 非有意 | 非有意 | 非有意 | 非有意 | **真に因果なし（EMH支持）** |
| 補正前有意 | 平常期有意 | - | - | **頑健な弱い因果性** |

---

## インストール・依存パッケージ

```bash
pip install pandas numpy scipy scikit-learn statsmodels openpyxl \
    matplotlib seaborn --break-system-packages -q
```

> statsmodelsが利用不可の場合はscipyベース実装（scripts内に含む）を使用。

---

## 論文での非線形分析の位置づけ

**推奨章構成**:
```
第4章 実証分析
  4.1 線形グレンジャー因果性検定（ff5-macro-analysisスキル）
  4.2 非線形・体制転換分析（本スキル）
    4.2.1 サブサンプル分析
    4.2.2 閾値グレンジャー検定
    4.2.3 ローリングウィンドウ分析
  4.3 ランダムフォレスト分析
第5章 考察
  5.1 線形 vs 非線形の結果比較
  5.2 遅行性の体制依存性
```

**主張の強化フレーム**:
> 「線形グレンジャー検定のみでは見落とされる体制依存的・非線形な先行関係が存在する。
> 本研究は線形分析を出発点とし、非線形・体制転換分析によってより包括的な
> マクロ変数とFFファクターの動的関係を明らかにした。」

---

## トラブルシューティング

| 問題 | 対処法 |
|------|--------|
| サブサンプルのn数不足（<20） | 体制区分を広げる（例: GFC: 2008-01〜2010-06） |
| カーネル検定が遅い | n_permutationsを99に減らしてプレビュー確認 |
| TAR閾値が収束しない | threshold_varを変更（VIX→DEF_SPREAD） |
| ローリングp値が全て非有意 | windowを36ヶ月に短縮して局所的有意期間を探す |

---

## 参考文献

- Diks, C. & Panchenko, V. (2006). A new statistic and practical guidelines for nonparametric Granger causality testing. *Journal of Economic Dynamics and Control*.
- Hansen, B.E. (2011). Threshold autoregression in economics. *Statistics and Its Interface*.
- Shi, S., Phillips, P.C.B., & Hurn, S. (2020). Change detection and the causal impact of the yield curve. *Journal of Time Series Analysis*.
- Balke, N.S. (2000). Credit and economic activity: Credit regimes and nonlinear propagation. *Review of Economics and Statistics*.
