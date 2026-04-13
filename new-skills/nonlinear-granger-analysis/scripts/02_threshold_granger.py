"""
02_threshold_granger.py
========================
閾値グレンジャー因果性検定（TAR: Threshold Autoregression）
特定の体制指標（DEF_SPREAD等）の閾値によって因果関係が変化するかを検定。
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

FACTORS    = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
MACRO_VARS = ["CPI_growth", "IP_growth", "TERM_SPREAD", "DEF_SPREAD",
              "FX_return", "OIL_return", "VIX"]


def granger_f_test(y, x, lag):
    n = len(y); T = n - lag
    if T < lag * 3 + 5 or n < 20: return np.nan, np.nan
    Xu = np.ones((T, 1 + 2*lag))
    for k in range(1, lag+1):
        Xu[:, k]     = y[lag-k:n-k]
        Xu[:, lag+k] = x[lag-k:n-k]
    Xr = Xu[:, :lag+1]; yv = y[lag:]
    try:
        cu, _, _, _ = np.linalg.lstsq(Xu, yv, rcond=None)
        cr, _, _, _ = np.linalg.lstsq(Xr, yv, rcond=None)
        RSSu = np.sum((yv - Xu@cu)**2); RSSr = np.sum((yv - Xr@cr)**2)
        df1 = lag; df2 = T - 2*lag - 1
        if df2 <= 0 or RSSu <= 1e-12: return np.nan, np.nan
        F = ((RSSr - RSSu)/df1) / (RSSu/df2)
        return float(F), float(1 - stats.f.cdf(F, df1, df2))
    except: return np.nan, np.nan


def run_threshold_granger(data, threshold_var="DEF_SPREAD", max_lag=3):
    if threshold_var not in data.columns:
        print(f"警告: {threshold_var} がデータに存在しません")
        return pd.DataFrame()

    rows = []
    # 閾値候補: 15th〜85th percentile の各5%点
    thresholds = np.percentile(data[threshold_var].dropna(), np.arange(20, 81, 10))

    for mv in MACRO_VARS:
        if mv not in data.columns: continue
        for factor in FACTORS:
            if factor not in data.columns: continue

            best_result = None
            best_threshold = None
            min_pval_low = 1.0

            for thr in thresholds:
                high_regime = data[data[threshold_var] > thr]
                low_regime  = data[data[threshold_var] <= thr]

                for regime_name, regime_data in [("高体制", high_regime), ("低体制", low_regime)]:
                    subset = regime_data[[factor, mv]].dropna()
                    if len(subset) < 20: continue
                    lag = min(max_lag, max(1, len(subset)//15))
                    F, p = granger_f_test(subset[factor].values, subset[mv].values, lag)
                    if not np.isnan(p) and p < min_pval_low:
                        min_pval_low = p
                        best_threshold = thr
                        best_result = (regime_name, p, F, lag, len(subset))

            # 全体p値
            full_sub = data[[factor, mv]].dropna()
            F_full, p_full = granger_f_test(full_sub[factor].values, full_sub[mv].values,
                                             min(max_lag, max(1, len(full_sub)//15)))

            row = {
                "マクロ変数": mv,
                "ファクター": factor,
                "閾値変数": threshold_var,
                "最適閾値": round(best_threshold, 3) if best_threshold is not None else np.nan,
                "全体p値": round(p_full, 4) if not np.isnan(p_full) else np.nan,
                "最小体制p値": round(min_pval_low, 4),
                "有意体制": best_result[0] if best_result else "なし",
                "体制F統計量": round(best_result[2], 3) if best_result and not np.isnan(best_result[2]) else np.nan,
                "体制n": best_result[4] if best_result else 0,
                "閾値効果": "★ あり" if (not np.isnan(p_full) and p_full >= 0.05 and min_pval_low < 0.05) else "",
            }
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",          default="data/merged_data.csv")
    parser.add_argument("--threshold-var", default="DEF_SPREAD")
    parser.add_argument("--output",        default="results/tar_results.csv")
    parser.add_argument("--max-lag",       type=int, default=3)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    data = pd.read_csv(args.data, index_col=0, parse_dates=True, encoding="utf-8-sig")

    print(f"閾値変数: {args.threshold_var}")
    print("閾値グレンジャー検定実行中...")
    result_df = run_threshold_granger(data, args.threshold_var, args.max_lag)
    result_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    # 閾値効果が検出されたペアを表示
    threshold_detected = result_df[result_df["閾値効果"] == "★ あり"]
    print(f"\n★ 閾値効果が検出されたペア: {len(threshold_detected)}件")
    if len(threshold_detected) > 0:
        print(threshold_detected[["マクロ変数","ファクター","最適閾値","全体p値","最小体制p値","有意体制"]].to_string(index=False))
    print(f"\n✓ 保存: {args.output}")


if __name__ == "__main__":
    main()
