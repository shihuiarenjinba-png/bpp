"""
04_rolling_granger.py
======================
ローリングウィンドウ・グレンジャー因果性検定
時変因果性（因果関係の強さが時間とともに変化するか）を可視化。
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
    if T < lag*3+5: return np.nan, np.nan
    Xu = np.ones((T, 1+2*lag))
    for k in range(1, lag+1):
        Xu[:,k] = y[lag-k:n-k]; Xu[:,lag+k] = x[lag-k:n-k]
    Xr = Xu[:, :lag+1]; yv = y[lag:]
    try:
        cu,_,_,_ = np.linalg.lstsq(Xu, yv, rcond=None)
        cr,_,_,_ = np.linalg.lstsq(Xr, yv, rcond=None)
        RSSu = np.sum((yv-Xu@cu)**2); RSSr = np.sum((yv-Xr@cr)**2)
        df1=lag; df2=T-2*lag-1
        if df2<=0 or RSSu<=1e-12: return np.nan, np.nan
        F = ((RSSr-RSSu)/df1)/(RSSu/df2)
        return float(F), float(1-stats.f.cdf(F,df1,df2))
    except: return np.nan, np.nan


def run_rolling_granger(data, window=60, lag=2):
    rows = []
    pairs = [(mv, f) for mv in MACRO_VARS if mv in data.columns
             for f in FACTORS if f in data.columns]

    for mv, factor in pairs:
        subset = data[[factor, mv]].dropna()
        n = len(subset)
        p_series = []
        dates_out = []

        for start in range(0, n - window + 1):
            win = subset.iloc[start:start+window]
            F, p = granger_f_test(win[factor].values, win[mv].values, lag)
            p_series.append(p)
            dates_out.append(subset.index[start + window - 1])

        p_arr = np.array([x if not np.isnan(x) else 1.0 for x in p_series])

        # 有意期間の特定
        sig_periods = [(dates_out[i].strftime("%Y-%m") if not np.isnan(p_series[i]) and p_series[i] < 0.05 else None)
                       for i in range(len(p_series))]
        sig_dates = [d for d in sig_periods if d is not None]

        rows.append({
            "マクロ変数":     mv,
            "ファクター":     factor,
            "ウィンドウ":     window,
            "期間数":         len(p_series),
            "平均p値":        round(np.nanmean(p_arr), 4),
            "最小p値":        round(np.nanmin(p_arr), 4),
            "有意ウィンドウ数": sum(1 for p in p_series if not np.isnan(p) and p < 0.05),
            "有意率(%)":      round(100 * sum(1 for p in p_series if not np.isnan(p) and p < 0.05) / len(p_series), 1),
            "有意期間（開始月）": sig_dates[0] if sig_dates else "なし",
            "時変因果性":     "★" if sum(1 for p in p_series if not np.isnan(p) and p < 0.05) >= 6 else "",
        })

        # 時系列CSV
        ts_rows = [{"日付": d.strftime("%Y-%m"), "マクロ変数": mv, "ファクター": factor,
                    "p値": round(p,4) if not np.isnan(p) else np.nan,
                    "有意": "★" if not np.isnan(p) and p<0.05 else ""}
                   for d, p in zip(dates_out, p_series)]
        ts_df_pair = pd.DataFrame(ts_rows)
        # 保存は後でまとめて
        if not hasattr(run_rolling_granger, '_ts_data'):
            run_rolling_granger._ts_data = []
        run_rolling_granger._ts_data.append(ts_df_pair)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/merged_data.csv")
    parser.add_argument("--window",  type=int, default=60)
    parser.add_argument("--lag",     type=int, default=2)
    parser.add_argument("--output",  default="results/rolling_results.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    data = pd.read_csv(args.data, index_col=0, parse_dates=True, encoding="utf-8-sig")

    print(f"ローリングウィンドウ分析（window={args.window}ヶ月、lag={args.lag}）...")
    result_df = run_rolling_granger(data, args.window, args.lag)
    result_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    # 時変因果性が検出されたペア
    tvc = result_df[result_df["時変因果性"] == "★"]
    print(f"\n★ 時変因果性が検出されたペア（有意ウィンドウ≥6）: {len(tvc)}件")
    if len(tvc) > 0:
        print(tvc[["マクロ変数","ファクター","有意ウィンドウ数","有意率(%)","有意期間（開始月）"]].to_string(index=False))

    # 時系列データも保存
    if hasattr(run_rolling_granger, '_ts_data') and run_rolling_granger._ts_data:
        ts_all = pd.concat(run_rolling_granger._ts_data)
        ts_path = args.output.replace(".csv", "_timeseries.csv")
        ts_all.to_csv(ts_path, index=False, encoding="utf-8-sig")
        print(f"✓ 時系列データ保存: {ts_path}")

    print(f"✓ 保存: {args.output}")


if __name__ == "__main__":
    main()
