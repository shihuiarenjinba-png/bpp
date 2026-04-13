"""
01_subsample.py
================
サブサンプル・グレンジャー因果性分析
体制別（危機期/平常期/低金利期/高金利期）にグレンジャー検定を実行し
全サンプルとの比較表を生成する。
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

FACTORS    = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
MACRO_VARS = ["CPI_growth", "IP_growth", "TERM_SPREAD", "DEF_SPREAD",
              "FX_return", "OIL_return", "VIX"]

REGIMES = {
    "全サンプル":    (None, None),
    "GFC危機期":     ("2008-07-01", "2009-12-31"),
    "COVID危機期":   ("2020-02-01", "2020-12-31"),
    "危機期合算":    [("2008-07-01","2009-12-31"), ("2020-02-01","2020-12-31")],
    "平常期":        "exclude_crisis",
    "低金利期":      ("2010-01-01", "2021-12-31"),
    "高金利期":      ("2022-01-01", "2025-12-31"),
}


def granger_f_test(y, x, lag):
    n = len(y); T = n - lag
    if T < lag * 3 + 5: return np.nan, np.nan
    Xu = np.ones((T, 1 + 2*lag))
    for k in range(1, lag+1):
        Xu[:, k]      = y[lag-k:n-k]
        Xu[:, lag+k]  = x[lag-k:n-k]
    Xr = Xu[:, :lag+1]
    yv = y[lag:]
    try:
        cu, _, _, _ = np.linalg.lstsq(Xu, yv, rcond=None)
        cr, _, _, _ = np.linalg.lstsq(Xr, yv, rcond=None)
        RSSu = np.sum((yv - Xu@cu)**2)
        RSSr = np.sum((yv - Xr@cr)**2)
        df1 = lag; df2 = T - 2*lag - 1
        if df2 <= 0 or RSSu <= 1e-12: return np.nan, np.nan
        F = ((RSSr - RSSu)/df1) / (RSSu/df2)
        return float(F), float(1 - stats.f.cdf(F, df1, df2))
    except: return np.nan, np.nan


def get_regime_data(data, regime_name, regime_def):
    if regime_def is None:
        return data
    if regime_def == "exclude_crisis":
        mask = ~(
            ((data.index >= "2008-07-01") & (data.index <= "2009-12-31")) |
            ((data.index >= "2020-02-01") & (data.index <= "2020-12-31"))
        )
        return data[mask]
    if isinstance(regime_def, list):
        frames = []
        for s, e in regime_def:
            frames.append(data[(data.index >= s) & (data.index <= e)])
        return pd.concat(frames)
    s, e = regime_def
    return data[(data.index >= s) & (data.index <= e)]


def run_subsample(data, max_lag=4):
    rows = []
    for mv in MACRO_VARS:
        if mv not in data.columns: continue
        for factor in FACTORS:
            if factor not in data.columns: continue
            row = {"マクロ変数": mv, "ファクター": factor}
            for regime_name, regime_def in REGIMES.items():
                sub = get_regime_data(data, regime_name, regime_def)
                subset = sub[[factor, mv]].dropna()
                n = len(subset)
                if n < 20:
                    row[f"{regime_name}_p値"] = "N/A(n不足)"
                    row[f"{regime_name}_n"]    = n
                    continue
                lag = min(max_lag, max(1, n//15))
                F, p = granger_f_test(subset[factor].values, subset[mv].values, lag)
                row[f"{regime_name}_p値"] = round(p, 4) if not np.isnan(p) else np.nan
                row[f"{regime_name}_n"]   = n
                sig = ""
                if not isinstance(row[f"{regime_name}_p値"], str) and not np.isnan(row[f"{regime_name}_p値"]):
                    if row[f"{regime_name}_p値"] < 0.01: sig = "★★"
                    elif row[f"{regime_name}_p値"] < 0.05: sig = "★"
                row[f"{regime_name}_有意"] = sig
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/merged_data.csv")
    parser.add_argument("--output", default="results/subsample_results.csv")
    parser.add_argument("--max-lag", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    data = pd.read_csv(args.data, index_col=0, parse_dates=True, encoding="utf-8-sig")
    print(f"データ: {len(data)}行 × {len(data.columns)}列")

    print("サブサンプル分析実行中...")
    result_df = run_subsample(data, args.max_lag)
    result_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    # 体制間で有意性が変わるペアをハイライト
    print("\n=== 体制転換が検出されたペア（全サンプル非有意→体制別有意）===")
    for _, row in result_df.iterrows():
        full_p = row.get("全サンプル_p値")
        if isinstance(full_p, float) and not np.isnan(full_p) and full_p >= 0.05:
            for regime in ["GFC危機期", "COVID危機期", "低金利期", "高金利期"]:
                rp = row.get(f"{regime}_p値")
                if isinstance(rp, float) and not np.isnan(rp) and rp < 0.05:
                    print(f"  {row['マクロ変数']:15s} → {row['ファクター']:8s}  "
                          f"全サンプル:p={full_p:.3f} → {regime}:p={rp:.3f} ★")

    print(f"\n✓ 保存: {args.output}")


if __name__ == "__main__":
    main()
