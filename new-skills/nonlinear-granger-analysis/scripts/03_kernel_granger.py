"""
03_kernel_granger.py
=====================
カーネルベース非線形グレンジャー因果性検定（置換検定）
Diks & Panchenko (2006) の簡易実装。線形検定非有意ペアの非線形関係を探索。
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

FACTORS    = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
MACRO_VARS = ["CPI_growth", "IP_growth", "TERM_SPREAD", "DEF_SPREAD",
              "FX_return", "OIL_return", "VIX"]


def rbf_kernel(X, Y, bandwidth=None):
    """RBFカーネルによる密度比推定"""
    if bandwidth is None:
        bandwidth = np.median(np.abs(X - np.median(X))) * 1.06 * len(X)**(-0.2)
        bandwidth = max(bandwidth, 1e-6)
    diff = X[:, None] - Y[None, :]
    return np.exp(-0.5 * (diff / bandwidth)**2) / (bandwidth * np.sqrt(2*np.pi))


def nonlinear_granger_stat(y, x, lag=1):
    """
    非線形グレンジャー統計量（カーネル密度比ベース）
    H0: x_{t-lag}はy_tに情報をもたらさない（条件付き独立）
    """
    n = len(y)
    T = n - lag
    if T < 20: return np.nan

    y_curr  = y[lag:]          # y_t
    y_past  = y[lag-1:n-1]     # y_{t-1}
    x_past  = x[lag-1:n-1]     # x_{t-lag}

    # 標準化
    def zscore(v): return (v - np.mean(v)) / (np.std(v) + 1e-10)
    y_c = zscore(y_curr); y_p = zscore(y_past); x_p = zscore(x_past)

    # 条件付き期待値の比較（カーネル回帰）
    h = T**(-1/5) * 1.5  # bandwidth

    # E[y_t | y_{t-1}]（制約モデル）
    K_yp = rbf_kernel(y_p, y_p, h)
    np.fill_diagonal(K_yp, 0)
    weights_r = K_yp / (K_yp.sum(axis=1, keepdims=True) + 1e-10)
    yhat_r = weights_r @ y_c

    # E[y_t | y_{t-1}, x_{t-lag}]（非制約モデル）
    K_joint = rbf_kernel(y_p, y_p, h) * rbf_kernel(x_p, x_p, h)
    np.fill_diagonal(K_joint, 0)
    weights_u = K_joint / (K_joint.sum(axis=1, keepdims=True) + 1e-10)
    yhat_u = weights_u @ y_c

    # 改善量（統計量）
    resid_r = y_c - yhat_r
    resid_u = y_c - yhat_u
    stat = (np.mean(resid_r**2) - np.mean(resid_u**2)) / (np.std(resid_r**2) / np.sqrt(T) + 1e-10)
    return float(stat)


def permutation_test(y, x, lag=1, n_perm=199, seed=42):
    """置換検定でp値を計算"""
    rng = np.random.default_rng(seed)
    obs_stat = nonlinear_granger_stat(y, x, lag)
    if np.isnan(obs_stat): return np.nan, np.nan

    perm_stats = []
    for _ in range(n_perm):
        x_perm = rng.permutation(x)
        s = nonlinear_granger_stat(y, x_perm, lag)
        if not np.isnan(s): perm_stats.append(s)

    if len(perm_stats) == 0: return obs_stat, np.nan
    p_val = np.mean(np.array(perm_stats) >= obs_stat)
    return obs_stat, float(p_val)


def run_kernel_granger(data, n_perm=199, lag=1):
    print(f"カーネルグレンジャー検定（置換数={n_perm}）実行中...")
    rows = []
    total = sum(1 for mv in MACRO_VARS if mv in data.columns
                for f in FACTORS if f in data.columns)
    count = 0
    for mv in MACRO_VARS:
        if mv not in data.columns: continue
        for factor in FACTORS:
            if factor not in data.columns: continue
            count += 1
            subset = data[[factor, mv]].dropna()
            stat, p = permutation_test(subset[factor].values, subset[mv].values, lag, n_perm)
            rows.append({
                "マクロ変数": mv, "ファクター": factor,
                "ラグ": lag, "観測数": len(subset),
                "カーネルT統計量": round(stat, 4) if not np.isnan(stat) else np.nan,
                "置換p値": round(p, 4) if not np.isnan(p) else np.nan,
                "有意（p<0.05）": "★" if (not np.isnan(p) and p < 0.05) else "",
            })
            ps = f"{p:.3f}" if not np.isnan(p) else "N/A"
            print(f"  [{count}/{total}] {mv:15s} → {factor:8s} | T={stat:.2f if not np.isnan(stat) else 0:.2f} | p={ps}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",           default="data/merged_data.csv")
    parser.add_argument("--n-permutations", type=int, default=199)
    parser.add_argument("--lag",            type=int, default=1)
    parser.add_argument("--output",         default="results/kernel_results.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    data = pd.read_csv(args.data, index_col=0, parse_dates=True, encoding="utf-8-sig")

    result_df = run_kernel_granger(data, args.n_permutations, args.lag)
    result_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    sig = result_df[result_df["有意（p<0.05）"] == "★"]
    print(f"\n★ 非線形グレンジャー有意ペア: {len(sig)}件")
    if len(sig) > 0:
        print(sig[["マクロ変数","ファクター","カーネルT統計量","置換p値"]].to_string(index=False))
    print(f"\n✓ 保存: {args.output}")


if __name__ == "__main__":
    main()
