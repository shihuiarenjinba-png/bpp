"""
02_granger_analysis.py
======================
Fama-French 5ファクターとマクロ経済変数のグレンジャー因果性検定を実施する。

検定設計:
- 対象: 7マクロ変数 × 5ファクター = 35通りのペア
- ラグ選択: BIC（ベイズ情報量規準）による最適ラグ決定（最大6ヶ月）
- 検定統計量: F統計量
- 多重検定補正: Bonferroni補正（補正後有意水準 = 0.05 / 35 ≈ 0.0014）

使用方法:
    python 02_granger_analysis.py \
        --data data/merged_data.csv \
        --max-lag 6 \
        --alpha 0.05 \
        --output results/granger_results.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")

# ─── 定数 ──────────────────────────────────────────────────────────────────────

FACTORS   = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
MACRO_VARS = ["CPI_growth", "IP_growth", "TERM_SPREAD", "DEF_SPREAD",
              "FX_return", "OIL_return", "VIX"]

# VIXは1990年以降のみ利用可能（サブサンプル）
VIX_START = pd.Timestamp("1990-01-01")


# ─── グレンジャー検定 ────────────────────────────────────────────────────────

def select_lag_bic(y: pd.Series, x: pd.Series, max_lag: int) -> int:
    """
    BIC（ベイズ情報量規準）に基づいて最適ラグ次数を選択する。

    Args:
        y       : 目的変数（FFファクター）
        x       : 説明変数（マクロ変数）
        max_lag : 探索する最大ラグ次数

    Returns:
        BIC最小化ラグ次数（1〜max_lag）
    """
    df = pd.concat([y, x], axis=1).dropna()
    bics = []
    for lag in range(1, max_lag + 1):
        try:
            model = VAR(df)
            result = model.fit(lag, ic=None)
            bics.append((lag, result.bic))
        except Exception:
            bics.append((lag, np.inf))
    best_lag = min(bics, key=lambda t: t[1])[0]
    return best_lag


def run_granger_test(
    y: pd.Series,
    x: pd.Series,
    lag: int,
) -> dict:
    """
    単一ペアのグレンジャー因果性検定を実行する（F統計量を使用）。

    H0: xのラグ値はyの予測に統計的に有用でない（xはyをグレンジャー原因しない）

    Args:
        y   : 目的変数（FFファクター）
        x   : 説明変数（マクロ変数）
        lag : ラグ次数

    Returns:
        {'f_stat': float, 'p_value': float, 'lag': int, 'n_obs': int}
    """
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < lag * 3 + 10:
        return {"f_stat": np.nan, "p_value": np.nan, "lag": lag, "n_obs": len(df)}

    try:
        results = grangercausalitytests(df[[y.name, x.name]], maxlag=lag, verbose=False)
        # ラグ次数でのF検定結果を取得
        f_stat = results[lag][0]["ssr_ftest"][0]
        p_val  = results[lag][0]["ssr_ftest"][1]
        return {"f_stat": round(f_stat, 4), "p_value": round(p_val, 6), "lag": lag, "n_obs": len(df)}
    except Exception as e:
        return {"f_stat": np.nan, "p_value": np.nan, "lag": lag, "n_obs": len(df), "error": str(e)}


def run_all_granger(data: pd.DataFrame, max_lag: int = 6) -> pd.DataFrame:
    """
    全35通りのグレンジャー因果性検定を実行する。

    Returns:
        DataFrame（行=マクロ変数、列=ファクター、値=p値）+ 詳細情報
    """
    print(f"\n=== グレンジャー因果性検定（最大ラグ={max_lag}ヶ月、BIC選択） ===")
    print(f"  対象: {len(MACRO_VARS)} マクロ変数 × {len(FACTORS)} ファクター = {len(MACRO_VARS)*len(FACTORS)} ペア")

    rows = []
    for macro_var in MACRO_VARS:
        if macro_var not in data.columns:
            print(f"  ⚠️  {macro_var} がデータに存在しません（スキップ）")
            continue

        for factor in FACTORS:
            if factor not in data.columns:
                continue

            # VIXはサブサンプルを使用
            if macro_var == "VIX":
                subset = data[[factor, macro_var]].loc[data.index >= VIX_START].dropna()
            else:
                subset = data[[factor, macro_var]].dropna()

            y = subset[factor]
            x = subset[macro_var]

            # BICでラグ選択
            best_lag = select_lag_bic(y, x, max_lag)

            # グレンジャー検定
            result = run_granger_test(y, x, best_lag)

            row = {
                "マクロ変数": macro_var,
                "ファクター": factor,
                "BIC選択ラグ": best_lag,
                "F統計量": result.get("f_stat"),
                "p値": result.get("p_value"),
                "観測数": result.get("n_obs"),
            }
            rows.append(row)

            p_str = f"{result.get('p_value', 'N/A'):.4f}" if result.get("p_value") is not None else "N/A"
            print(f"  {macro_var:15s} → {factor:8s} | ラグ={best_lag} | p={p_str}")

    return pd.DataFrame(rows)


def add_significance_flags(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Bonferroni補正と補正前の有意性フラグを追加する。

    Args:
        df    : run_all_granger()の出力DataFrame
        alpha : 有意水準（デフォルト0.05）
    """
    n_tests = len(df.dropna(subset=["p値"]))
    alpha_bonf = alpha / n_tests

    df = df.copy()
    df["補正前有意（p<0.05）"]  = df["p値"] < alpha
    df["Bonferroni補正後有意"] = df["p値"] < alpha_bonf
    df["Bonferroni補正後α"]   = round(alpha_bonf, 6)
    df["総検定数"]              = n_tests

    # 有意度マーク
    def significance_mark(row):
        if pd.isna(row["p値"]):
            return ""
        if row["Bonferroni補正後有意"]:
            return "★ (Bonferroni補正後も有意)"
        if row["補正前有意（p<0.05）"]:
            return "* (補正前のみ有意)"
        return ""

    df["有意度"] = df.apply(significance_mark, axis=1)
    return df


def make_p_value_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    p値マトリックス（行=マクロ変数、列=ファクター）を作成する。
    論文の表4.3に対応。
    """
    pivot = df.pivot_table(index="マクロ変数", columns="ファクター", values="p値", aggfunc="first")
    # 列順を論文と同じに
    cols = [c for c in FACTORS if c in pivot.columns]
    pivot = pivot[cols]
    return pivot.round(4)


# ─── メイン処理 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="グレンジャー因果性検定")
    parser.add_argument("--data",    type=str, default="data/merged_data.csv")
    parser.add_argument("--max-lag", type=int, default=6)
    parser.add_argument("--alpha",   type=float, default=0.05)
    parser.add_argument("--output",  type=str, default="results/granger_results.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # --- データ読み込み ---
    print(f"データ読み込み: {args.data}")
    data = pd.read_csv(args.data, index_col=0, parse_dates=True, encoding="utf-8-sig")
    print(f"  {len(data)}行 × {len(data.columns)}列")

    # --- グレンジャー検定 ---
    results_df = run_all_granger(data, args.max_lag)
    results_df = add_significance_flags(results_df, args.alpha)

    # --- p値マトリックス ---
    p_matrix = make_p_value_matrix(results_df)

    # --- 結果表示 ---
    print("\n=== p値マトリックス（論文 表4.3） ===")
    print(p_matrix.to_string())

    n_total = len(results_df.dropna(subset=["p値"]))
    alpha_bonf = args.alpha / n_total
    print(f"\nBonferroni補正後有意水準: {alpha_bonf:.4f} (= {args.alpha}/{n_total})")

    bonf_sig = results_df[results_df["Bonferroni補正後有意"] == True]
    print(f"\n★ Bonferroni補正後も有意な結果: {len(bonf_sig)}件")
    if len(bonf_sig) > 0:
        print(bonf_sig[["マクロ変数", "ファクター", "BIC選択ラグ", "F統計量", "p値"]].to_string(index=False))

    pre_sig = results_df[(results_df["補正前有意（p<0.05）"] == True) & (results_df["Bonferroni補正後有意"] == False)]
    print(f"\n* 補正前のみ有意な結果（参考）: {len(pre_sig)}件")
    if len(pre_sig) > 0:
        print(pre_sig[["マクロ変数", "ファクター", "BIC選択ラグ", "F統計量", "p値"]].to_string(index=False))

    # --- 保存 ---
    results_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    p_matrix.to_csv(args.output.replace(".csv", "_pmatrix.csv"), encoding="utf-8-sig", index=True)

    print(f"\n✓ 詳細結果を保存: {args.output}")
    print(f"✓ p値マトリックスを保存: {args.output.replace('.csv', '_pmatrix.csv')}")


if __name__ == "__main__":
    main()
