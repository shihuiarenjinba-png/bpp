"""
03_random_forest.py
===================
ランダムフォレストを用いてFF5ファクターのアウトオブサンプル予測を実施し、
特徴量重要度によって遅行構造を探索する。

分析設計:
- 説明変数: 7マクロ変数 × 4ラグ（1,3,6,12ヶ月）= 28特徴量
- 目的変数: 各FFファクター（Mkt-RF, SMB, HML, RMW, CMA）
- サンプル分割: ウォークフォワード検証（初期70%訓練、残り30%テスト）
- 評価指標: OOS R²、方向的一致率（Directional Accuracy）
- 再現性: random_state=42 を固定

使用方法:
    python 03_random_forest.py \
        --data data/merged_data.csv \
        --train-ratio 0.7 \
        --n-estimators 500 \
        --output results/rf_results.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ─── 定数 ──────────────────────────────────────────────────────────────────────

FACTORS    = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
MACRO_VARS = ["CPI_growth", "IP_growth", "TERM_SPREAD", "DEF_SPREAD",
              "FX_return", "OIL_return", "VIX"]
LAGS       = [1, 3, 6, 12]

RANDOM_STATE = 42  # 再現性のため固定

VIX_START = pd.Timestamp("1990-01-01")


# ─── 特徴量生成 ──────────────────────────────────────────────────────────────

def create_lag_features(data: pd.DataFrame, lags: list[int] = LAGS) -> pd.DataFrame:
    """
    マクロ変数のラグ付き特徴量を生成する（28特徴量）。

    Returns:
        DataFrame（元データ + ラグ特徴量）
    """
    result = data.copy()
    for var in MACRO_VARS:
        if var not in data.columns:
            continue
        for lag in lags:
            result[f"{var}_lag{lag}"] = data[var].shift(lag)
    return result


def get_feature_names(macro_vars: list[str] = MACRO_VARS, lags: list[int] = LAGS) -> list[str]:
    """特徴量名のリストを返す（VIXはサブサンプル専用）。"""
    return [f"{var}_lag{lag}" for var in macro_vars for lag in lags]


# ─── OOS R²（アウトオブサンプル決定係数）────────────────────────────────────

def oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    アウトオブサンプルR²（Campbell and Thompson, 2008の定義）を計算する。

    R²_OOS = 1 - Σ(y_t - ŷ_t)² / Σ(y_t - ȳ_hist)²

    ȳ_histはその時点までのヒストリカル平均（ウォークフォワード）

    注: 単純なsklearn.r2_scoreではなく、ヒストリカル平均をベンチマークとして使用。
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    benchmark = np.mean(y_true)  # ヒストリカル平均をベンチマーク
    ss_tot = np.sum((y_true - benchmark) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    方向的一致率（Directional Accuracy）を計算する。
    符号が一致した割合（50%超でランダムより良い予測）。
    """
    correct = np.sign(y_true) == np.sign(y_pred)
    return np.mean(correct)


# ─── ランダムフォレスト（ウォークフォワード）────────────────────────────────

def run_random_forest_factor(
    data_with_lags: pd.DataFrame,
    factor: str,
    train_ratio: float = 0.7,
    n_estimators: int = 500,
    include_vix: bool = True,
) -> dict:
    """
    単一ファクターのランダムフォレスト分析を実行する。

    Args:
        data_with_lags : ラグ特徴量付きDataFrame
        factor         : 目的変数（FFファクター名）
        train_ratio    : 訓練データ比率
        n_estimators   : 決定木の本数
        include_vix    : VIX特徴量を含めるか

    Returns:
        分析結果のdict
    """
    all_features = get_feature_names()
    if not include_vix:
        all_features = [f for f in all_features if "VIX" not in f]

    # 利用可能な特徴量のみ使用
    features = [f for f in all_features if f in data_with_lags.columns]

    # 欠損を除いた完全なデータ
    subset = data_with_lags[[factor] + features].dropna()

    if len(subset) < 50:
        return {
            "factor": factor,
            "n_obs": len(subset),
            "oos_r2": np.nan,
            "dir_accuracy": np.nan,
            "note": "観測数不足",
        }

    # 時系列分割（ウォークフォワード）
    split_idx = int(len(subset) * train_ratio)
    train = subset.iloc[:split_idx]
    test  = subset.iloc[split_idx:]

    X_train = train[features].values
    y_train = train[factor].values
    X_test  = test[features].values
    y_test  = test[factor].values

    # ランダムフォレスト学習
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",  # 標準設定
    )
    rf.fit(X_train, y_train)

    # 予測・評価
    y_pred = rf.predict(X_test)
    oos_r2_val   = oos_r2(y_test, y_pred)
    dir_acc      = directional_accuracy(y_test, y_pred)

    # 特徴量重要度
    importance_df = pd.DataFrame({
        "feature":    features,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "factor":         factor,
        "n_train":        len(train),
        "n_test":         len(test),
        "oos_r2":         round(oos_r2_val, 4),
        "dir_accuracy":   round(dir_acc, 4),
        "dir_acc_pct":    f"{dir_acc*100:.1f}%",
        "top5_features":  importance_df.head(5)["feature"].tolist(),
        "importance_df":  importance_df,
        "include_vix":    include_vix,
    }


def run_all_factors(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    n_estimators: int = 500,
) -> tuple[pd.DataFrame, dict]:
    """
    全5ファクターのランダムフォレスト分析を実行する。

    Returns:
        (summary_df, importance_dict)
        summary_df: 各ファクターの評価指標サマリー
        importance_dict: ファクター別の特徴量重要度DataFrame
    """
    print(f"\n=== ランダムフォレスト分析（n_estimators={n_estimators}、訓練比率={train_ratio}） ===")
    print(f"  特徴量: {len(MACRO_VARS)}変数 × {len(LAGS)}ラグ = {len(MACRO_VARS)*len(LAGS)}特徴量")
    print(f"  random_state={RANDOM_STATE}（再現性固定）\n")

    # ラグ特徴量を生成
    data_with_lags = create_lag_features(data)

    summary_rows = []
    importance_dict = {}

    for factor in FACTORS:
        if factor not in data.columns:
            print(f"  ⚠️  {factor} がデータに存在しません（スキップ）")
            continue

        # メインサンプル（VIXなし）
        result = run_random_forest_factor(
            data_with_lags, factor, train_ratio, n_estimators, include_vix=False
        )

        # VIXサブサンプル（1990年以降）
        data_vix = data_with_lags.loc[data_with_lags.index >= VIX_START]
        result_vix = run_random_forest_factor(
            data_vix, factor, train_ratio, n_estimators, include_vix=True
        )

        print(f"  {factor}:")
        print(f"    [全期間, VIXなし] OOS R²={result['oos_r2']:.4f}, 方向的一致率={result['dir_acc_pct']}")
        print(f"    [1990〜, VIX含む] OOS R²={result_vix['oos_r2']:.4f}, 方向的一致率={result_vix['dir_acc_pct']}")
        if result.get("top5_features"):
            print(f"    Top-5特徴量: {result['top5_features']}")

        summary_rows.append({
            "ファクター":          factor,
            "訓練数（全期間）":    result.get("n_train"),
            "テスト数（全期間）":  result.get("n_test"),
            "OOS_R2（全期間）":   result.get("oos_r2"),
            "方向的一致率（全）":  result.get("dir_accuracy"),
            "OOS_R2（VIX含む）":  result_vix.get("oos_r2"),
            "方向的一致率（VIX）": result_vix.get("dir_accuracy"),
            "Top1特徴量":         result["top5_features"][0] if result.get("top5_features") else None,
        })

        if result.get("importance_df") is not None:
            importance_dict[factor] = result["importance_df"]

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, importance_dict


def build_importance_comparison(importance_dict: dict) -> pd.DataFrame:
    """
    ファクター間での特徴量重要度比較表を作成する。
    論文の考察に使用する遅行構造の比較表。
    """
    rows = []
    for factor, imp_df in importance_dict.items():
        for rank, (_, row) in enumerate(imp_df.iterrows(), 1):
            feature = row["feature"]
            # 変数名とラグを分解
            parts = feature.rsplit("_lag", 1)
            if len(parts) == 2:
                var_name = parts[0]
                lag_num  = int(parts[1])
            else:
                var_name = feature
                lag_num  = 0

            rows.append({
                "ファクター":   factor,
                "順位":         rank,
                "特徴量":       feature,
                "マクロ変数":   var_name,
                "ラグ（ヶ月）": lag_num,
                "重要度":       round(row["importance"], 6),
            })

    return pd.DataFrame(rows)


# ─── メイン処理 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ランダムフォレスト分析")
    parser.add_argument("--data",         type=str,   default="data/merged_data.csv")
    parser.add_argument("--train-ratio",  type=float, default=0.7)
    parser.add_argument("--n-estimators", type=int,   default=500)
    parser.add_argument("--output",       type=str,   default="results/rf_results.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # --- データ読み込み ---
    print(f"データ読み込み: {args.data}")
    data = pd.read_csv(args.data, index_col=0, parse_dates=True, encoding="utf-8-sig")

    # --- 分析実行 ---
    summary_df, importance_dict = run_all_factors(
        data,
        train_ratio=args.train_ratio,
        n_estimators=args.n_estimators,
    )

    # --- サマリー表示 ---
    print("\n=== 結果サマリー ===")
    print(summary_df[["ファクター", "OOS_R2（全期間）", "方向的一致率（全）",
                        "OOS_R2（VIX含む）", "方向的一致率（VIX）"]].to_string(index=False))
    print("\n補足:")
    print("  OOS R² > 0 → ヒストリカル平均より優れた予測")
    print("  方向的一致率 > 0.5 → ランダムより良い予測")

    # --- 遅行構造の比較 ---
    imp_comparison = build_importance_comparison(importance_dict)

    # ファクター別Top-10の平均ラグを計算（遅行度の指標）
    print("\n=== ファクター別 Top-10特徴量の平均ラグ（遅行度指標） ===")
    for factor in FACTORS:
        sub = imp_comparison[imp_comparison["ファクター"] == factor].head(10)
        if len(sub) > 0:
            avg_lag = sub["ラグ（ヶ月）"].mean()
            top_var = sub.iloc[0]["マクロ変数"]
            top_lag = sub.iloc[0]["ラグ（ヶ月）"]
            print(f"  {factor:8s}: 平均ラグ={avg_lag:.1f}ヶ月, 最重要={top_var}（lag{top_lag}）")

    # --- 保存 ---
    summary_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    imp_path = args.output.replace(".csv", "_feature_importance.csv")
    imp_comparison.to_csv(imp_path, index=False, encoding="utf-8-sig")

    # ファクター別の重要度も個別保存
    for factor, imp_df in importance_dict.items():
        factor_path = args.output.replace(".csv", f"_importance_{factor}.csv")
        imp_df.to_csv(factor_path, index=False, encoding="utf-8-sig")

    print(f"\n✓ 評価サマリーを保存: {args.output}")
    print(f"✓ 特徴量重要度（全ファクター比較）を保存: {imp_path}")


if __name__ == "__main__":
    main()
