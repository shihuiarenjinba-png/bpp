"""
01_data_collection.py
=====================
FF5ファクターデータ（Kenneth French Data Library）と
マクロ経済変数（FRED）を取得し、月次パネルデータとして統合する。

データソース（信頼性の高い公的機関のみ）:
- Kenneth French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- FRED (Federal Reserve Bank of St. Louis): https://fred.stlouisfed.org/

使用方法:
    python 01_data_collection.py --start-year 2006 --end-year 2025 --output data/merged_data.csv

引数:
    --start-year   : サンプル開始年（デフォルト: 2006）
    --end-year     : サンプル終了年（デフォルト: 2025）
    --fred-api-key : FRED APIキー（デフォルト設定済み）
    --output       : 出力CSVパス（デフォルト: data/merged_data.csv）
"""

import argparse
import io
import os
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# ─── 定数 ──────────────────────────────────────────────────────────────────────

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)

# FRED系列ID -> 変数名
FRED_SERIES = {
    "CPIAUCSL":    "CPI_level",       # 消費者物価指数（都市部、全品目）
    "INDPRO":      "IP_level",        # 鉱工業生産指数
    "GS10":        "T10Y",            # 10年国債利回り（月次、%）
    "TB3MS":       "T3M",             # 3ヶ月T-Bill利回り（月次、%）
    "DBAA":        "BAA_yield",       # Moody's BAA格社債利回り
    "DAAA":        "AAA_yield",       # Moody's AAA格社債利回り
    "DTWEXBGS":    "DXY_level",       # 広義ドル指数（日次→月次末値）
    "DCOILWTICO":  "OIL_level",       # WTI原油スポット価格（日次→月次末値）
    "VIXCLS":      "VIX_level",       # VIX恐怖指数（日次→月次平均）
}

FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# ─── データ取得関数 ─────────────────────────────────────────────────────────────

def fetch_ff5(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Kenneth French Data LibraryからFF5月次データを取得する。

    Returns:
        DataFrame (index=DatetimeIndex 月末, columns=['Mkt-RF','SMB','HML','RMW','CMA','RF'])
        単位: % (パーセント表示のまま)
    """
    print(f"[1/3] FF5データ取得中: {FF5_URL}")
    resp = requests.get(FF5_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".CSV")][0]
        with zf.open(csv_name) as f:
            raw = f.read().decode("utf-8", errors="replace")

    # ヘッダー行の検出（"Mkt-RF"を含む行）
    lines = raw.splitlines()
    header_idx = next(i for i, l in enumerate(lines) if "Mkt-RF" in l)

    # 月次データ部分のみ読み込み（年次データ等を除外）
    # 月次セクション終端: 5桁でない行（年次は4桁）が現れるまで
    data_lines = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        first_col = stripped.split(",")[0].strip()
        # 月次データは YYYYMM の6桁
        if len(first_col) == 6 and first_col.isdigit():
            data_lines.append(stripped)
        elif first_col.isdigit() and len(first_col) == 4:
            # 年次データが始まったら終了
            break

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None,
        names=["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    )
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m")
    df = df.set_index("date")
    df.index = df.index + pd.offsets.MonthEnd(0)  # 月末に統一

    # 型変換・欠損処理
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    # サンプル期間でフィルタリング
    df = df[
        (df.index.year >= start_year) & (df.index.year <= end_year)
    ]

    print(f"    → {len(df)}行取得 ({df.index.min().strftime('%Y-%m')} 〜 {df.index.max().strftime('%Y-%m')})")
    return df


def fetch_fred_series(series_id: str, api_key: str | None = None) -> pd.Series:
    """
    FREDから単一の経済指標を取得する。

    Args:
        series_id : FREDの系列ID（例: 'CPIAUCSL'）
        api_key   : FRED APIキー（任意。なければブラウザ向けCSVエンドポイントを使用）

    Returns:
        pd.Series（index=DatetimeIndex）
    """
    if api_key:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={api_key}&file_type=json"
        )
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        obs = resp.json()["observations"]
        s = pd.Series(
            {o["date"]: float(o["value"]) if o["value"] != "." else np.nan for o in obs}
        )
        s.index = pd.to_datetime(s.index)
    else:
        url = f"{FRED_BASE_URL}?id={series_id}"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        s = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True).squeeze()
        s = pd.to_numeric(s, errors="coerce")

    s.name = series_id
    return s


def fetch_all_macro(start_year: int, end_year: int, api_key: str | None = None) -> pd.DataFrame:
    """
    全マクロ変数をFREDから取得し、月次に統合する。

    Returns:
        DataFrame（index=月末DatetimeIndex、columns=変数名）
    """
    print(f"[2/3] マクロ変数取得中（FRED）...")
    start_dt = pd.Timestamp(f"{start_year}-01-01")
    end_dt   = pd.Timestamp(f"{end_year}-12-31")

    monthly_frames = []

    for series_id, var_name in FRED_SERIES.items():
        print(f"    FRED: {series_id} ({var_name})")
        try:
            s = fetch_fred_series(series_id, api_key)
            s = s[(s.index >= start_dt) & (s.index <= end_dt)]

            # 日次データは月次へ変換
            if series_id in ("DTWEXBGS", "DCOILWTICO"):
                # 月末値を使用
                s = s.resample("ME").last()
            elif series_id == "VIXCLS":
                # 月次平均を使用
                s = s.resample("ME").mean()
            else:
                # 月次データはそのまま（月末に整形）
                s.index = s.index + pd.offsets.MonthEnd(0)
                s = s.groupby(s.index).last()

            s.name = var_name
            monthly_frames.append(s)
        except Exception as e:
            print(f"    ⚠️  {series_id} 取得失敗: {e}")
            print(f"    → 手動でFREDサイトからCSVをダウンロードしてください: https://fred.stlouisfed.org/series/{series_id}")

    macro_df = pd.concat(monthly_frames, axis=1)
    return macro_df


def transform_macro(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    マクロ変数を分析可能な形式に変換する。
    - 水準変数 → 前月比成長率（log差分）
    - 利回り・スプレッド → 水準または差分（定常性に基づく）

    Returns:
        変換後DataFrame（変数名は論文と整合）
    """
    result = pd.DataFrame(index=macro_df.index)

    # CPI成長率（前月比 log差分）
    if "CPI_level" in macro_df.columns:
        result["CPI_growth"] = np.log(macro_df["CPI_level"]).diff() * 100

    # 鉱工業生産成長率（前月比 log差分）
    if "IP_level" in macro_df.columns:
        result["IP_growth"] = np.log(macro_df["IP_level"]).diff() * 100

    # タームスプレッド（10年 - 3ヶ月、単位%）
    if "T10Y" in macro_df.columns and "T3M" in macro_df.columns:
        result["TERM_SPREAD"] = macro_df["T10Y"] - macro_df["T3M"]

    # デフォルトスプレッド（BAA - AAA、単位%）
    if "BAA_yield" in macro_df.columns and "AAA_yield" in macro_df.columns:
        result["DEF_SPREAD"] = macro_df["BAA_yield"] - macro_df["AAA_yield"]

    # 為替変動率（前月比 log差分）
    if "DXY_level" in macro_df.columns:
        result["FX_return"] = np.log(macro_df["DXY_level"]).diff() * 100

    # 原油価格変動率（前月比 log差分）
    if "OIL_level" in macro_df.columns:
        result["OIL_return"] = np.log(macro_df["OIL_level"]).diff() * 100

    # VIX（水準）
    if "VIX_level" in macro_df.columns:
        result["VIX"] = macro_df["VIX_level"]

    result = result.dropna(how="all")
    return result


# ─── ADF定常性検定 ───────────────────────────────────────────────────────────

def check_stationarity(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    ADF検定（Augmented Dickey-Fuller）で各系列の定常性を確認する。

    Returns:
        検定結果のDataFrame（変数名、ADF統計量、p値、定常性判定）
    """
    from statsmodels.tsa.stattools import adfuller

    results = []
    for col in df.columns:
        series = df[col].dropna()
        if len(series) < 20:
            continue
        try:
            adf_stat, p_val, _, _, critical_values, _ = adfuller(series, autolag="AIC")
            is_stationary = p_val < alpha
            results.append({
                "変数": col,
                "ADF統計量": round(adf_stat, 4),
                "p値": round(p_val, 4),
                "5%臨界値": round(critical_values["5%"], 4),
                "定常性（p<0.05）": "✓ 定常" if is_stationary else "✗ 非定常",
            })
        except Exception as e:
            results.append({"変数": col, "ADF統計量": None, "p値": None, "定常性": f"エラー: {e}"})

    return pd.DataFrame(results)


# ─── ADF自動修正 ────────────────────────────────────────────────────────────

# FF5ファクターは変換対象外（すでにリターン系列のため差分を取ると意味が変わる）
FF5_COLS = {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"}


def auto_correct_stationarity(
    df: pd.DataFrame,
    adf_results: pd.DataFrame,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    非定常と判定された系列に対して段階的に変換を試み、定常化を自動で試みる。

    変換の優先順位（各段階でADF再検定し、定常化できた時点で停止）:
        Stage 1 — 1階差分 : diff()
        Stage 2 — 対数1階差分 : log().diff()  ← 全値が正の場合のみ試みる

    FF5ファクター (Mkt-RF, SMB, HML, RMW, CMA, RF) は対象外。
    変換を適用した場合、列名に "_d1"（差分）または "_ld1"（対数差分）サフィックスを付与し、
    元の列を削除する。

    Parameters
    ----------
    df         : 統合済みDataFrame（FF5 + マクロ変数）
    adf_results: check_stationarity() の返り値
    alpha      : 有意水準（デフォルト0.05）

    Returns
    -------
    corrected_df   : 変換後DataFrame
    transform_log  : 変換ログDataFrame（変数ごとに何をしたか記録）
    """
    from statsmodels.tsa.stattools import adfuller

    # 非定常と判定された変数を特定（FF5は除外）
    non_stat_mask = (
        adf_results["定常性（p<0.05）"].str.contains("非定常", na=False)
        & ~adf_results["変数"].isin(FF5_COLS)
    )
    non_stat_vars = adf_results.loc[non_stat_mask, "変数"].tolist()

    if not non_stat_vars:
        print("\n✓ 非定常変数なし。変換不要です。")
        log_df = pd.DataFrame(columns=["変数", "元のp値", "適用変換", "変換後p値", "変換後定常性"])
        return df.copy(), log_df

    print(f"\n[自動修正] 非定常変数 {non_stat_vars} に対して変換を試みます...")

    corrected = df.copy()
    log_rows = []

    for var in non_stat_vars:
        if var not in corrected.columns:
            continue

        series = corrected[var].dropna()
        orig_row = adf_results[adf_results["変数"] == var].iloc[0]
        orig_pval = orig_row["p値"]

        applied = "変換なし（定常化不可）"
        final_pval = orig_pval
        final_stat = "✗ 非定常"

        # ── Stage 1: 1階差分 ──────────────────────────────────────────────
        s1 = series.diff().dropna()
        try:
            _, p1, *_ = adfuller(s1, autolag="AIC")
            if p1 < alpha:
                new_col = f"{var}_d1"
                corrected[new_col] = corrected[var].diff()
                corrected.drop(columns=[var], inplace=True)
                applied = f"1階差分 → {new_col}"
                final_pval = round(p1, 4)
                final_stat = "✓ 定常"
                print(f"    {var}: 1階差分で定常化 (p={p1:.4f}) → 列名: {new_col}")
                log_rows.append({
                    "変数": var,
                    "元のp値": orig_pval,
                    "適用変換": applied,
                    "変換後p値": final_pval,
                    "変換後定常性": final_stat,
                })
                continue  # 次の変数へ
        except Exception as e:
            print(f"    {var}: Stage1差分エラー: {e}")

        # ── Stage 2: 対数1階差分（全値が正の場合のみ） ────────────────────
        if (series > 0).all():
            s2 = np.log(series).diff().dropna()
            try:
                _, p2, *_ = adfuller(s2, autolag="AIC")
                if p2 < alpha:
                    new_col = f"{var}_ld1"
                    corrected[new_col] = np.log(corrected[var]).diff()
                    corrected.drop(columns=[var], inplace=True)
                    applied = f"対数1階差分 → {new_col}"
                    final_pval = round(p2, 4)
                    final_stat = "✓ 定常"
                    print(f"    {var}: 対数差分で定常化 (p={p2:.4f}) → 列名: {new_col}")
                else:
                    print(f"    ⚠️  {var}: 対数差分でも非定常 (p={p2:.4f})。元の系列を保持します。")
                    final_pval = round(p2, 4)
            except Exception as e:
                print(f"    {var}: Stage2対数差分エラー: {e}")
        else:
            print(f"    {var}: 負値を含むため対数変換をスキップ。元の系列を保持します。")

        log_rows.append({
            "変数": var,
            "元のp値": orig_pval,
            "適用変換": applied,
            "変換後p値": final_pval,
            "変換後定常性": final_stat,
        })

    transform_log = pd.DataFrame(log_rows)

    # 変換後に先頭行がNaNになるため再度dropna
    corrected = corrected.dropna(subset=[c for c in corrected.columns if c not in FF5_COLS], how="all")

    return corrected, transform_log


# ─── メイン処理 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FF5 + FRED データ取得スクリプト")
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year",   type=int, default=2025)
    parser.add_argument("--fred-api-key", type=str, default="a578d4c9bfdaabd0b502321ffaea965b",
                        help="FRED APIキー（デフォルト設定済み）")
    parser.add_argument("--output", type=str, default="data/merged_data.csv")
    parser.add_argument(
        "--no-auto-correct", action="store_true",
        help="ADF非定常検出時の自動変換を無効化する（手動で変換したい場合）",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # --- データ取得 ---
    ff5_df   = fetch_ff5(args.start_year, args.end_year)
    macro_raw = fetch_all_macro(args.start_year, args.end_year, args.fred_api_key)
    macro_df  = transform_macro(macro_raw)

    # --- 結合 ---
    print("[3/3] データを結合しています...")
    merged = ff5_df.join(macro_df, how="inner")
    merged = merged.dropna(subset=["Mkt-RF", "SMB", "HML", "RMW", "CMA"])

    # --- ADF検定レポート ---
    print("\n=== ADF定常性検定結果 ===")
    adf_results = check_stationarity(merged)
    print(adf_results.to_string(index=False))

    non_stationary = adf_results[adf_results["定常性（p<0.05）"].str.contains("非定常")]
    if len(non_stationary) > 0:
        print(f"\n⚠️ 非定常変数が検出されました: {non_stationary['変数'].tolist()}")

        if args.no_auto_correct:
            print("   --no-auto-correct が指定されているため、手動変換が必要です。")
        else:
            merged, transform_log = auto_correct_stationarity(merged, adf_results)

            # 変換ログの保存
            log_path = args.output.replace(".csv", "_transform_log.csv")
            transform_log.to_csv(log_path, index=False, encoding="utf-8-sig")
            print(f"\n✓ 変換ログを保存しました: {log_path}")

            # 変換後に再検定して確認
            print("\n=== 変換後 ADF再検定 ===")
            adf_after = check_stationarity(merged)
            print(adf_after.to_string(index=False))
            adf_results = adf_after  # 最終ADF結果を更新
    else:
        transform_log = None
        print("   全変数が定常です。変換不要です。")

    # --- 保存 ---
    merged.to_csv(args.output, index=True, encoding="utf-8-sig")   # index=日付列（読み込み側でindex_col=0）
    adf_results.to_csv(args.output.replace(".csv", "_adf_results.csv"), index=False, encoding="utf-8-sig")

    print(f"\n✓ 統合データを保存しました: {args.output}")
    print(f"  行数: {len(merged)}、列数: {len(merged.columns)}")
    print(f"  期間: {merged.index.min().strftime('%Y-%m')} 〜 {merged.index.max().strftime('%Y-%m')}")
    print(f"  列: {list(merged.columns)}")
    print(f"\n✓ ADF検定結果を保存しました: {args.output.replace('.csv', '_adf_results.csv')}")

    # --- 取得日時を記録（再現性のため） ---
    meta = {
        "取得日時": datetime.now().isoformat(),
        "FF5_URL": FF5_URL,
        "FRED_系列": FRED_SERIES,
        "サンプル開始": str(merged.index.min()),
        "サンプル終了": str(merged.index.max()),
        "観測数": len(merged),
        "自動変換": (
            transform_log.to_dict(orient="records")
            if transform_log is not None and len(transform_log) > 0
            else []
        ),
    }
    import json
    meta_path = args.output.replace(".csv", "_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"✓ メタデータ（取得日時・データソース）を保存: {meta_path}")


if __name__ == "__main__":
    main()
