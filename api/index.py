from __future__ import annotations

from html import escape
from pathlib import Path
import sys
from typing import Callable
from urllib.parse import parse_qs

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from shared_finance.accounting_engine import (
    build_accounting_summary,
    build_budget_template,
    build_budget_vs_actual,
    build_cash_settings,
    build_source_manifest,
    demo_transactions,
    normalize_transactions,
    parse_manual_table,
)
from shared_finance.behavioral_engine import simulate_behavioral_gap
from shared_finance.factor_data_loader import load_factor_dataset
from shared_finance.factor_engine import (
    compute_factor_forecast,
    generate_sample_factor_dataset,
    prepare_factor_frame,
    simulate_factor_outlook,
)
from shared_finance.policy_engine import simulate_policy_paths


def _get_arg(params: dict[str, list[str]], key: str, default: str) -> str:
    values = params.get(key)
    if not values:
        return default
    return values[0]


def _get_int(params: dict[str, list[str]], key: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        value = int(float(_get_arg(params, key, str(default))))
    except Exception:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _get_float(
    params: dict[str, list[str]],
    key: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        value = float(_get_arg(params, key, str(default)))
    except Exception:
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _get_bool(params: dict[str, list[str]], key: str, default: bool = False) -> bool:
    value = _get_arg(params, key, "1" if default else "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _fmt_num(value: float) -> str:
    return f"{value:,.2f}"


def _frame_to_html(frame: pd.DataFrame, *, max_rows: int = 12) -> str:
    preview = frame.head(max_rows).copy()
    for col in preview.columns:
        if pd.api.types.is_numeric_dtype(preview[col]):
            preview[col] = preview[col].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
        elif pd.api.types.is_datetime64_any_dtype(preview[col]):
            preview[col] = preview[col].dt.strftime("%Y-%m-%d")
    return preview.to_html(index=False, classes="data-table", border=0, escape=True)


def _render_metrics(metrics: list[tuple[str, str]]) -> str:
    cards = []
    for label, value in metrics:
        cards.append(
            f"<div class='metric-card'><div class='metric-label'>{escape(label)}</div><div class='metric-value'>{escape(value)}</div></div>"
        )
    return "<div class='metric-grid'>" + "".join(cards) + "</div>"


def _render_line_chart(title: str, series_list: list[tuple[str, pd.Series, str]], *, percent: bool = False) -> str:
    clean_series = []
    for name, series, color in series_list:
        cleaned = pd.to_numeric(series, errors="coerce").dropna()
        if cleaned.empty:
            continue
        clean_series.append((name, cleaned.reset_index(drop=True), color))
    if not clean_series:
        return ""

    width = 860
    height = 280
    padding = 28
    all_values = pd.concat([series for _, series, _ in clean_series], ignore_index=True)
    min_y = float(all_values.min())
    max_y = float(all_values.max())
    if abs(max_y - min_y) < 1e-9:
        max_y += 1.0
        min_y -= 1.0

    def project_x(index: int, total: int) -> float:
        if total <= 1:
            return padding
        return padding + (width - 2 * padding) * (index / (total - 1))

    def project_y(value: float) -> float:
        return height - padding - (height - 2 * padding) * ((value - min_y) / (max_y - min_y))

    polylines = []
    legend = []
    for name, series, color in clean_series:
        points = " ".join(
            f"{project_x(idx, len(series)):.1f},{project_y(float(val)):.1f}"
            for idx, val in enumerate(series)
        )
        polylines.append(f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{points}' />")
        legend.append(f"<span><i style='background:{color}'></i>{escape(name)}</span>")

    axis_labels = (
        f"<div class='chart-axis'><span>{escape(_fmt_pct(max_y) if percent else _fmt_num(max_y))}</span>"
        f"<span>{escape(_fmt_pct(min_y) if percent else _fmt_num(min_y))}</span></div>"
    )

    return (
        "<div class='chart-card'>"
        f"<h3>{escape(title)}</h3>"
        f"<svg viewBox='0 0 {width} {height}' class='chart-svg'>"
        f"<line x1='{padding}' y1='{height - padding}' x2='{width - padding}' y2='{height - padding}' stroke='rgba(30,41,59,0.2)' stroke-width='1' />"
        f"<line x1='{padding}' y1='{padding}' x2='{padding}' y2='{height - padding}' stroke='rgba(30,41,59,0.2)' stroke-width='1' />"
        + "".join(polylines)
        + "</svg>"
        + axis_labels
        + "<div class='chart-legend'>"
        + "".join(legend)
        + "</div></div>"
    )


def _nav(active: str) -> str:
    items = [
        ("behavioral", "Behavioral Gap"),
        ("factor", "Factor Forecast"),
        ("policy", "Monetary Policy"),
        ("accounting", "Accounting Workbench"),
    ]
    links = []
    for key, label in items:
        cls = "nav-link active" if key == active else "nav-link"
        links.append(f"<a class='{cls}' href='/?tool={key}'>{escape(label)}</a>")
    return "<div class='nav-row'>" + "".join(links) + "</div>"


def _render_behavioral(params: dict[str, list[str]]) -> str:
    initial_price = _get_float(params, "initial_price", 100.0, minimum=10.0)
    expected_return_pct = _get_float(params, "expected_return_pct", 7.5, minimum=-5.0, maximum=20.0)
    volatility_pct = _get_float(params, "volatility_pct", 16.0, minimum=5.0, maximum=45.0)
    years = _get_int(params, "years", 7, minimum=2, maximum=15)
    n_sims = _get_int(params, "n_sims", 800, minimum=200, maximum=1500)
    overreaction = _get_float(params, "overreaction", 1.2, minimum=0.0, maximum=2.5)
    loss_aversion = _get_float(params, "loss_aversion", 1.0, minimum=0.0, maximum=2.5)
    herding = _get_float(params, "herding", 0.8, minimum=0.0, maximum=2.5)
    anchoring = _get_float(params, "anchoring", 0.6, minimum=0.0, maximum=2.5)
    sentiment_beta = _get_float(params, "sentiment_beta", 0.9, minimum=0.0, maximum=2.5)
    seed = _get_int(params, "seed", 42, minimum=1)

    result = simulate_behavioral_gap(
        initial_price=initial_price,
        expected_return=expected_return_pct / 100.0,
        volatility=volatility_pct / 100.0,
        years=years,
        n_sims=n_sims,
        overreaction=overreaction,
        loss_aversion=loss_aversion,
        herding=herding,
        anchoring=anchoring,
        sentiment_beta=sentiment_beta,
        seed=seed,
    )

    comparison = result["comparison"]
    diagnostics = result["diagnostics"]
    fan_chart = result["fan_chart"]
    divergence = result["divergence"]

    form = f"""
    <form class="control-grid" method="get">
      <input type="hidden" name="tool" value="behavioral">
      <label>初期価格<input name="initial_price" type="number" step="1" value="{initial_price:.0f}"></label>
      <label>期待年率リターン (%)<input name="expected_return_pct" type="number" step="0.1" value="{expected_return_pct:.1f}"></label>
      <label>年率ボラティリティ (%)<input name="volatility_pct" type="number" step="0.1" value="{volatility_pct:.1f}"></label>
      <label>年数<input name="years" type="number" step="1" value="{years}"></label>
      <label>本数<input name="n_sims" type="number" step="100" value="{n_sims}"></label>
      <label>乱数シード<input name="seed" type="number" step="1" value="{seed}"></label>
      <label>過剰反応<input name="overreaction" type="number" step="0.1" value="{overreaction:.1f}"></label>
      <label>損失回避<input name="loss_aversion" type="number" step="0.1" value="{loss_aversion:.1f}"></label>
      <label>群集行動<input name="herding" type="number" step="0.1" value="{herding:.1f}"></label>
      <label>アンカリング<input name="anchoring" type="number" step="0.1" value="{anchoring:.1f}"></label>
      <label>センチメント感応度<input name="sentiment_beta" type="number" step="0.1" value="{sentiment_beta:.1f}"></label>
      <div class="actions"><button type="submit">シミュレーションを更新</button></div>
    </form>
    """

    metrics = _render_metrics(
        [
            ("行動ファイナンス期待年率", _fmt_pct(float(comparison.iloc[1]["expected_return"]))),
            ("行動モデル平均最大DD", _fmt_pct(float(comparison.iloc[1]["avg_max_drawdown"]))),
            ("終値ギャップ > 10%", _fmt_pct(float(diagnostics["gap_probability_gt_10pct"]))),
            ("終値ギャップ < -10%", _fmt_pct(float(diagnostics["gap_probability_lt_minus_10pct"]))),
        ]
    )
    chart_one = _render_line_chart(
        "平均パスの比較",
        [
            ("Traditional", fan_chart["rational_mean"], "#0e7c86"),
            ("Behavioral", fan_chart["behavioral_mean"], "#d56c1b"),
        ],
    )
    chart_two = _render_line_chart(
        "価格乖離の推移",
        [("Relative gap", divergence["relative_gap"] * 100.0, "#3056c9")],
        percent=False,
    )

    return (
        "<section class='panel'><div class='kicker'>Behavioral Gap</div><h2>伝統モデルと行動モデルの差を実行する</h2>"
        "<p class='copy'>価格形成のクセを入れたときに、平均パスや終値の分布がどれだけずれるかを見ます。</p>"
        + form
        + metrics
        + chart_one
        + chart_two
        + "<div class='table-card'><h3>モデル比較</h3>"
        + _frame_to_html(comparison)
        + "</div></section>"
    )


def _render_factor(params: dict[str, list[str]]) -> str:
    source = _get_arg(params, "source", "demo")
    region = _get_arg(params, "region", "Japan")
    horizon_months = _get_int(params, "horizon_months", 6, minimum=1, maximum=12)
    temperature = _get_float(params, "temperature", 0.65, minimum=0.2, maximum=1.2)
    n_sims = _get_int(params, "n_sims", 1200, minimum=500, maximum=2500)
    seed = _get_int(params, "seed", 42, minimum=1)

    raw_factor_data = None
    load_note = ""
    if source == "live":
        try:
            live = load_factor_dataset(region=region, start_date="2005-01-01", end_date="2026-03-01")
            raw_factor_data = live.reset_index().rename(columns={live.index.name or "index": "Date"})
            load_note = f"{region} の Ken French データを利用"
        except Exception as exc:
            raw_factor_data = generate_sample_factor_dataset()
            load_note = f"ライブ取得に失敗したためサンプルへ切り替え: {exc}"
    else:
        raw_factor_data = generate_sample_factor_dataset()
        load_note = "サンプルデータを利用"

    factor_df = prepare_factor_frame(raw_factor_data)
    forecast_df, regime_df = compute_factor_forecast(
        factor_df,
        horizon_months=horizon_months,
        temperature=temperature,
    )
    simulation = simulate_factor_outlook(
        factor_df,
        forecast_df,
        horizon_months=horizon_months,
        n_sims=n_sims,
        seed=seed,
    )

    top_row = forecast_df.iloc[0]
    summary = simulation["summary"]
    regime_tail = regime_df.tail(48)

    form = f"""
    <form class="control-grid" method="get">
      <input type="hidden" name="tool" value="factor">
      <label>データソース
        <select name="source">
          <option value="demo" {'selected' if source == 'demo' else ''}>Demo sample</option>
          <option value="live" {'selected' if source == 'live' else ''}>Live Ken French</option>
        </select>
      </label>
      <label>地域
        <select name="region">
          <option value="US" {'selected' if region == 'US' else ''}>US</option>
          <option value="Japan" {'selected' if region == 'Japan' else ''}>Japan</option>
          <option value="Global" {'selected' if region == 'Global' else ''}>Global</option>
        </select>
      </label>
      <label>予測ホライズン(月)<input name="horizon_months" type="number" step="1" value="{horizon_months}"></label>
      <label>確率の鋭さ<input name="temperature" type="number" step="0.05" value="{temperature:.2f}"></label>
      <label>シミュレーション本数<input name="n_sims" type="number" step="100" value="{n_sims}"></label>
      <label>乱数シード<input name="seed" type="number" step="1" value="{seed}"></label>
      <div class="actions"><button type="submit">予測を更新</button></div>
    </form>
    """

    metrics = _render_metrics(
        [
            ("現在レジーム", str(top_row["current_regime"])),
            ("次優位ファクター", str(top_row["factor"])),
            ("優位確率", _fmt_pct(float(top_row["lead_probability"]))),
            ("期待ポートフォリオ収益", _fmt_pct(float(summary["expected_portfolio_return"]))),
        ]
    )
    chart_one = _render_line_chart(
        "レジーム確率の推移",
        [
            ("Supportive", regime_tail["supportive_prob"], "#2f8f5b"),
            ("Transition", regime_tail["transition_prob"], "#d5951f"),
            ("Fragile", regime_tail["fragile_prob"], "#c44e40"),
        ],
        percent=True,
    )
    chart_two = _render_line_chart(
        "将来ポートフォリオ分布",
        [
            ("Median", simulation["fan_chart"]["p50"] * 100.0, "#6b3fd1"),
            ("Mean", simulation["fan_chart"]["mean"] * 100.0, "#0b8f66"),
        ],
        percent=False,
    )
    return (
        "<section class='panel'><div class='kicker'>Factor Forecast</div><h2>ファクターの優位候補を実行する</h2>"
        "<p class='copy'>ローリング指標、レジーム、周期シグナルを使って、どのファクターが次に強くなりやすいかを見ます。</p>"
        + form
        + f"<p class='note'>{escape(load_note)}</p>"
        + metrics
        + chart_one
        + chart_two
        + "<div class='two-col'><div class='table-card'><h3>ファクタースコア</h3>"
        + _frame_to_html(forecast_df[["rank", "factor", "lead_probability", "expected_horizon_return", "confidence"]])
        + "</div><div class='table-card'><h3>勝ち残り確率</h3>"
        + _frame_to_html(simulation["winner_probabilities"])
        + "</div></div></section>"
    )


def _render_policy(params: dict[str, list[str]]) -> str:
    initial_inflation_pct = _get_float(params, "initial_inflation_pct", 4.2, minimum=0.0, maximum=10.0)
    target_inflation_pct = _get_float(params, "target_inflation_pct", 2.0, minimum=0.0, maximum=5.0)
    initial_policy_rate_pct = _get_float(params, "initial_policy_rate_pct", 1.0, minimum=0.0, maximum=10.0)
    neutral_rate_pct = _get_float(params, "neutral_rate_pct", 1.0, minimum=-1.0, maximum=5.0)
    initial_output_gap_pct = _get_float(params, "initial_output_gap_pct", 0.4, minimum=-4.0, maximum=4.0)
    total_hike_bps = _get_int(params, "total_hike_bps", 175, minimum=0, maximum=500)
    hike_months = _get_int(params, "hike_months", 8, minimum=1, maximum=18)
    horizon_months = _get_int(params, "horizon_months", 24, minimum=12, maximum=60)
    n_sims = _get_int(params, "n_sims", 900, minimum=300, maximum=1500)
    expectations_persistence = _get_float(params, "expectations_persistence", 0.7, minimum=0.1, maximum=0.95)
    inflation_persistence = _get_float(params, "inflation_persistence", 0.62, minimum=0.1, maximum=0.95)
    rate_sensitivity = _get_float(params, "rate_sensitivity", 0.55, minimum=0.1, maximum=1.5)
    phillips_slope = _get_float(params, "phillips_slope", 0.35, minimum=0.05, maximum=1.0)
    supply_shock_pct = _get_float(params, "supply_shock_pct", 1.3, minimum=0.0, maximum=6.0)
    supply_decay = _get_float(params, "supply_decay", 0.82, minimum=0.4, maximum=0.98)
    credibility = _get_float(params, "credibility", 0.65, minimum=0.0, maximum=1.5)
    seed = _get_int(params, "seed", 42, minimum=1)

    result = simulate_policy_paths(
        horizon_months=horizon_months,
        n_sims=n_sims,
        initial_inflation=initial_inflation_pct / 100.0,
        target_inflation=target_inflation_pct / 100.0,
        initial_policy_rate=initial_policy_rate_pct / 100.0,
        neutral_rate=neutral_rate_pct / 100.0,
        initial_output_gap=initial_output_gap_pct / 100.0,
        total_hike_bps=total_hike_bps,
        hike_months=hike_months,
        expectations_persistence=expectations_persistence,
        inflation_persistence=inflation_persistence,
        rate_sensitivity=rate_sensitivity,
        phillips_slope=phillips_slope,
        supply_shock=supply_shock_pct / 100.0,
        supply_decay=supply_decay,
        credibility=credibility,
        seed=seed,
    )

    summary = result["summary"]
    quantiles = result["quantiles"]
    terminal = result["terminal"]

    form = f"""
    <form class="control-grid" method="get">
      <input type="hidden" name="tool" value="policy">
      <label>現在インフレ率 (%)<input name="initial_inflation_pct" type="number" step="0.1" value="{initial_inflation_pct:.1f}"></label>
      <label>目標インフレ率 (%)<input name="target_inflation_pct" type="number" step="0.1" value="{target_inflation_pct:.1f}"></label>
      <label>政策金利 (%)<input name="initial_policy_rate_pct" type="number" step="0.1" value="{initial_policy_rate_pct:.1f}"></label>
      <label>累積利上げ幅 (bps)<input name="total_hike_bps" type="number" step="25" value="{total_hike_bps}"></label>
      <label>利上げ月数<input name="hike_months" type="number" step="1" value="{hike_months}"></label>
      <label>期間(月)<input name="horizon_months" type="number" step="1" value="{horizon_months}"></label>
      <label>シミュレーション本数<input name="n_sims" type="number" step="100" value="{n_sims}"></label>
      <label>政策信認<input name="credibility" type="number" step="0.05" value="{credibility:.2f}"></label>
      <label>需要の金利感応度<input name="rate_sensitivity" type="number" step="0.05" value="{rate_sensitivity:.2f}"></label>
      <label>供給ショック (%)<input name="supply_shock_pct" type="number" step="0.1" value="{supply_shock_pct:.1f}"></label>
      <label>乱数シード<input name="seed" type="number" step="1" value="{seed}"></label>
      <div class="actions"><button type="submit">政策シナリオを更新</button></div>
    </form>
    """

    metrics = _render_metrics(
        [
            ("目標達成確率", _fmt_pct(float(summary["target_hit_probability"]))),
            ("ソフトランディング確率", _fmt_pct(float(summary["soft_landing_probability"]))),
            ("期末インフレ中央値", _fmt_pct(float(summary["median_terminal_inflation"]))),
            ("期末政策金利中央値", _fmt_pct(float(summary["median_terminal_policy_rate"]))),
        ]
    )
    chart_one = _render_line_chart(
        "インフレと政策金利の中央値",
        [
            ("Inflation median", quantiles["inflation_p50"] * 100.0, "#3056c9"),
            ("Policy rate median", quantiles["policy_rate_p50"] * 100.0, "#d27b2f"),
        ],
        percent=False,
    )
    chart_two = _render_line_chart(
        "GDPギャップの中央値",
        [("Output gap median", quantiles["output_gap_p50"] * 100.0, "#7d4cc9")],
        percent=False,
    )
    return (
        "<section class='panel'><div class='kicker'>Monetary Policy</div><h2>利上げシナリオを実行する</h2>"
        "<p class='copy'>政策金利の引き上げが、インフレと景気ギャップにどう効くかを確率的に眺めます。</p>"
        + form
        + metrics
        + chart_one
        + chart_two
        + "<div class='table-card'><h3>期末シナリオのサンプル</h3>"
        + _frame_to_html(terminal[["terminal_inflation", "terminal_output_gap", "terminal_policy_rate", "hit_target", "soft_landing"]])
        + "</div></section>"
    )


def _render_accounting(params: dict[str, list[str]]) -> str:
    company_name = _get_arg(params, "company_name", "New Project Holdings")
    base_currency = _get_arg(params, "base_currency", "JPY")
    opening_cash = _get_float(params, "opening_cash", 12000000.0, minimum=0.0)
    reserve_ratio_pct = _get_float(params, "reserve_ratio_pct", 20.0, minimum=0.0, maximum=100.0)
    use_demo = _get_bool(params, "use_demo", True)
    manual_table = _get_arg(
        params,
        "manual_table",
        "Date,Description,Amount,Category\n2026-01-05,Advisory revenue,850000,Sales\n2026-01-08,Office rent,-180000,Rent\n2026-01-15,Payroll transfer,-420000,Payroll",
    )

    source_rows: list[dict[str, object]] = []
    journal_parts: list[pd.DataFrame] = []

    if use_demo:
        demo_frame = demo_transactions()
        journal_parts.append(normalize_transactions(demo_frame, source_name="Demo", currency=base_currency))
        source_rows.append(
            {"file_name": "Demo seed", "file_type": "generated", "size_kb": 0.0, "rows_detected": len(demo_frame), "status": "sample_loaded", "ocr_preview": ""}
        )

    manual_df = parse_manual_table(manual_table)
    if not manual_df.empty:
        journal_parts.append(normalize_transactions(manual_df, source_name="Manual paste", currency=base_currency))
        source_rows.append(
            {
                "file_name": "Manual paste",
                "file_type": "text",
                "size_kb": round(len(manual_table.encode("utf-8")) / 1024.0, 1),
                "rows_detected": len(manual_df),
                "status": "parsed",
                "ocr_preview": "",
            }
        )

    journal_df = pd.concat(journal_parts, ignore_index=True) if journal_parts else pd.DataFrame()
    source_manifest_df = build_source_manifest(source_rows)
    budget_df = build_budget_template(journal_df)
    budget_vs_actual_df = build_budget_vs_actual(journal_df, budget_df)
    cash_settings_df = build_cash_settings(
        company_name=company_name,
        base_currency=base_currency,
        opening_cash=opening_cash,
        reserve_ratio=reserve_ratio_pct / 100.0,
        bank_accounts=["Main Operating Account", "Tax Reserve Account", "Payroll Account"],
    )
    summary = build_accounting_summary(journal_df)

    form = f"""
    <form class="control-grid accounting-form" method="get">
      <input type="hidden" name="tool" value="accounting">
      <label>会社名<input name="company_name" value="{escape(company_name)}"></label>
      <label>基軸通貨
        <select name="base_currency">
          <option value="JPY" {'selected' if base_currency == 'JPY' else ''}>JPY</option>
          <option value="USD" {'selected' if base_currency == 'USD' else ''}>USD</option>
          <option value="EUR" {'selected' if base_currency == 'EUR' else ''}>EUR</option>
        </select>
      </label>
      <label>期首現預金<input name="opening_cash" type="number" step="1000" value="{opening_cash:.0f}"></label>
      <label>流動性リザーブ比率 (%)<input name="reserve_ratio_pct" type="number" step="1" value="{reserve_ratio_pct:.0f}"></label>
      <label class="checkbox-row"><input name="use_demo" type="checkbox" value="1" {'checked' if use_demo else ''}> デモ取引を使う</label>
      <label class="full">手入力テーブル<textarea name="manual_table" rows="7">{escape(manual_table)}</textarea></label>
      <div class="actions"><button type="submit">会計整理を更新</button></div>
    </form>
    """

    metrics = _render_metrics(
        [
            ("仕訳候補数", str(int(summary["entries"]))),
            ("ネット現預金変動", f"{summary['net_cash']:,.0f} {base_currency}"),
            ("推定売上", f"{summary['revenue']:,.0f} {base_currency}"),
            ("推定支出", f"{summary['expenses']:,.0f} {base_currency}"),
        ]
    )
    return (
        "<section class='panel'><div class='kicker'>Accounting</div><h2>会計整理のコードを実行する</h2>"
        "<p class='copy'>Vercel 版では、デモデータと手入力テーブルをもとに仕訳候補と予算ひな型をすぐ確認できます。</p>"
        + form
        + metrics
        + "<div class='two-col'><div class='table-card'><h3>自動分類された仕訳候補</h3>"
        + _frame_to_html(journal_df[["tx_id", "booking_date", "description", "amount", "budget_bucket", "inferred_account"]])
        + "</div><div class='table-card'><h3>Budget vs Actual</h3>"
        + _frame_to_html(budget_vs_actual_df[["budget_bucket", "monthly_budget", "actual_monthly", "variance", "variance_pct"]])
        + "</div></div><div class='two-col'><div class='table-card'><h3>現預金設定</h3>"
        + _frame_to_html(cash_settings_df)
        + "</div><div class='table-card'><h3>入力元マニフェスト</h3>"
        + _frame_to_html(source_manifest_df)
        + "</div></div></section>"
    )


def _page_shell(active_tool: str, content: str) -> str:
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>bpp interactive lab</title>
  <style>
    :root {{
      --ink: #1e293b;
      --muted: #5b677a;
      --line: rgba(30, 41, 59, 0.12);
      --card: rgba(255, 255, 255, 0.9);
      --accent: #ef6b57;
      --accent-2: #1f4fd6;
      --bg-a: #edf6ff;
      --bg-b: #fff4ea;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Hiragino Sans", "Yu Gothic", "Meiryo", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.92), transparent 35%),
        linear-gradient(135deg, var(--bg-a) 0%, var(--bg-b) 100%);
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 56px; }}
    .hero, .panel, .metric-card, .table-card, .chart-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 18px 44px rgba(30, 41, 59, 0.08);
    }}
    .hero {{ padding: 26px 28px; margin-bottom: 16px; }}
    .hero h1 {{ margin: 0 0 10px 0; font-size: 42px; line-height: 1.05; }}
    .hero p {{ margin: 0; color: var(--muted); line-height: 1.7; max-width: 900px; }}
    .kicker {{ color: var(--accent); font-size: 12px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 10px; }}
    .nav-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 16px 0; }}
    .nav-link {{
      display: inline-flex; align-items: center; justify-content: center; padding: 10px 14px;
      border-radius: 999px; text-decoration: none; color: var(--ink); background: rgba(255,255,255,0.7);
      border: 1px solid var(--line); font-weight: 600;
    }}
    .nav-link.active {{ background: var(--accent-2); color: white; border-color: transparent; }}
    .panel {{ padding: 20px; }}
    .panel h2 {{ margin: 0 0 8px 0; font-size: 28px; }}
    .copy, .note {{ color: var(--muted); line-height: 1.7; }}
    .note {{ margin-top: 8px; font-size: 14px; }}
    .control-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 18px 0; }}
    .control-grid label {{ display: flex; flex-direction: column; gap: 6px; font-size: 14px; color: var(--muted); }}
    .control-grid label.full {{ grid-column: 1 / -1; }}
    input, select, textarea, button {{
      font: inherit;
    }}
    input, select, textarea {{
      width: 100%;
      border: 1px solid rgba(30,41,59,0.15);
      border-radius: 14px;
      padding: 11px 12px;
      background: white;
      color: var(--ink);
    }}
    textarea {{ resize: vertical; min-height: 120px; }}
    .checkbox-row {{ flex-direction: row !important; align-items: center; gap: 10px; margin-top: 32px; }}
    .checkbox-row input {{ width: auto; }}
    .actions {{ display: flex; align-items: end; }}
    button {{
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 10px 22px rgba(239, 107, 87, 0.28);
    }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric-card {{ padding: 16px; }}
    .metric-label {{ font-size: 13px; color: var(--muted); margin-bottom: 8px; }}
    .metric-value {{ font-size: 26px; font-weight: 700; line-height: 1.1; }}
    .two-col {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; margin-top: 14px; }}
    .table-card, .chart-card {{ padding: 16px; margin-top: 14px; }}
    .table-card h3, .chart-card h3 {{ margin: 0 0 12px 0; font-size: 20px; }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .data-table th, .data-table td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid rgba(30,41,59,0.08); }}
    .data-table th {{ background: rgba(31,79,214,0.06); }}
    .chart-svg {{ width: 100%; height: auto; display: block; }}
    .chart-axis {{ display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; margin-top: 8px; }}
    .chart-legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; color: var(--muted); font-size: 13px; }}
    .chart-legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .chart-legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 999px; }}
    .footer {{
      margin-top: 18px; color: var(--muted); font-size: 14px; line-height: 1.7;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="kicker">bpp interactive lab</div>
      <h1>Vercel でも実行できる金融研究 UI</h1>
      <p>この画面は、`shared_finance/` にある分析エンジンを Vercel 上でそのまま動かすための軽量入口です。Streamlit 版ほど多機能ではありませんが、主要ロジックはここでも触れます。</p>
    </section>
    {_nav(active_tool)}
    {content}
    <div class="footer">
      ローカルの Streamlit 版を使う場合は <code>pip install -r requirements-local.txt</code> のあと <code>streamlit run app.py</code> で起動できます。
    </div>
  </div>
</body>
</html>"""


def _route_content(tool: str, params: dict[str, list[str]]) -> str:
    mapping: dict[str, Callable[[dict[str, list[str]]], str]] = {
        "behavioral": _render_behavioral,
        "factor": _render_factor,
        "policy": _render_policy,
        "accounting": _render_accounting,
    }
    renderer = mapping.get(tool, _render_behavioral)
    return renderer(params)


def app(environ, start_response):
    params = parse_qs(environ.get("QUERY_STRING", ""), keep_blank_values=True)
    path = environ.get("PATH_INFO", "/")

    if path == "/health":
        body = b"ok"
        start_response("200 OK", [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))])
        return [body]

    tool = _get_arg(params, "tool", "behavioral")
    html = _page_shell(tool, _route_content(tool, params))
    body = html.encode("utf-8")
    headers = [
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    start_response("200 OK", headers)
    return [body]
