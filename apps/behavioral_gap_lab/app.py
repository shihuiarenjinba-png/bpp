from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from shared_finance.behavioral_engine import simulate_behavioral_gap
from shared_finance.ui import apply_theme, render_hero, render_section_header


def load_calibration(uploaded_file) -> tuple[float | None, float | None]:
    if uploaded_file is None:
        return None, None
    suffix = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()
    if suffix == ".csv":
        frame = pd.read_csv(io.BytesIO(file_bytes))
    else:
        frame = pd.read_excel(io.BytesIO(file_bytes))
    normalized = {str(col).strip().lower().replace(" ", "_"): col for col in frame.columns}
    return_col = normalized.get("return") or normalized.get("monthly_return") or normalized.get("ret")
    price_col = normalized.get("price") or normalized.get("close")
    if return_col:
        returns = pd.to_numeric(frame[return_col], errors="coerce").dropna()
    elif price_col:
        prices = pd.to_numeric(frame[price_col], errors="coerce").dropna()
        returns = prices.pct_change().dropna()
    else:
        return None, None

    if returns.abs().max() > 1.5:
        returns = returns / 100.0
    mu = float(returns.mean() * 12.0)
    sigma = float(returns.std(ddof=0) * (12.0**0.5))
    return mu, sigma


apply_theme(
    page_title="Behavioral Gap Lab",
    page_icon="🧠",
    accent="#0e7c86",
    gradient_start="#ebfbff",
    gradient_end="#fff8ef",
)

render_hero(
    "Behavioral Gap Lab",
    "伝統的ファイナンスの価格形成と、過剰反応・損失回避・群集行動を含む行動ファイナンスの価格形成を並列に比較します。",
    kicker="Traditional vs Behavioral Finance",
    tags=["Monte Carlo", "Overreaction", "Loss Aversion", "Herding"],
)

with st.sidebar:
    st.subheader("マーケット設定")
    calibration_file = st.file_uploader("任意のリターン系列でキャリブレーション", type=["csv", "xlsx", "xls"])
    calibrated_mu, calibrated_sigma = load_calibration(calibration_file)
    st.caption("ファイルがある場合は、期待リターンとボラティリティの初期値に反映されます。")
    expected_return = st.slider(
        "期待年率リターン",
        min_value=-5.0,
        max_value=20.0,
        value=float(round((calibrated_mu if calibrated_mu is not None else 7.5), 2)),
        step=0.25,
    ) / 100.0
    volatility = st.slider(
        "年率ボラティリティ",
        min_value=5.0,
        max_value=45.0,
        value=float(round((calibrated_sigma if calibrated_sigma is not None else 16.0), 2)),
        step=0.5,
    ) / 100.0
    initial_price = st.number_input("初期価格", min_value=10.0, value=100.0, step=10.0)
    years = st.slider("シミュレーション年数", min_value=2, max_value=15, value=7, step=1)
    n_sims = st.slider("シミュレーション本数", min_value=200, max_value=4000, value=1200, step=100)

    st.subheader("行動バイアス")
    overreaction = st.slider("過剰反応", min_value=0.0, max_value=2.5, value=1.2, step=0.1)
    loss_aversion = st.slider("損失回避", min_value=0.0, max_value=2.5, value=1.0, step=0.1)
    herding = st.slider("群集行動", min_value=0.0, max_value=2.5, value=0.8, step=0.1)
    anchoring = st.slider("アンカリング", min_value=0.0, max_value=2.5, value=0.6, step=0.1)
    sentiment_beta = st.slider("センチメント感応度", min_value=0.0, max_value=2.5, value=0.9, step=0.1)
    seed = st.number_input("乱数シード", min_value=1, value=42, step=1)

result = simulate_behavioral_gap(
    initial_price=float(initial_price),
    expected_return=float(expected_return),
    volatility=float(volatility),
    years=years,
    n_sims=n_sims,
    overreaction=float(overreaction),
    loss_aversion=float(loss_aversion),
    herding=float(herding),
    anchoring=float(anchoring),
    sentiment_beta=float(sentiment_beta),
    seed=int(seed),
)

comparison = result["comparison"]
fan_chart = result["fan_chart"]
divergence = result["divergence"]
diagnostics = result["diagnostics"]

tabs = st.tabs(["Scenario Paths", "Gap Diagnostics", "Summary Table"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("行動ファイナンス期待年率", f"{comparison.loc[1, 'expected_return'] * 100:.2f}%")
    c2.metric("行動モデル平均最大DD", f"{comparison.loc[1, 'avg_max_drawdown'] * 100:.2f}%")
    c3.metric("終値ギャップ > 10%", f"{diagnostics['gap_probability_gt_10pct'] * 100:.1f}%")
    c4.metric("終値ギャップ < -10%", f"{diagnostics['gap_probability_lt_minus_10pct'] * 100:.1f}%")

    render_section_header(
        "Path View",
        "平均パスと不確実性帯",
        "伝統モデルと行動モデルの平均価格推移、10-90%レンジを重ねて見られます。",
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fan_chart["date"], y=fan_chart["rational_p90"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fan_chart["date"], y=fan_chart["rational_p10"], mode="lines", fill="tonexty", fillcolor="rgba(56, 168, 189, 0.12)", line=dict(width=0), name="Traditional 10-90%"))
    fig.add_trace(go.Scatter(x=fan_chart["date"], y=fan_chart["behavioral_p90"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fan_chart["date"], y=fan_chart["behavioral_p10"], mode="lines", fill="tonexty", fillcolor="rgba(232, 119, 34, 0.14)", line=dict(width=0), name="Behavioral 10-90%"))
    fig.add_trace(go.Scatter(x=fan_chart["date"], y=fan_chart["rational_mean"], mode="lines", name="Traditional mean", line=dict(color="#14808a", width=3)))
    fig.add_trace(go.Scatter(x=fan_chart["date"], y=fan_chart["behavioral_mean"], mode="lines", name="Behavioral mean", line=dict(color="#d56c1b", width=3)))
    fig.update_layout(margin=dict(t=30, l=20, r=20, b=20), yaxis_title="Price", xaxis_title="Date")
    st.plotly_chart(fig, width="stretch")

with tabs[1]:
    render_section_header(
        "Gap",
        "乖離の向きと強さ",
        "伝統モデル比の価格乖離と、各月の平均バイアス項を確認できます。",
    )
    gap_fig = go.Figure()
    gap_fig.add_trace(
        go.Scatter(
            x=divergence["date"],
            y=divergence["relative_gap"] * 100.0,
            mode="lines",
            name="Relative gap (%)",
            line=dict(color="#0e7c86", width=3),
        )
    )
    gap_fig.add_trace(
        go.Bar(
            x=divergence["date"],
            y=divergence["avg_bias_term"] * 100.0,
            name="Average bias term (bp)",
            marker_color="rgba(213, 108, 27, 0.42)",
            yaxis="y2",
        )
    )
    gap_fig.update_layout(
        margin=dict(t=30, l=20, r=20, b=20),
        yaxis_title="Relative gap (%)",
        yaxis2=dict(title="Bias term (%)", overlaying="y", side="right"),
        barmode="overlay",
    )
    st.plotly_chart(gap_fig, width="stretch")
    st.dataframe(pd.DataFrame([diagnostics]), width="stretch", hide_index=True)

with tabs[2]:
    render_section_header(
        "Comparison",
        "モデル別の比較表",
        "リターン、ボラティリティ、クラッシュ確率、シャープレシオを横並びにしています。",
    )
    display = comparison.copy()
    for col in ["expected_return", "volatility", "median_terminal", "crash_probability", "avg_max_drawdown", "gap_vs_traditional"]:
        if col in display.columns:
            display[col] = display[col] * 100.0
    st.dataframe(display, width="stretch", hide_index=True)
