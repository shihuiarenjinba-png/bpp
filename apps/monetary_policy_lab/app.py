from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from shared_finance.policy_engine import simulate_policy_paths
from shared_finance.ui import apply_theme, render_hero, render_section_header


apply_theme(
    page_title="Monetary Policy Lab",
    page_icon="🏦",
    accent="#3056c9",
    gradient_start="#edf3ff",
    gradient_end="#fff6ec",
)

render_hero(
    "Monetary Policy Lab",
    "金利引き上げによるインフレ抑制の効き方を、期待インフレ、供給ショック、景気ギャップ込みで確率的に眺めるための政策シミュレーターです。",
    kicker="Rate Hike vs Inflation",
    tags=["Inflation", "Policy Rate", "Soft Landing", "Monte Carlo"],
)

with st.sidebar:
    st.subheader("初期条件")
    initial_inflation = st.slider("現在のインフレ率", min_value=0.0, max_value=10.0, value=4.2, step=0.1) / 100.0
    target_inflation = st.slider("目標インフレ率", min_value=0.0, max_value=5.0, value=2.0, step=0.1) / 100.0
    initial_policy_rate = st.slider("政策金利", min_value=0.0, max_value=10.0, value=1.0, step=0.1) / 100.0
    neutral_rate = st.slider("中立実質金利の近似", min_value=-1.0, max_value=5.0, value=1.0, step=0.1) / 100.0
    initial_output_gap = st.slider("初期GDPギャップ", min_value=-4.0, max_value=4.0, value=0.4, step=0.1) / 100.0

    st.subheader("政策シナリオ")
    total_hike_bps = st.slider("累積利上げ幅", min_value=0, max_value=500, value=175, step=25)
    hike_months = st.slider("利上げを終えるまでの月数", min_value=1, max_value=18, value=8, step=1)
    horizon_months = st.slider("シミュレーション期間", min_value=12, max_value=60, value=24, step=6)
    n_sims = st.slider("シミュレーション本数", min_value=300, max_value=3000, value=1200, step=100)

    st.subheader("マクロ感応度")
    expectations_persistence = st.slider("期待インフレの粘着性", min_value=0.1, max_value=0.95, value=0.7, step=0.05)
    inflation_persistence = st.slider("実際のインフレ粘着性", min_value=0.1, max_value=0.95, value=0.62, step=0.05)
    rate_sensitivity = st.slider("需要の金利感応度", min_value=0.1, max_value=1.5, value=0.55, step=0.05)
    phillips_slope = st.slider("フィリップス曲線の傾き", min_value=0.05, max_value=1.0, value=0.35, step=0.05)
    supply_shock = st.slider("初期供給ショック", min_value=0.0, max_value=6.0, value=1.3, step=0.1) / 100.0
    supply_decay = st.slider("供給ショック減衰", min_value=0.4, max_value=0.98, value=0.82, step=0.02)
    credibility = st.slider("政策信認の強さ", min_value=0.0, max_value=1.5, value=0.65, step=0.05)
    seed = st.number_input("乱数シード", min_value=1, value=42, step=1)

result = simulate_policy_paths(
    horizon_months=horizon_months,
    n_sims=n_sims,
    initial_inflation=initial_inflation,
    target_inflation=target_inflation,
    initial_policy_rate=initial_policy_rate,
    neutral_rate=neutral_rate,
    initial_output_gap=initial_output_gap,
    total_hike_bps=total_hike_bps,
    hike_months=hike_months,
    expectations_persistence=expectations_persistence,
    inflation_persistence=inflation_persistence,
    rate_sensitivity=rate_sensitivity,
    phillips_slope=phillips_slope,
    supply_shock=supply_shock,
    supply_decay=supply_decay,
    credibility=credibility,
    seed=int(seed),
)

summary = result["summary"]
quantiles = result["quantiles"]
terminal = result["terminal"]

tabs = st.tabs(["Inflation & Rate", "Landing Distribution", "Terminal Table"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("目標達成確率", f"{summary['target_hit_probability'] * 100:.1f}%")
    c2.metric("ソフトランディング確率", f"{summary['soft_landing_probability'] * 100:.1f}%")
    c3.metric("期末インフレ中央値", f"{summary['median_terminal_inflation'] * 100:.2f}%")
    c4.metric("期末政策金利中央値", f"{summary['median_terminal_policy_rate'] * 100:.2f}%")

    render_section_header(
        "Path",
        "インフレと政策金利のファンチャート",
        "政策金利が先行し、その後にインフレ中央値とレンジがどう動くかを一目で確認できます。",
    )

    inflation_fig = go.Figure()
    inflation_fig.add_trace(go.Scatter(x=quantiles["month"], y=quantiles["inflation_p90"] * 100, mode="lines", line=dict(width=0), showlegend=False))
    inflation_fig.add_trace(go.Scatter(x=quantiles["month"], y=quantiles["inflation_p10"] * 100, mode="lines", fill="tonexty", fillcolor="rgba(48, 86, 201, 0.12)", line=dict(width=0), name="Inflation 10-90%"))
    inflation_fig.add_trace(go.Scatter(x=quantiles["month"], y=quantiles["inflation_p50"] * 100, mode="lines", name="Median inflation", line=dict(color="#3056c9", width=3)))
    inflation_fig.add_trace(go.Scatter(x=quantiles["month"], y=quantiles["policy_rate_p50"] * 100, mode="lines", name="Median policy rate", line=dict(color="#d27b2f", width=3)))
    inflation_fig.update_layout(margin=dict(t=30, l=20, r=20, b=20), xaxis_title="Month", yaxis_title="Percent")
    st.plotly_chart(inflation_fig, width="stretch")

    output_fig = go.Figure()
    output_fig.add_trace(go.Scatter(x=quantiles["month"], y=quantiles["output_gap_p90"] * 100, mode="lines", line=dict(width=0), showlegend=False))
    output_fig.add_trace(go.Scatter(x=quantiles["month"], y=quantiles["output_gap_p10"] * 100, mode="lines", fill="tonexty", fillcolor="rgba(210, 123, 47, 0.12)", line=dict(width=0), name="Output gap 10-90%"))
    output_fig.add_trace(go.Scatter(x=quantiles["month"], y=quantiles["output_gap_p50"] * 100, mode="lines", name="Median output gap", line=dict(color="#d27b2f", width=3)))
    output_fig.update_layout(margin=dict(t=30, l=20, r=20, b=20), xaxis_title="Month", yaxis_title="Percent")
    st.plotly_chart(output_fig, width="stretch")

with tabs[1]:
    render_section_header(
        "Trade-off",
        "インフレ抑制と景気コストの分布",
        "期末時点のインフレ率とGDPギャップの散布で、成功シナリオと痛みの大きいシナリオを切り分けます。",
    )
    scatter = go.Figure()
    scatter.add_trace(
        go.Scattergl(
            x=terminal["terminal_output_gap"] * 100,
            y=terminal["terminal_inflation"] * 100,
            mode="markers",
            marker=dict(
                size=7,
                color=terminal["soft_landing"].map({True: "#3056c9", False: "#d27b2f"}),
                opacity=0.45,
            ),
            text=terminal["terminal_policy_rate"] * 100,
            hovertemplate="Output gap: %{x:.2f}%<br>Inflation: %{y:.2f}%<br>Policy rate: %{text:.2f}%<extra></extra>",
        )
    )
    scatter.update_layout(margin=dict(t=30, l=20, r=20, b=20), xaxis_title="Terminal output gap (%)", yaxis_title="Terminal inflation (%)")
    st.plotly_chart(scatter, width="stretch")

with tabs[2]:
    render_section_header(
        "Terminal",
        "最終月のシミュレーション結果",
        "期末インフレ率、GDPギャップ、政策金利、目標達成フラグを一覧できます。",
    )
    display = terminal.copy()
    for col in ["terminal_inflation", "terminal_output_gap", "terminal_policy_rate"]:
        display[col] = display[col] * 100.0
    st.dataframe(display, width="stretch", hide_index=True)
