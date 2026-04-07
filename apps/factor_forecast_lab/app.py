from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from shared_finance.factor_engine import (
    compute_factor_forecast,
    generate_sample_factor_dataset,
    load_uploaded_factor_table,
    prepare_factor_frame,
    simulate_factor_outlook,
)
from shared_finance.ui import apply_theme, render_hero, render_section_header


def load_live_factor_data(region: str):
    try:
        from shared_finance.factor_data_loader import load_factor_dataset

        live = load_factor_dataset(region=region, start_date="2005-01-01", end_date="2026-03-01")
        live = live.reset_index().rename(columns={live.index.name or "index": "Date"})
        return live, None
    except Exception as exc:
        return None, exc


apply_theme(
    page_title="Factor Forecast Lab",
    page_icon="🧬",
    accent="#6b3fd1",
    gradient_start="#f2edff",
    gradient_end="#eefaf4",
)

render_hero(
    "Factor Forecast Lab",
    "ローリングファクター、レジーム推定、フーリエ由来の周期シグナルを組み合わせて、次に優位となるファクターを確率付きで予想します。",
    kicker="Probabilistic Factor Forecast",
    tags=["Rolling Factors", "Regime", "Fourier Signal", "Lead Probability"],
)

with st.sidebar:
    st.subheader("データソース")
    source_mode = st.radio("読み込み方法", ["Demo sample", "Upload", "Live Ken French"], index=0)
    uploaded_factor_file = None
    live_region = "Japan"
    if source_mode == "Upload":
        uploaded_factor_file = st.file_uploader("ファクターデータ", type=["csv", "xlsx", "xls"])
    if source_mode == "Live Ken French":
        live_region = st.selectbox("地域", ["US", "Japan", "Global"], index=1)

    st.subheader("予測設定")
    horizon_months = st.slider("予測ホライズン", min_value=1, max_value=12, value=6, step=1)
    temperature = st.slider("確率の鋭さ", min_value=0.2, max_value=1.2, value=0.65, step=0.05)
    n_sims = st.slider("将来パス本数", min_value=500, max_value=5000, value=2000, step=250)
    seed = st.number_input("乱数シード", min_value=1, value=42, step=1)

source_label = source_mode
raw_factor_data = None
load_error = None

if source_mode == "Upload" and uploaded_factor_file is not None:
    raw_factor_data = load_uploaded_factor_table(uploaded_factor_file.name, uploaded_factor_file.getvalue())
    source_label = uploaded_factor_file.name
elif source_mode == "Live Ken French":
    raw_factor_data, load_error = load_live_factor_data(live_region)
    source_label = f"Ken French {live_region}"
else:
    raw_factor_data = generate_sample_factor_dataset()
    source_label = "Synthetic sample"

factor_df = prepare_factor_frame(raw_factor_data) if raw_factor_data is not None else generate_sample_factor_dataset()

if factor_df.empty:
    st.error("有効なファクターデータを解釈できませんでした。列名は `Date, Mkt-RF, SMB, HML, RMW, CMA, MOM` のいずれかに合わせてください。")
    st.stop()

forecast_df, regime_df = compute_factor_forecast(
    factor_df,
    horizon_months=horizon_months,
    temperature=float(temperature),
)

if forecast_df.empty:
    st.error("予測に必要な期間が不足しています。最低でも月次データ18期間以上をおすすめします。")
    st.stop()

simulation = simulate_factor_outlook(
    factor_df,
    forecast_df,
    horizon_months=horizon_months,
    n_sims=n_sims,
    seed=int(seed),
)

tabs = st.tabs(["Overview", "Scoreboard", "Monte Carlo", "Data"])

with tabs[0]:
    if load_error is not None:
        st.warning(f"ライブ読み込みに失敗したため、別ソースの利用をご検討ください: {load_error}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("データソース", source_label)
    c2.metric("現在レジーム", str(forecast_df.iloc[0]['current_regime']))
    c3.metric("次優位ファクター", str(forecast_df.iloc[0]["factor"]))
    c4.metric("優位確率", f"{forecast_df.iloc[0]['lead_probability'] * 100:.1f}%")

    c5, c6, c7 = st.columns(3)
    c5.metric("期待ポートフォリオ収益", f"{simulation['summary']['expected_portfolio_return'] * 100:.2f}%")
    c6.metric("中央値", f"{simulation['summary']['median_portfolio_return'] * 100:.2f}%")
    c7.metric("損失確率", f"{simulation['summary']['loss_probability'] * 100:.1f}%")

    render_section_header(
        "Regime",
        "レジーム推定の推移",
        "市場ファクターの傾き、ボラティリティ、ドローダウン、ファクターブレッドスを使って状態を推定しています。",
    )

    regime_plot = regime_df.reset_index().rename(columns={"index": "date"})
    regime_fig = go.Figure()
    regime_fig.add_trace(go.Scatter(x=regime_plot["date"], y=regime_plot["supportive_prob"], mode="lines", name="Supportive", line=dict(color="#2f8f5b", width=3)))
    regime_fig.add_trace(go.Scatter(x=regime_plot["date"], y=regime_plot["transition_prob"], mode="lines", name="Transition", line=dict(color="#d5951f", width=3)))
    regime_fig.add_trace(go.Scatter(x=regime_plot["date"], y=regime_plot["fragile_prob"], mode="lines", name="Fragile", line=dict(color="#c44e40", width=3)))
    regime_fig.update_layout(margin=dict(t=30, l=20, r=20, b=20), yaxis_title="Probability", xaxis_title="Date")
    st.plotly_chart(regime_fig, width="stretch")

with tabs[1]:
    render_section_header(
        "Factors",
        "ファクター別スコアボード",
        "ローリングIR、レジーム条件付き平均、周期シグナル、勝ち残り確率をまとめています。",
    )
    score_chart = px.bar(
        forecast_df,
        x="factor",
        y="lead_probability",
        color="expected_horizon_return",
        text=forecast_df["lead_probability"].map(lambda value: f"{value * 100:.1f}%"),
        color_continuous_scale="Tealrose",
        labels={"lead_probability": "Lead probability", "expected_horizon_return": f"Expected {horizon_months}M return"},
    )
    score_chart.update_layout(margin=dict(t=30, l=20, r=20, b=20))
    st.plotly_chart(score_chart, width="stretch")
    st.dataframe(forecast_df, width="stretch", hide_index=True)

with tabs[2]:
    render_section_header(
        "Simulation",
        "優位ファクターを重み付けした将来分布",
        "予測確率をウェイトとして、将来ホライズンのポートフォリオ収益分布をシミュレートしています。",
    )
    fan_chart = simulation["fan_chart"]
    fan_fig = go.Figure()
    fan_fig.add_trace(go.Scatter(x=fan_chart["month"], y=fan_chart["p90"] * 100, mode="lines", line=dict(width=0), showlegend=False))
    fan_fig.add_trace(go.Scatter(x=fan_chart["month"], y=fan_chart["p10"] * 100, mode="lines", fill="tonexty", fillcolor="rgba(107, 63, 209, 0.15)", line=dict(width=0), name="10-90%"))
    fan_fig.add_trace(go.Scatter(x=fan_chart["month"], y=fan_chart["p50"] * 100, mode="lines", name="Median", line=dict(color="#6b3fd1", width=3)))
    fan_fig.add_trace(go.Scatter(x=fan_chart["month"], y=fan_chart["mean"] * 100, mode="lines", name="Mean", line=dict(color="#0b8f66", width=3)))
    fan_fig.update_layout(margin=dict(t=30, l=20, r=20, b=20), xaxis_title="Month", yaxis_title="Portfolio return (%)")
    st.plotly_chart(fan_fig, width="stretch")

    dominance_chart = px.bar(
        simulation["winner_probabilities"],
        x="factor",
        y="dominance_probability",
        color="dominance_probability",
        color_continuous_scale="Magma",
        labels={"dominance_probability": "Dominance probability"},
    )
    dominance_chart.update_layout(margin=dict(t=30, l=20, r=20, b=20))
    st.plotly_chart(dominance_chart, width="stretch")

with tabs[3]:
    render_section_header(
        "Dataset",
        "入力データ確認",
        "アップロードまたは生成した月次ファクターデータを確認できます。",
    )
    st.dataframe(factor_df.reset_index(), width="stretch", hide_index=True)
