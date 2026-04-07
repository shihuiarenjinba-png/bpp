from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="Finance App Collection", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1220px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }
    .hero {
        background: linear-gradient(135deg, #eef6ff 0%, #fff5ea 100%);
        border: 1px solid rgba(24, 34, 48, 0.08);
        border-radius: 24px;
        padding: 1.3rem 1.4rem;
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2.2rem;
        line-height: 1.1;
    }
    .hero p {
        margin: 0;
        color: #546173;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Finance App Collection</h1>
        <p>会計、行動ファイナンス、ファクター予測、金融政策シミュレーションを、1つの入口から切り替えて使えるようにした研究用コレクションです。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("左のナビゲーションから見たいテーマを選んでください。`auto_agent` はこのリポジトリでは起動対象に含めていません。")

navigation = st.navigation(
    [
        st.Page("apps/accounting_workbench/app.py", title="Accounting Workbench", icon=":material/receipt_long:"),
        st.Page("apps/behavioral_gap_lab/app.py", title="Behavioral Gap Lab", icon=":material/psychology:"),
        st.Page("apps/factor_forecast_lab/app.py", title="Factor Forecast Lab", icon=":material/query_stats:"),
        st.Page("apps/monetary_policy_lab/app.py", title="Monetary Policy Lab", icon=":material/account_balance:"),
    ]
)
navigation.run()
