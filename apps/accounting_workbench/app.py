from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from shared_finance.accounting_engine import (
    build_accounting_summary,
    build_accounting_workbook,
    build_budget_template,
    build_budget_vs_actual,
    build_cash_settings,
    build_mapping_catalog,
    build_source_manifest,
    demo_transactions,
    maybe_ocr_image,
    normalize_transactions,
    parse_manual_table,
    parse_tabular_upload,
)
from shared_finance.ui import apply_theme, render_hero, render_section_header


apply_theme(
    page_title="Accounting Workbench",
    page_icon="🧾",
    accent="#b54d24",
    gradient_start="#fff2e7",
    gradient_end="#eef4ff",
)

render_hero(
    "Accounting Workbench",
    "CSV、Excel、手入力、画像メモを受け取り、仕訳候補・予算・現預金設定をまとめた整理済みExcelへ落とし込みます。",
    kicker="Accounting Automation",
    tags=["CSV / XLSX", "Image Notes", "Budget", "Cash Settings", "Excel Export"],
)

with st.sidebar:
    st.subheader("基本設定")
    company_name = st.text_input("会社名", value="New Project Holdings")
    base_currency = st.selectbox("基軸通貨", ["JPY", "USD", "EUR"], index=0)
    opening_cash = st.number_input("期首現預金", min_value=0.0, value=12000000.0, step=100000.0)
    reserve_ratio = st.slider("流動性リザーブ比率", min_value=0.0, max_value=100.0, value=20.0, step=1.0) / 100.0
    bank_accounts_text = st.text_area(
        "現預金口座",
        value="Main Operating Account\nTax Reserve Account\nPayroll Account",
        height=110,
    )
    enable_ocr = st.toggle("画像OCRを試す", value=False)
    use_demo_data = st.toggle("デモ取引を読み込む", value=True)
    workbook_name = st.text_input("出力ファイル名", value="accounting_workbench.xlsx")

bank_accounts = [line.strip() for line in bank_accounts_text.splitlines() if line.strip()]

render_section_header(
    "Inputs",
    "取引データの取り込み",
    "複数ファイルをまとめて読み込み、できるだけ自動で科目とキャッシュフローを整理します。",
)

uploaded_files = st.file_uploader(
    "取引ファイルまたは証憑画像を追加",
    type=["csv", "xlsx", "xls", "xlsm", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)
manual_table_text = st.text_area(
    "そのまま貼り付ける表データ",
    value="Date,Description,Amount,Category\n2026-01-05,Advisory revenue,850000,Sales\n2026-01-08,Office rent,-180000,Rent\n2026-01-15,Payroll transfer,-420000,Payroll",
    height=110,
)

source_manifest_rows: list[dict[str, object]] = []
journal_parts: list[pd.DataFrame] = []
raw_preview_frames: list[tuple[str, pd.DataFrame]] = []

if use_demo_data:
    demo_frame = demo_transactions()
    raw_preview_frames.append(("Demo transactions", demo_frame.head(20)))
    journal_parts.append(
        normalize_transactions(demo_frame, source_name="Demo", currency=base_currency)
    )
    source_manifest_rows.append(
        {
            "file_name": "Demo seed",
            "file_type": "generated",
            "size_kb": 0.0,
            "rows_detected": len(demo_frame),
            "status": "sample_loaded",
            "ocr_preview": "",
        }
    )

for uploaded_file in uploaded_files or []:
    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    suffix = Path(file_name).suffix.lower()
    parsed_tables = parse_tabular_upload(file_name, file_bytes)

    if parsed_tables:
        row_count = 0
        for sheet_name, raw_df in parsed_tables:
            if raw_df is None or raw_df.empty:
                continue
            row_count += len(raw_df)
            raw_preview_frames.append((f"{file_name} / {sheet_name}", raw_df.head(20)))
            normalized = normalize_transactions(
                raw_df,
                source_name=f"{file_name}:{sheet_name}",
                currency=base_currency,
            )
            if not normalized.empty:
                journal_parts.append(normalized)

        source_manifest_rows.append(
            {
                "file_name": file_name,
                "file_type": suffix.replace(".", "") or "tabular",
                "size_kb": round(len(file_bytes) / 1024.0, 1),
                "rows_detected": row_count,
                "status": "imported",
                "ocr_preview": "",
            }
        )
    else:
        ocr_text = maybe_ocr_image(file_bytes) if enable_ocr else ""
        source_manifest_rows.append(
            {
                "file_name": file_name,
                "file_type": suffix.replace(".", "") or "binary",
                "size_kb": round(len(file_bytes) / 1024.0, 1),
                "rows_detected": 0,
                "status": "image_logged",
                "ocr_preview": ocr_text[:120],
            }
        )

manual_df = parse_manual_table(manual_table_text)
if not manual_df.empty:
    raw_preview_frames.append(("Manual paste", manual_df.head(20)))
    journal_parts.append(
        normalize_transactions(manual_df, source_name="Manual paste", currency=base_currency)
    )
    source_manifest_rows.append(
        {
            "file_name": "Manual paste",
            "file_type": "text",
            "size_kb": round(len(manual_table_text.encode("utf-8")) / 1024.0, 1),
            "rows_detected": len(manual_df),
            "status": "parsed",
            "ocr_preview": "",
        }
    )

journal_df = pd.concat(journal_parts, ignore_index=True) if journal_parts else pd.DataFrame()
source_manifest_df = build_source_manifest(source_manifest_rows)
cash_settings_df = build_cash_settings(
    company_name=company_name,
    base_currency=base_currency,
    opening_cash=opening_cash,
    reserve_ratio=reserve_ratio,
    bank_accounts=bank_accounts,
)
summary = build_accounting_summary(journal_df)

tabs = st.tabs(["Overview", "Raw Inputs", "Journal", "Budget & Cash", "Export"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("仕訳候補数", f"{int(summary['entries'])}")
    c2.metric("ネット現預金変動", f"{summary['net_cash']:,.0f} {base_currency}")
    c3.metric("推定売上", f"{summary['revenue']:,.0f} {base_currency}")
    c4.metric("推定支出", f"{summary['expenses']:,.0f} {base_currency}")

    if not journal_df.empty:
        bucket_view = (
            journal_df.groupby("budget_bucket", as_index=False)["amount"]
            .sum()
            .assign(display_amount=lambda frame: frame["amount"].abs())
            .sort_values("display_amount", ascending=False)
        )
        fig = px.bar(
            bucket_view,
            x="budget_bucket",
            y="display_amount",
            color="budget_bucket",
            title="主要バケット別の取引規模",
            labels={"budget_bucket": "Budget bucket", "display_amount": f"Amount ({base_currency})"},
        )
        fig.update_layout(showlegend=False, margin=dict(t=60, l=20, r=20, b=20))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("表データまたはデモデータを読み込むと、ここに集計が表示されます。")

    st.dataframe(source_manifest_df, width="stretch", hide_index=True)

with tabs[1]:
    render_section_header(
        "Trace",
        "取り込み元プレビュー",
        "自動整形の前に、どのような表を拾えているかを確認できます。",
    )
    if raw_preview_frames:
        for title, preview_df in raw_preview_frames[:6]:
            st.caption(title)
            st.dataframe(preview_df, width="stretch")
    else:
        st.info("まだプレビューできる入力がありません。")

with tabs[2]:
    render_section_header(
        "Journal",
        "自動分類された仕訳候補",
        "キーワード規則に基づいて勘定科目、予算バケット、IFRSヒントを付与しています。",
    )
    st.dataframe(journal_df, width="stretch", hide_index=True)
    st.caption("分類辞書")
    st.dataframe(build_mapping_catalog(), width="stretch", hide_index=True)

with tabs[3]:
    render_section_header(
        "Budgeting",
        "予算編成と現預金設計",
        "月次予算を編集し、現預金設定とあわせて出力ブックへ反映します。",
    )
    default_budget = build_budget_template(journal_df)
    editable_budget = st.data_editor(
        default_budget,
        width="stretch",
        num_rows="dynamic",
        key="budget_editor",
    )
    editable_budget["monthly_budget"] = pd.to_numeric(editable_budget["monthly_budget"], errors="coerce").fillna(0.0)
    editable_budget["annual_budget"] = pd.to_numeric(editable_budget["annual_budget"], errors="coerce").fillna(
        editable_budget["monthly_budget"] * 12.0
    )
    budget_vs_actual_df = build_budget_vs_actual(journal_df, editable_budget)

    c1, c2 = st.columns([1.4, 1.0])
    with c1:
        comparison_chart = px.bar(
            budget_vs_actual_df,
            x="budget_bucket",
            y=["monthly_budget", "actual_monthly"],
            barmode="group",
            title="Budget vs Actual",
            labels={"value": f"Monthly amount ({base_currency})", "budget_bucket": "Bucket"},
        )
        comparison_chart.update_layout(margin=dict(t=60, l=20, r=20, b=20))
        st.plotly_chart(comparison_chart, width="stretch")
    with c2:
        st.dataframe(cash_settings_df, width="stretch", hide_index=True)
        st.dataframe(budget_vs_actual_df, width="stretch", hide_index=True)

with tabs[4]:
    render_section_header(
        "Export",
        "整理済みワークブック",
        "Overview、Normalized Journal、Budget、Cash Settings、Source Files、Account Mapping を1冊にまとめます。",
    )
    budget_for_export = editable_budget.copy()
    if "monthly_budget" in budget_for_export.columns:
        budget_for_export["monthly_budget"] = pd.to_numeric(budget_for_export["monthly_budget"], errors="coerce").fillna(0.0)
    if "annual_budget" in budget_for_export.columns:
        budget_for_export["annual_budget"] = pd.to_numeric(budget_for_export["annual_budget"], errors="coerce").fillna(0.0)
    budget_vs_actual_for_export = build_budget_vs_actual(journal_df, budget_for_export)

    workbook_bytes = build_accounting_workbook(
        journal_df=journal_df,
        budget_df=budget_for_export,
        budget_vs_actual_df=budget_vs_actual_for_export,
        cash_settings_df=cash_settings_df,
        source_manifest_df=source_manifest_df,
        company_name=company_name,
    )
    st.download_button(
        "整理済みExcelをダウンロード",
        data=workbook_bytes,
        file_name=workbook_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    st.caption("IFRS変換コードはまだヒント出力に留めています。次段階で正式な科目マッピングロジックへ拡張できます。")
