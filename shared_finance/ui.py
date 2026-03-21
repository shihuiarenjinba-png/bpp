from __future__ import annotations

from html import escape
from typing import Iterable

import streamlit as st


FONT_STACK = "'Avenir Next', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', sans-serif"


def apply_theme(
    *,
    page_title: str,
    page_icon: str,
    accent: str,
    gradient_start: str,
    gradient_end: str,
    paper: str = "#fffaf4",
    ink: str = "#182230",
) -> None:
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    st.markdown(
        f"""
        <style>
        :root {{
            --accent: {accent};
            --gradient-start: {gradient_start};
            --gradient-end: {gradient_end};
            --paper: {paper};
            --ink: {ink};
            --muted: rgba(24, 34, 48, 0.72);
            --card: rgba(255, 255, 255, 0.82);
            --border: rgba(24, 34, 48, 0.12);
        }}

        html, body, [class*="css"] {{
            font-family: {FONT_STACK};
            color: var(--ink);
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(255, 255, 255, 0.88), transparent 36%),
                linear-gradient(135deg, var(--gradient-start) 0%, var(--gradient-end) 100%);
        }}

        .main .block-container {{
            max-width: 1320px;
            padding-top: 1.5rem;
            padding-bottom: 4rem;
        }}

        .hero-card {{
            background: linear-gradient(145deg, rgba(255,255,255,0.82), rgba(255,255,255,0.55));
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 1.5rem 1.6rem;
            box-shadow: 0 18px 48px rgba(24, 34, 48, 0.08);
            backdrop-filter: blur(18px);
            margin-bottom: 1.1rem;
        }}

        .hero-kicker {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            color: var(--accent);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 700;
        }}

        .hero-title {{
            margin: 0.6rem 0 0.35rem 0;
            font-size: 2.35rem;
            line-height: 1.05;
        }}

        .hero-copy {{
            margin: 0;
            color: var(--muted);
            font-size: 1.02rem;
            max-width: 860px;
        }}

        .chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }}

        .chip {{
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(24, 34, 48, 0.08);
            color: var(--ink);
            font-size: 0.86rem;
        }}

        .section-card {{
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 28px rgba(24, 34, 48, 0.05);
            margin-bottom: 1rem;
        }}

        .section-kicker {{
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.2rem;
        }}

        .section-title {{
            margin: 0;
            font-size: 1.3rem;
            line-height: 1.2;
        }}

        .section-copy {{
            margin: 0.35rem 0 0 0;
            color: var(--muted);
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.68);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 0.75rem 0.85rem;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.65rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            height: 48px;
            border-radius: 14px 14px 0 0;
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(24, 34, 48, 0.08);
            padding: 0 1rem;
        }}

        .stTabs [aria-selected="true"] {{
            background: rgba(255, 255, 255, 0.92);
            color: var(--ink);
            box-shadow: inset 0 -3px 0 var(--accent);
        }}

        .stDownloadButton > button,
        .stButton > button {{
            border-radius: 14px;
            border: 1px solid rgba(24, 34, 48, 0.1);
            box-shadow: 0 8px 20px rgba(24, 34, 48, 0.08);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, *, kicker: str, tags: Iterable[str] | None = None) -> None:
    tag_html = ""
    if tags:
        chips = "".join(f"<span class='chip'>{escape(str(tag))}</span>" for tag in tags)
        tag_html = f"<div class='chip-row'>{chips}</div>"

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-kicker">{escape(kicker)}</div>
            <h1 class="hero-title">{escape(title)}</h1>
            <p class="hero-copy">{escape(subtitle)}</p>
            {tag_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-kicker">{escape(kicker)}</div>
            <h2 class="section-title">{escape(title)}</h2>
            <p class="section-copy">{escape(copy)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

