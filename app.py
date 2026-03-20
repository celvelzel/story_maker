"""StoryWeaver – Streamlit chat-based text adventure UI.

Layout (Streamlit):
    Sidebar:    KG visualisation · consistency trend · debug info · download
    Main area:  controls · story chat · option buttons · evaluation dashboard
"""
from __future__ import annotations

import os
import sys
import time
import logging
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.game_engine import GameEngine, TurnResult
from src.nlg.option_generator import StoryOption
from src.evaluation.metrics import full_evaluation
from src.evaluation.llm_judge import judge as llm_judge
from config import settings

logger = logging.getLogger(__name__)


# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StoryWeaver - AI Story Generator",
    page_icon="🎭",
    layout="wide",
)

# Lock UI to dark mode to avoid readability issues from theme switching.
st.session_state.ui_mode = "dark"


def _theme_tokens(mode: str) -> dict[str, str]:
    """Return CSS tokens — cyberpunk neon palette."""
    if mode == "dark":
        return {
            "bg": "#06080f",
            "text": "#e0e8ff",
            "muted": "#7b8db5",
            "hero1": "#00f0ff",
            "hero2": "#7b2fff",
            "hero3": "#ff00aa",
            "hero_shadow": "rgba(0, 240, 255, 0.18)",
            "panel": "rgba(8, 12, 28, 0.88)",
            "panel_border": "rgba(0, 240, 255, 0.12)",
            "title": "#00f0ff",
            "primary": "#00f0ff",
            "primary_hover": "#7b2fff",
            "input_bg": "rgba(6, 10, 24, 0.92)",
            "input_border": "rgba(0, 240, 255, 0.22)",
            "chat_bg": "rgba(0, 240, 255, 0.03)",
            "neon_cyan": "#00f0ff",
            "neon_magenta": "#ff00aa",
            "neon_purple": "#7b2fff",
            "neon_gold": "#ffd700",
        }

    # Light mode fallback — still cyberpunk-tinted
    return {
        "bg": "#06080f",
        "text": "#e0e8ff",
        "muted": "#7b8db5",
        "hero1": "#00f0ff",
        "hero2": "#7b2fff",
        "hero3": "#ff00aa",
        "hero_shadow": "rgba(0, 240, 255, 0.18)",
        "panel": "rgba(8, 12, 28, 0.88)",
        "panel_border": "rgba(0, 240, 255, 0.12)",
        "title": "#00f0ff",
        "primary": "#00f0ff",
        "primary_hover": "#7b2fff",
        "input_bg": "rgba(6, 10, 24, 0.92)",
        "input_border": "rgba(0, 240, 255, 0.22)",
        "chat_bg": "rgba(0, 240, 255, 0.03)",
        "neon_cyan": "#00f0ff",
        "neon_magenta": "#ff00aa",
        "neon_purple": "#7b2fff",
        "neon_gold": "#ffd700",
    }


tokens = _theme_tokens(st.session_state.ui_mode)

st.markdown(
    f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;800;900&family=Rajdhani:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;700;800&display=swap');

    /* ── Keyframe Animations ─────────────────────────────── */
    @keyframes gradientShift {{
        0%   {{ background-position: 0% 50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    @keyframes borderPulse {{
        0%, 100% {{ box-shadow: 0 0 8px rgba(0,240,255,0.3), 0 0 20px rgba(0,240,255,0.1); }}
        50%      {{ box-shadow: 0 0 16px rgba(0,240,255,0.5), 0 0 40px rgba(123,47,255,0.2); }}
    }}
    @keyframes scanLine {{
        0%   {{ transform: translateX(-100%); opacity: 0.6; }}
        100% {{ transform: translateX(100%); opacity: 0; }}
    }}
    @keyframes slideInLeft {{
        from {{ opacity: 0; transform: translateX(-24px); }}
        to   {{ opacity: 1; transform: translateX(0); }}
    }}
    @keyframes slideInRight {{
        from {{ opacity: 0; transform: translateX(24px); }}
        to   {{ opacity: 1; transform: translateX(0); }}
    }}
    @keyframes neonPulse {{
        0%, 100% {{ box-shadow: 0 0 6px {tokens['neon_cyan']}44, 0 0 18px {tokens['neon_cyan']}22; }}
        50%      {{ box-shadow: 0 0 12px {tokens['neon_cyan']}88, 0 0 36px {tokens['neon_purple']}33; }}
    }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(16px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes gridMove {{
        0%   {{ background-position: 0 0; }}
        100% {{ background-position: 0 40px; }}
    }}
    @keyframes glowText {{
        0%, 100% {{ text-shadow: 0 0 8px {tokens['neon_cyan']}88, 0 0 20px {tokens['neon_cyan']}44; }}
        50%      {{ text-shadow: 0 0 14px {tokens['neon_cyan']}cc, 0 0 32px {tokens['neon_purple']}55; }}
    }}
    @keyframes chatBreath {{
        0%, 100% {{
            border-color: {tokens['neon_cyan']}66;
            box-shadow: 0 0 12px {tokens['neon_cyan']}40, 0 0 32px {tokens['neon_cyan']}18, inset 0 0 14px {tokens['neon_cyan']}10;
        }}
        50% {{
            border-color: {tokens['neon_cyan']}cc;
            box-shadow: 0 0 22px {tokens['neon_cyan']}66, 0 0 56px {tokens['neon_purple']}30, inset 0 0 24px {tokens['neon_cyan']}18;
        }}
    }}
    @keyframes particleDrift {{
        0%   {{ transform: translateY(0) translateX(0); opacity: 0; }}
        10%  {{ opacity: 1; }}
        90%  {{ opacity: 1; }}
        100% {{ transform: translateY(-120px) translateX(30px); opacity: 0; }}
    }}

    /* ── Global Reset & Base ──────────────────────────────── */
    html, body, [class*="css"] {{
        font-family: 'Rajdhani', 'Noto Sans SC', sans-serif;
        color: {tokens['text']};
    }}

    .stApp {{
        background: {tokens['bg']};
        color: {tokens['text']};
    }}

    /* Grid overlay on main area */
    .stApp::before {{
        content: '';
        position: fixed;
        inset: 0;
        background:
            repeating-linear-gradient(
                0deg,
                transparent,
                transparent 39px,
                rgba(0, 240, 255, 0.03) 39px,
                rgba(0, 240, 255, 0.03) 40px
            ),
            repeating-linear-gradient(
                90deg,
                transparent,
                transparent 39px,
                rgba(0, 240, 255, 0.03) 39px,
                rgba(0, 240, 255, 0.03) 40px
            );
        animation: gridMove 8s linear infinite;
        pointer-events: none;
        z-index: 0;
    }}

    /* Subtle scanline overlay */
    .stApp::after {{
        content: '';
        position: fixed;
        inset: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0, 240, 255, 0.015) 2px,
            rgba(0, 240, 255, 0.015) 4px
        );
        pointer-events: none;
        z-index: 0;
    }}

    /* ── Scrollbar ────────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: #06080f; }}
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {tokens['neon_cyan']}66, {tokens['neon_purple']}66);
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: {tokens['neon_cyan']}; }}

    /* ── Hero Banner ──────────────────────────────────────── */
    .hero {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1224 40%, #0f0a20 100%);
        border: 1px solid rgba(0,240,255,0.15);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 16px;
        animation: borderPulse 4s ease-in-out infinite;
    }}

    .hero::before {{
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, {tokens['neon_cyan']}08, {tokens['neon_purple']}06, {tokens['neon_magenta']}04);
        border-radius: 16px;
    }}

    .hero::after {{
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60%;
        height: 2px;
        background: linear-gradient(90deg, {tokens['neon_cyan']}, {tokens['neon_purple']}, transparent);
        animation: scanLine 4s ease-in-out infinite;
    }}

    .hero h2 {{
        position: relative;
        margin: 0 0 10px 0;
        font-family: 'Orbitron', 'Noto Sans SC', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 1.5px;
        animation: glowText 3s ease-in-out infinite;
    }}

    .hero p {{
        position: relative;
        margin: 0;
        font-size: 0.92rem;
        color: rgba(224, 232, 255, 0.85);
        letter-spacing: 0.5px;
    }}

    .hero .hero-tag {{
        display: inline-block;
        margin-top: 12px;
        padding: 3px 14px;
        font-family: 'Orbitron', monospace;
        font-size: 0.65rem;
        font-weight: 500;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: {tokens['neon_cyan']};
        border: 1px solid {tokens['neon_cyan']}44;
        border-radius: 20px;
        background: {tokens['neon_cyan']}0a;
        position: relative;
    }}

    /* ── Section Titles ───────────────────────────────────── */
    .section-title {{
        font-family: 'Orbitron', 'Noto Sans SC', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: {tokens['neon_cyan']};
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 4px;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid {tokens['neon_cyan']}22;
        text-shadow: 0 0 8px {tokens['neon_cyan']}44;
    }}

    .muted-note {{
        color: {tokens['muted']};
        font-size: 0.88rem;
    }}

    .option-meta-center {{
        text-align: center;
        color: {tokens['muted']};
        font-size: 0.88rem;
        margin-top: 6px;
    }}

    /* ── Neon Divider (replaces st.markdown("---")) ───────── */
    .neon-divider {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, {tokens['neon_cyan']}55, {tokens['neon_purple']}55, transparent);
        margin: 16px 0;
        box-shadow: 0 0 8px {tokens['neon_cyan']}22;
    }}

    /* ── Glassmorphism Cards ──────────────────────────────── */
    .metric-card {{
        background: {tokens['panel']};
        border: 1px solid {tokens['panel_border']};
        border-left: 3px solid {tokens['neon_cyan']}88;
        border-radius: 12px;
        padding: 14px 16px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.5s ease-out;
        position: relative;
    }}
    .metric-card:hover {{
        transform: translateY(-5px);
        border-color: {tokens['neon_cyan']}66;
        border-left-color: {tokens['neon_cyan']}cc;
        box-shadow: 0 10px 40px rgba(0,240,255,0.18), 0 0 20px rgba(0,240,255,0.1), inset 0 0 20px rgba(0,240,255,0.06);
        background: linear-gradient(135deg, rgba(0,240,255,0.05), {tokens['panel']});
    }}

    /* ── Chat Messages ────────────────────────────────────── */
    .stChatMessage {{
        background: {tokens['chat_bg']};
        border: 1px solid {tokens['panel_border']};
        border-radius: 12px;
        animation: slideInLeft 0.4s ease-out;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .stChatMessage:hover {{
        border-color: {tokens['neon_cyan']}44;
        box-shadow: 0 4px 20px rgba(0,240,255,0.1);
        background: rgba(0,240,255,0.05);
        transform: translateX(2px);
    }}

    /* User messages — cyan accent */
    .stChatMessage[data-testid="stChatMessage"]:has(.stMarkdown) {{
        border-left: 3px solid {tokens['neon_cyan']}55;
    }}

    .st-expander {{
        background: {tokens['chat_bg']};
        border: 1px solid {tokens['panel_border']};
        border-radius: 12px;
        animation: fadeInUp 0.4s ease-out;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .st-expander:hover {{
        border-color: {tokens['neon_cyan']}55;
        box-shadow: 0 4px 20px rgba(0,240,255,0.1);
        background: rgba(0,240,255,0.05);
        transform: translateY(-2px);
    }}

    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {{
        border: none;
        border-radius: 8px;
    }}

    /* ── Input Fields ─────────────────────────────────────── */
    .stTextInput > div > div > input,
    div[data-baseweb="select"] > div,
    .stTextArea textarea {{
        background: {tokens['input_bg']} !important;
        border: 1px solid {tokens['input_border']} !important;
        color: {tokens['text']} !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {{
        border-color: {tokens['neon_cyan']}66 !important;
        box-shadow: 0 0 12px {tokens['neon_cyan']}22 !important;
        outline: none !important;
    }}

    /* ── GLOBAL: kill Streamlit default red focus ring everywhere ── */
    *:focus,
    *:focus-visible,
    *:focus-within {{
        outline-color: {tokens['neon_cyan']}55 !important;
    }}
    .stTextInput > div:focus-within,
    .stTextInput > div > div:focus-within {{
        border-color: {tokens['neon_cyan']}55 !important;
        box-shadow: 0 0 10px {tokens['neon_cyan']}18 !important;
    }}
    /* Streamlit uses a colored shadow on wrapper divs — override ALL layers */
    .stTextInput,
    .stTextInput > *,
    .stTextInput > div,
    .stTextInput > div > *,
    .stTextInput > div > div {{
        background: transparent !important;
        border-color: {tokens['input_border']} !important;
        box-shadow: none !important;
    }}
    .stTextInput > div:focus-within {{
        border-color: {tokens['neon_cyan']}55 !important;
    }}
    /* Also for chat input wrapper divs */
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div {{
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }}

    /* ── Buttons ──────────────────────────────────────────── */
    .stButton > button {{
        border-radius: 10px;
        border: 1px solid {tokens['panel_border']};
        background: {tokens['panel']};
        color: {tokens['text']};
        font-family: 'Rajdhani', 'Noto Sans SC', sans-serif;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(8px);
    }}
    .stButton > button:hover {{
        border-color: {tokens['neon_cyan']}55;
        color: {tokens['neon_cyan']};
        box-shadow: 0 4px 20px rgba(0,240,255,0.15), 0 0 8px rgba(0,240,255,0.1);
        transform: translateY(-1px);
    }}
    .stButton > button:active {{
        transform: scale(0.97);
        box-shadow: 0 0 24px {tokens['neon_cyan']}44 inset;
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {tokens['neon_cyan']}22, {tokens['neon_purple']}22) !important;
        border: 1px solid {tokens['neon_cyan']}55 !important;
        color: {tokens['neon_cyan']} !important;
        font-family: 'Orbitron', 'Noto Sans SC', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
        animation: neonPulse 3s ease-in-out infinite;
    }}
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, {tokens['neon_cyan']}33, {tokens['neon_purple']}33) !important;
        border-color: {tokens['neon_cyan']}88 !important;
        box-shadow: 0 0 24px {tokens['neon_cyan']}33, 0 0 48px {tokens['neon_purple']}18 !important;
        transform: translateY(-2px);
    }}

    /* ── Text ─────────────────────────────────────────────── */
    .stMarkdown, .stCaption, label, p, span {{
        color: {tokens['text']};
    }}

    /* ── Sidebar ──────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(6,8,15,0.97) 0%, rgba(8,12,28,0.95) 100%);
        border-right: 1px solid {tokens['neon_cyan']}15;
        backdrop-filter: blur(16px);
    }}

    section[data-testid="stSidebar"]::before {{
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 1px;
        height: 100%;
        background: linear-gradient(180deg, transparent, {tokens['neon_cyan']}33, {tokens['neon_purple']}33, transparent);
    }}

    /* ── KG Visualization Frame ───────────────────────────── */
    .kg-frame {{
        border: 1px solid {tokens['neon_cyan']}22;
        border-radius: 12px;
        padding: 2px;
        background: rgba(0,240,255,0.02);
        box-shadow: 0 0 20px rgba(0,240,255,0.05), inset 0 0 20px rgba(0,240,255,0.02);
        animation: borderPulse 6s ease-in-out infinite;
    }}

    /* ── Progress bars — neon override ────────────────────── */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {tokens['neon_cyan']}, {tokens['neon_purple']}) !important;
        border-radius: 4px;
        box-shadow: 0 0 8px {tokens['neon_cyan']}44;
    }}

    /* ── Metric components ────────────────────────────────── */
    [data-testid="stMetric"] {{
        background: {tokens['panel']};
        border: 1px solid {tokens['panel_border']};
        border-radius: 12px;
        padding: 12px 16px;
        backdrop-filter: blur(8px);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    [data-testid="stMetric"]:hover {{
        border-color: {tokens['neon_cyan']}55;
        box-shadow: 0 6px 24px rgba(0,240,255,0.15), 0 0 12px rgba(123,47,255,0.08);
        transform: translateY(-3px);
        background: linear-gradient(135deg, rgba(0,240,255,0.03), transparent);
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {tokens['neon_cyan']} !important;
        font-family: 'Orbitron', monospace;
        font-weight: 700;
    }}
    [data-testid="stMetric"] [data-testid="stMetricDelta"] svg {{
        display: inline;
    }}

    /* ── st.info / st.warning custom ─────────────────────── */
    .stAlert {{
        border-radius: 10px;
        border: 1px solid {tokens['panel_border']};
        backdrop-filter: blur(8px);
        background: {tokens['panel']} !important;
        transition: all 0.3s ease;
        box-shadow: 0 0 8px rgba(0,240,255,0.05);
    }}
    .stAlert:hover {{
        border-color: {tokens['neon_cyan']}33;
        box-shadow: 0 0 16px rgba(0,240,255,0.12);
    }}

    /* ── Streamlit top header bar (Deploy / "..." menu) ──── */
    header[data-testid="stHeader"] {{
        background: rgba(6, 8, 15, 0.92) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid {tokens['neon_cyan']}10;
    }}

    /* ── Chat input bottom container — nuclear dark override ─ */
    [data-testid="stBottom"],
    [data-testid="stBottom"] *,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottomBlockContainer"] *,
    .stBottom,
    .stBottom *,
    .stChatInput,
    .stChatInput *,
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] *,
    [data-testid="stChatInputContainer"],
    [data-testid="stChatInputContainer"] * {{
        background: rgba(6, 8, 15, 0.95) !important;
        background-color: rgba(6, 8, 15, 0.95) !important;
    }}
    [data-testid="stBottom"] {{
        border-top: 1px solid {tokens['neon_cyan']}12;
    }}
    /* The actual textarea/input inside — transparent on top of dark parent */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input,
    .stChatInput textarea,
    .stChatInput input,
    .stChatInputContainer textarea,
    .stChatInputContainer input {{
        color: {tokens['text']} !important;
        background: transparent !important;
        background-color: transparent !important;
    }}
    /* Neon breathing border — always visible */
    [data-testid="stChatInput"],
    .stChatInput {{
        border: 1px solid {tokens['neon_cyan']}35 !important;
        border-radius: 10px !important;
        animation: chatBreath 4s ease-in-out infinite !important;
        outline: none !important;
    }}
    /* Kill ALL red/orange focus outlines inside chat input */
    [data-testid="stChatInput"] *:focus,
    [data-testid="stChatInput"] *:focus-visible,
    [data-testid="stChatInput"] *:focus-within,
    [data-testid="stChatInput"]:focus,
    [data-testid="stChatInput"]:focus-visible,
    [data-testid="stChatInput"]:focus-within,
    .stChatInput *:focus,
    .stChatInput *:focus-visible,
    .stChatInput *:focus-within,
    .stChatInput:focus,
    .stChatInput:focus-visible,
    .stChatInput:focus-within {{
        outline: none !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder,
    .stChatInput textarea::placeholder {{
        color: {tokens['muted']} !important;
    }}
    /* Send button inside chat input */
    [data-testid="stChatInput"] button,
    .stChatInput button {{
        background: transparent !important;
        background-color: transparent !important;
        color: {tokens['neon_cyan']} !important;
    }}

    /* ── Expander panels (NLU Path / KG Strategy / NLU Details) ── */
    [data-testid="stExpander"] {{
        background: {tokens['panel']} !important;
        border: 1px solid {tokens['panel_border']} !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    [data-testid="stExpander"]:hover {{
        border-color: {tokens['neon_cyan']}44 !important;
        background: rgba(8, 12, 28, 0.96) !important;
        box-shadow: 0 4px 24px rgba(0,240,255,0.12), 0 0 12px rgba(0,240,255,0.08);
        transform: translateY(-2px);
    }}
    [data-testid="stExpander"] details {{
        background: transparent !important;
    }}
    [data-testid="stExpander"] summary {{
        background: transparent !important;
        color: {tokens['text']} !important;
        cursor: pointer;
        transition: all 0.25s ease;
    }}
    [data-testid="stExpander"] summary:hover {{
        color: {tokens['neon_cyan']} !important;
        text-shadow: 0 0 8px {tokens['neon_cyan']}44;
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        background: transparent !important;
    }}

    /* ── Selectbox / dropdown inside expanders ────────────── */
    div[data-baseweb="select"] {{
        background: transparent !important;
        transition: all 0.3s ease;
    }}
    div[data-baseweb="select"] > div {{
        background: {tokens['input_bg']} !important;
        border: 1px solid {tokens['input_border']} !important;
        color: {tokens['text']} !important;
        transition: all 0.3s ease;
    }}
    div[data-baseweb="select"]:hover > div {{
        border-color: {tokens['neon_cyan']}44 !important;
        box-shadow: 0 0 12px {tokens['neon_cyan']}15 !important;
        background: rgba(6, 10, 24, 0.98) !important;
    }}
    /* Dropdown menu / popover */
    div[data-baseweb="popover"] {{
        background: #0a0e1a !important;
        border: 1px solid {tokens['neon_cyan']}22 !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="popover"] ul {{
        background: #0a0e1a !important;
    }}
    div[data-baseweb="popover"] li {{
        background: transparent !important;
        color: {tokens['text']} !important;
    }}
    div[data-baseweb="popover"] li:hover {{
        background: {tokens['neon_cyan']}15 !important;
    }}
    /* Selected option highlight */
    div[data-baseweb="popover"] li[aria-selected="true"] {{
        background: {tokens['neon_cyan']}18 !important;
        color: {tokens['neon_cyan']} !important;
    }}

    /* ── Tooltip / help popovers ──────────────────────────── */
    [data-testid="stTooltipContent"] {{
        background: #0a0e1a !important;
        color: {tokens['text']} !important;
        border: 1px solid {tokens['neon_cyan']}22 !important;
    }}

    /* ── General block containers (catch-all for white leaks) */
    .stMainBlockContainer, .stSidebarBlockContainer,
    [data-testid="stAppViewBlockContainer"] {{
        background: transparent !important;
    }}

    /* ── Toggle switch ────────────────────────────────────── */
    .stCheckbox label span[data-testid="stCheckbox"] {{
        color: {tokens['text']};
    }}

    /* ── Line chart neon ──────────────────────────────────── */
    .stLineChart {{
        border: 1px solid {tokens['panel_border']};
        border-radius: 10px;
        padding: 4px;
    }}

    /* ── Download button ──────────────────────────────────── */
    .stDownloadButton > button {{
        border: 1px solid {tokens['neon_cyan']}33 !important;
        background: {tokens['neon_cyan']}0a !important;
        color: {tokens['neon_cyan']} !important;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    .stDownloadButton > button:hover {{
        background: {tokens['neon_cyan']}18 !important;
        box-shadow: 0 0 20px {tokens['neon_cyan']}33, 0 0 40px {tokens['neon_cyan']}15;
        transform: translateY(-2px);
        border-color: {tokens['neon_cyan']}66 !important;
    }}

    /* ── Eval section title decoration ────────────────────── */
    .eval-title {{
        font-family: 'Orbitron', 'Noto Sans SC', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: {tokens['neon_cyan']};
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding-bottom: 8px;
        border-bottom: 2px solid transparent;
        background-image: linear-gradient({tokens['bg']}, {tokens['bg']}), linear-gradient(90deg, {tokens['neon_cyan']}, {tokens['neon_purple']});
        background-origin: border-box;
        background-clip: padding-box, border-box;
        text-shadow: 0 0 10px {tokens['neon_cyan']}44;
    }}

    /* ── Modal / Dialog Boxes (Deploy, Settings popups) ───── */
    [role="dialog"],
    dialog,
    .stModal,
    [data-testid="stModal"] {{
        background: {tokens['bg']} !important;
        border: 1px solid {tokens['neon_cyan']}33 !important;
        border-radius: 12px !important;
        box-shadow: 0 0 40px rgba(0,240,255,0.2), 0 0 80px rgba(123,47,255,0.1) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }}
    [role="dialog"] * ,
    dialog *,
    .stModal *,
    [data-testid="stModal"] * {{
        background: transparent !important;
        color: {tokens['text']} !important;
    }}
    /* Dialog overlay / backdrop */
    [role="dialog"]::backdrop,
    dialog::backdrop {{
        background: rgba(0, 0, 0, 0.7) !important;
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
    }}
    /* Dialog buttons */
    [role="dialog"] button,
    dialog button,
    .stModal button,
    [data-testid="stModal"] button {{
        background: linear-gradient(135deg, {tokens['neon_cyan']}22, {tokens['neon_purple']}22) !important;
        border: 1px solid {tokens['neon_cyan']}55 !important;
        color: {tokens['neon_cyan']} !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }}
    [role="dialog"] button:hover,
    dialog button:hover,
    .stModal button:hover,
    [data-testid="stModal"] button:hover {{
        background: linear-gradient(135deg, {tokens['neon_cyan']}33, {tokens['neon_purple']}33) !important;
        box-shadow: 0 0 20px {tokens['neon_cyan']}44, 0 0 40px {tokens['neon_purple']}22 !important;
        transform: translateY(-1px);
    }}
    /* Dialog headings */
    [role="dialog"] h1,
    [role="dialog"] h2,
    [role="dialog"] h3,
    dialog h1,
    dialog h2,
    dialog h3,
    .stModal h1,
    .stModal h2,
    .stModal h3 {{
        color: {tokens['neon_cyan']} !important;
        font-family: 'Orbitron', 'Noto Sans SC', sans-serif;
    }}
    /* Dialog text inputs */
    [role="dialog"] input,
    [role="dialog"] textarea,
    dialog input,
    dialog textarea,
    .stModal input,
    .stModal textarea {{
        background: {tokens['input_bg']} !important;
        border: 1px solid {tokens['input_border']} !important;
        color: {tokens['text']} !important;
        border-radius: 8px !important;
    }}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h2>&#x1F3AD; STORYWEAVER</h2>
    <p>Multi-turn interactive storytelling system powered by NLU + LLM + KG &mdash; Dynamic Knowledge Graph · Real-time World State Tracking · Session Evaluation</p>
  <span class="hero-tag">&#x26A1; Dynamic KG Story Engine</span>
</div>
""",
    unsafe_allow_html=True,
)


# ── Session State initialisation ─────────────────────────────────────────

_DEFAULTS = {
    "engine": None,
    "history": [],                # list[dict] with role / content
    "consistency_history": [],    # float per turn
    "kg_html": "",
    "options": [],                # list[StoryOption]
    "nlu_debug": {},
    "eval_result": "",
    "eval_auto": {},
    "eval_llm": {},
    "eval_prev_auto": {},
    "eval_prev_llm": {},
    "eval_at": "",
    "chat_fold_mode": False,
    "last_elapsed": 0.0,
    "intent_model_path": str(settings.INTENT_MODEL_PATH),
    # KG strategy settings
    "kg_conflict_resolution": settings.KG_CONFLICT_RESOLUTION,
    "kg_extraction_mode": settings.KG_EXTRACTION_MODE,
    "kg_importance_mode": settings.KG_IMPORTANCE_MODE,
    "kg_summary_mode": settings.KG_SUMMARY_MODE,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ──────────────────────────────────────────────────────────────

def _process_action(action: str) -> None:
    """Run a player action through the engine and update session state."""
    engine: GameEngine | None = st.session_state.engine
    if engine is None:
        st.warning("Please start a new game before entering an action.")
        return

    start = time.time()
    st.session_state.history.append({"role": "user", "content": action})

    result: TurnResult = engine.process_turn(action)

    assistant_msg = result.story_text
    if result.conflicts:
        assistant_msg += "\n\n*⚠ World-consistency notes:*\n" + "\n".join(
            f"- {c}" for c in result.conflicts
        )
    st.session_state.history.append({"role": "assistant", "content": assistant_msg})

    st.session_state.kg_html = result.kg_html
    st.session_state.options = result.options
    if result.nlu_debug:
        st.session_state.nlu_debug = result.nlu_debug

    # Consistency tracking (1.0 = perfect, decreases with conflicts)
    n_conflicts = len(result.conflicts)
    score = 1.0 if n_conflicts == 0 else max(0.0, 1.0 - n_conflicts * 0.2)
    st.session_state.consistency_history.append(score)

    st.session_state.last_elapsed = time.time() - start


def _run_evaluation() -> tuple[str, dict, dict]:
    """Collect session data from the engine and run all evaluations."""
    engine: GameEngine | None = st.session_state.engine
    if engine is None or not engine.all_story_texts:
        return "*No evaluable content yet. Please start and progress the story first.*", {}, {}

    texts = engine.all_story_texts

    # Automatic metrics
    auto = full_evaluation(
        texts=texts,
        entity_names=engine.kg_entity_names,
        turn_conflict_counts=engine.turn_conflict_counts,
    )

    # LLM judge
    transcript = "\n".join(texts)
    llm_scores = llm_judge(transcript)

    # Format as Markdown table (kept for one-click report display)
    lines = [
        "### Automatic Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Distinct-1 | {auto.get('distinct_1', 0):.4f} |",
        f"| Distinct-2 | {auto.get('distinct_2', 0):.4f} |",
        f"| Distinct-3 | {auto.get('distinct_3', 0):.4f} |",
        f"| Self-BLEU | {auto.get('self_bleu', 0):.4f} |",
        f"| Entity Coverage | {auto.get('entity_coverage', 0):.2%} |",
        f"| Consistency Rate | {auto.get('consistency_rate', 0):.2%} |",
        "",
        "### LLM Judge (1-10)",
        "",
        "| Dimension | Score |",
        "|-----------|-------|",
        f"| Narrative Quality | {llm_scores.get('narrative_quality', 0)} |",
        f"| Consistency | {llm_scores.get('consistency', 0)} |",
        f"| Player Agency | {llm_scores.get('player_agency', 0)} |",
        f"| Creativity | {llm_scores.get('creativity', 0)} |",
        f"| Pacing | {llm_scores.get('pacing', 0)} |",
        f"| **Average** | **{llm_scores.get('average', 0)}** |",
    ]
    return "\n".join(lines), auto, llm_scores


def _story_turn_count() -> int:
    """Return number of user turns in current chat history."""
    return sum(1 for m in st.session_state.history if m["role"] == "user")


def _build_turn_pairs(history: list[dict]) -> tuple[str, list[tuple[int, str, str]]]:
    """Return opening narration and user-assistant turn pairs."""
    opening = ""
    pairs: list[tuple[int, str, str]] = []
    i = 0

    if history and history[0]["role"] == "assistant":
        opening = history[0]["content"]
        i = 1

    turn = 1
    while i < len(history):
        user_text = ""
        ai_text = ""

        if i < len(history) and history[i]["role"] == "user":
            user_text = history[i]["content"]
            i += 1
        if i < len(history) and history[i]["role"] == "assistant":
            ai_text = history[i]["content"]
            i += 1

        if user_text or ai_text:
            pairs.append((turn, user_text, ai_text))
            turn += 1

    return opening, pairs


def _delta_str(current: float, previous: dict, key: str, fmt: str = ".4f") -> str | None:
    """Format metric delta string for st.metric."""
    if not previous or key not in previous:
        return None
    diff = float(current) - float(previous.get(key, 0))
    return f"{diff:+{fmt}}"


def _delta_pct(current: float, previous: dict, key: str) -> str | None:
    """Format percentage-point delta for st.metric."""
    if not previous or key not in previous:
        return None
    diff = float(current) - float(previous.get(key, 0))
    return f"{diff * 100:+.2f}pp"


# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    with st.expander("🧠 NLU Model Path", expanded=False):
        st.session_state.intent_model_path = st.text_input(
            "Intent model directory",
            value=st.session_state.intent_model_path,
            help="Leave empty to use the default directory; if it doesn't exist, it falls back to rule_fallback.",
        )

    with st.expander("⚙ KG Strategy Settings", expanded=False):
        st.caption("Strategy changes take effect after the next \"Start New Game\".")

        st.session_state.kg_conflict_resolution = st.selectbox(
            "Conflict resolution strategy",
            ["llm_arbitrate", "keep_latest"],
            index=0 if st.session_state.kg_conflict_resolution == "llm_arbitrate" else 1,
            help=(
                "llm_arbitrate: LLM decides which information to keep (best quality)\n"
                "keep_latest: Keep the most recently updated information (no LLM call required)"
            ),
        )

        st.session_state.kg_extraction_mode = st.selectbox(
            "Entity extraction mode",
            ["dual_extract", "story_only"],
            index=0 if st.session_state.kg_extraction_mode == "dual_extract" else 1,
            help=(
                "dual_extract: Extract from both player input and story text for fuller context\n"
                "story_only: Extract only from story text (backward compatible)"
            ),
        )

        st.session_state.kg_summary_mode = st.selectbox(
            "KG summary format",
            ["layered", "flat"],
            index=0 if st.session_state.kg_summary_mode == "layered" else 1,
            help=(
                "layered: Group by importance (core/secondary/background), with descriptions and timeline\n"
                "flat: Simple list format (backward compatible)"
            ),
        )

        st.session_state.kg_importance_mode = st.selectbox(
            "Entity pruning strategy",
            ["composite", "degree_only"],
            index=0 if st.session_state.kg_importance_mode == "composite" else 1,
            help=(
                "composite: Score by degree + recency + mention_count\n"
                "degree_only: Degree-only pruning (backward compatible)"
            ),
        )

    st.markdown("<div class='section-title'>&#x1F4CA; Story World Knowledge Graph</div>", unsafe_allow_html=True)
    if st.session_state.kg_html:
        st.markdown("<div class='kg-frame'>", unsafe_allow_html=True)
        components.html(st.session_state.kg_html, height=480, scrolling=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("The knowledge graph will appear after starting a game.")

    st.markdown("<div class='section-title'>&#x1F4C8; Consistency Trend</div>", unsafe_allow_html=True)
    if st.session_state.consistency_history:
        recent = st.session_state.consistency_history[-5:]
        offset = max(0, len(st.session_state.consistency_history) - 5)
        for i, score in enumerate(recent):
            st.progress(
                max(0.0, min(1.0, score)),
                text=f"Turn {offset + i + 1}: {score:.2f}",
            )
        st.line_chart(st.session_state.consistency_history, height=120, use_container_width=True)
    else:
        st.caption("Consistency scores appear after the first action")

    with st.expander("🔍 NLU Parsing Details", expanded=False):
        nlu = st.session_state.nlu_debug
        if nlu:
            st.markdown(f"**Resolved Input:** {nlu.get('resolved_input', '')}")
            st.markdown(
                f"**Intent:** {nlu.get('intent', '?')}  "
                f"(confidence {nlu.get('confidence', 0):.2f})"
            )
            st.markdown(f"**Backend:** {nlu.get('intent_backend', 'rule_fallback')}")
            st.markdown(f"**Model Loaded:** {nlu.get('intent_model_loaded', False)}")
            st.markdown(f"**Entities:** {nlu.get('entities', [])}")
        else:
            st.caption("NLU parsing details will appear after each action")

    st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)

    turn_count = _story_turn_count()
    engine: GameEngine | None = st.session_state.engine
    entity_count = len(engine.kg_entity_names) if engine else 0
    conflict_total = sum(engine.turn_conflict_counts) if engine else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Turns", turn_count)
    c2.metric("Entities", entity_count)
    c3.metric("Conflicts", conflict_total)

    if engine:
        st.caption(
            f"Strategy: {engine.conflict_resolution} | {engine.extraction_mode} | "
            f"{engine.summary_mode} | {engine.importance_mode}"
        )

    st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)

    if st.session_state.history:
        full_story = "\n\n".join(
            f"{'[Player]' if m['role'] == 'user' else '[Story]'}: {m['content']}"
            for m in st.session_state.history
        )
        st.download_button(
            "📥 Download Full Story",
            full_story,
            file_name="full_story.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ── Main area – Game controls ────────────────────────────────────────────

col_genre, col_btn = st.columns([3, 1])
with col_genre:
    genre = st.text_input(
        "Genre",
        value="fantasy",
        placeholder="Genre, e.g. fantasy / sci-fi / mystery",
        label_visibility="collapsed",
    )
with col_btn:
    new_game_clicked = st.button(
        "🎮 Start New Game", type="primary", use_container_width=True
    )

if new_game_clicked:
    with st.spinner("Initializing the adventure world…"):
        intent_model_path = st.session_state.intent_model_path.strip() or None
        engine = GameEngine(
            genre=genre or "fantasy",
            intent_model_path=intent_model_path,
            conflict_resolution=st.session_state.kg_conflict_resolution,
            extraction_mode=st.session_state.kg_extraction_mode,
            importance_mode=st.session_state.kg_importance_mode,
            summary_mode=st.session_state.kg_summary_mode,
        )
        st.session_state.engine = engine
        result: TurnResult = engine.start_game()
        st.session_state.history = [
            {"role": "assistant", "content": result.story_text}
        ]
        st.session_state.kg_html = result.kg_html
        st.session_state.options = result.options
        st.session_state.consistency_history = []
        st.session_state.nlu_debug = {}
        st.session_state.eval_result = ""
        st.session_state.eval_auto = {}
        st.session_state.eval_llm = {}
        st.session_state.eval_prev_auto = {}
        st.session_state.eval_prev_llm = {}
        st.session_state.eval_at = ""
        st.session_state.last_elapsed = 0.0
    st.rerun()

if st.session_state.engine is None:
    st.info("Click \"Start New Game\" above to begin the interactive story.")


# ── Chat history ─────────────────────────────────────────────────────────

chat_fold_mode = st.toggle(
    "Fold history by turn",
    value=st.session_state.chat_fold_mode,
    help="When enabled, each turn is folded as \"user input + system response\" for easier long-session reading.",
)
st.session_state.chat_fold_mode = chat_fold_mode

if chat_fold_mode:
    opening, turn_pairs = _build_turn_pairs(st.session_state.history)
    if opening:
        with st.chat_message("assistant"):
            st.markdown(opening)

    for turn, user_text, ai_text in turn_pairs:
        preview = user_text[:28] + ("..." if len(user_text) > 28 else "")
        title = f"Turn {turn} | {preview or '(empty input)'}"
        with st.expander(title, expanded=(turn == len(turn_pairs))):
            if user_text:
                st.markdown("**You:**")
                st.markdown(user_text)
            if ai_text:
                st.markdown("**System:**")
                st.markdown(ai_text)
else:
    for msg in st.session_state.history:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])


# ── Option buttons ───────────────────────────────────────────────────────

if st.session_state.options:
    st.markdown("<div class='section-title'>&#x1F9ED; Branch Options</div>", unsafe_allow_html=True)
    st.caption("You can click an option directly, or type a free-form action below.")
    opt_cols = st.columns(len(st.session_state.options))
    for idx, opt in enumerate(st.session_state.options):
        with opt_cols[idx]:
            #st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            #st.caption(f"Intent: {opt.intent_hint} | Risk: {opt.risk_level}")
            st.markdown(
                f"<div class='option-meta-center'>Intent: {opt.intent_hint} | Risk: {opt.risk_level}</div>",
                unsafe_allow_html=True,
            )
            btn_key = f"opt_{idx}_{len(st.session_state.history)}"
            if st.button(
                f"{idx + 1}. {opt.text}",
                key=btn_key,
                use_container_width=True,
            ):
                _process_action(opt.text)
                st.rerun()
            #st.markdown(
            #    f"<div class='option-meta-center'>Intent: {opt.intent_hint} | Risk: {opt.risk_level}</div>",
            #    unsafe_allow_html=True,
            #)
            st.markdown("</div>", unsafe_allow_html=True)


# ── Chat input ───────────────────────────────────────────────────────────

user_input = st.chat_input("Enter your action (e.g., investigate the runes in the ruins)…")
if user_input:
    _process_action(user_input)
    st.rerun()


# ── Performance footer ──────────────────────────────────────────────────

if st.session_state.last_elapsed > 0:
    st.caption(f"✅ Generation time for this turn: {st.session_state.last_elapsed:.2f}s")


# ── Evaluation Dashboard (kept and enhanced) ───────────────────────────

st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>&#x1F4CA; Session Evaluation Panel</div>", unsafe_allow_html=True)

col_eval_btn, col_eval_hint = st.columns([1, 2])
with col_eval_btn:
    run_eval = st.button("Run Evaluation", use_container_width=True)
with col_eval_hint:
    st.markdown("<div class='muted-note'>Evaluation computes automatic metrics from the current session and calls the LLM Judge for scoring.</div>", unsafe_allow_html=True)

if run_eval:
    with st.spinner("Calculating evaluation results…"):
        report_md, auto_scores, llm_scores = _run_evaluation()

        if st.session_state.eval_auto and st.session_state.eval_llm:
            st.session_state.eval_prev_auto = st.session_state.eval_auto.copy()
            st.session_state.eval_prev_llm = st.session_state.eval_llm.copy()

        st.session_state.eval_result = report_md
        st.session_state.eval_auto = auto_scores
        st.session_state.eval_llm = llm_scores
        st.session_state.eval_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if st.session_state.eval_result:
    if st.session_state.eval_auto and st.session_state.eval_llm:
        st.caption(f"Last evaluation time: {st.session_state.eval_at}")

        auto = st.session_state.eval_auto
        llm = st.session_state.eval_llm
        prev_auto = st.session_state.eval_prev_auto
        prev_llm = st.session_state.eval_prev_llm

        st.markdown("**Automatic Metrics**")
        a1, a2, a3 = st.columns(3)
        a1.metric(
            "Distinct-1",
            f"{auto.get('distinct_1', 0):.4f}",
            _delta_str(auto.get("distinct_1", 0), prev_auto, "distinct_1"),
        )
        a2.metric(
            "Distinct-2",
            f"{auto.get('distinct_2', 0):.4f}",
            _delta_str(auto.get("distinct_2", 0), prev_auto, "distinct_2"),
        )
        a3.metric(
            "Distinct-3",
            f"{auto.get('distinct_3', 0):.4f}",
            _delta_str(auto.get("distinct_3", 0), prev_auto, "distinct_3"),
        )

        a4, a5, a6 = st.columns(3)
        a4.metric(
            "Self-BLEU",
            f"{auto.get('self_bleu', 0):.4f}",
            _delta_str(auto.get("self_bleu", 0), prev_auto, "self_bleu"),
        )
        a5.metric(
            "Entity Coverage",
            f"{auto.get('entity_coverage', 0):.2%}",
            _delta_pct(auto.get("entity_coverage", 0), prev_auto, "entity_coverage"),
        )
        a6.metric(
            "Consistency Rate",
            f"{auto.get('consistency_rate', 0):.2%}",
            _delta_pct(auto.get("consistency_rate", 0), prev_auto, "consistency_rate"),
        )

        st.markdown("**LLM Judge Dimension Scores**")
        j1, j2, j3 = st.columns(3)
        j1.metric(
            "Narrative",
            llm.get("narrative_quality", 0),
            _delta_str(llm.get("narrative_quality", 0), prev_llm, "narrative_quality", ".2f"),
        )
        j2.metric(
            "Consistency",
            llm.get("consistency", 0),
            _delta_str(llm.get("consistency", 0), prev_llm, "consistency", ".2f"),
        )
        j3.metric(
            "Agency",
            llm.get("player_agency", 0),
            _delta_str(llm.get("player_agency", 0), prev_llm, "player_agency", ".2f"),
        )

        j4, j5, j6 = st.columns(3)
        j4.metric(
            "Creativity",
            llm.get("creativity", 0),
            _delta_str(llm.get("creativity", 0), prev_llm, "creativity", ".2f"),
        )
        j5.metric(
            "Pacing",
            llm.get("pacing", 0),
            _delta_str(llm.get("pacing", 0), prev_llm, "pacing", ".2f"),
        )
        j6.metric(
            "Average",
            llm.get("average", 0),
            _delta_str(llm.get("average", 0), prev_llm, "average", ".2f"),
        )

        with st.expander("View Raw Evaluation Report (Markdown)", expanded=False):
            st.markdown(st.session_state.eval_result)
    else:
        st.markdown(st.session_state.eval_result)
