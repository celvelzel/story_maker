"""Style injector for StoryWeaver cyberpunk theme.

StoryWeaver 赛博朋克主题样式注入器。
集中管理所有 CSS 样式，通过 st.markdown 注入。

Usage:
    from src.ui.style_injector import inject_styles
    
    # Inject all styles (call once at app startup)
    inject_styles("dark")
"""
from __future__ import annotations

import streamlit as st

from src.ui.theme_tokens import ThemeMode, get_tokens


def _build_css(tokens: dict[str, str]) -> str:
    """Build complete CSS string from theme tokens.
    
    从主题令牌构建完整的 CSS 字符串。
    
    Args:
        tokens: Theme token dictionary
        
    Returns:
        Complete CSS string ready for injection
    """
    return f"""
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

    /* ── Reduced Motion Guard ──────────────────────────────── */
    @media (prefers-reduced-motion: reduce) {{
        *,
        *::before,
        *::after {{
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }}
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
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: {tokens['neon_cyan']};
        margin: 18px 0 10px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid {tokens['neon_cyan']}22;
    }}

    /* ── Neon Divider ─────────────────────────────────────── */
    .neon-divider {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, {tokens['neon_cyan']}44, {tokens['neon_purple']}44, transparent);
        margin: 16px 0;
    }}

    /* ── KG Frame ─────────────────────────────────────────── */
    .kg-frame {{
        border: 1px solid {tokens['neon_cyan']}33;
        border-radius: 12px;
        overflow: hidden;
        background: rgba(0, 0, 0, 0.3);
        box-shadow: 0 0 20px {tokens['neon_cyan']}15, inset 0 0 30px rgba(0, 0, 0, 0.5);
    }}

    /* ── Chat Container ───────────────────────────────────── */
    .chat-container {{
        background: {tokens['chat_bg']};
        border: 1px solid {tokens['neon_cyan']}22;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }}

    /* ── Option Buttons ───────────────────────────────────── */
    .stButton > button {{
        background: linear-gradient(135deg, rgba(0, 240, 255, 0.08), rgba(123, 47, 255, 0.06));
        border: 1px solid {tokens['neon_cyan']}44;
        color: {tokens['text']};
        border-radius: 8px;
        padding: 10px 16px;
        font-family: 'Rajdhani', 'Noto Sans SC', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
        width: 100%;
        text-align: left;
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, rgba(0, 240, 255, 0.15), rgba(123, 47, 255, 0.12));
        border-color: {tokens['neon_cyan']};
        box-shadow: 0 0 12px {tokens['neon_cyan']}40;
        transform: translateY(-1px);
    }}

    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* ── Text Input ───────────────────────────────────────── */
    .stTextInput > div > div > input {{
        background: {tokens['input_bg']};
        border: 1px solid {tokens['input_border']};
        color: {tokens['text']};
        border-radius: 8px;
        font-family: 'Rajdhani', 'Noto Sans SC', sans-serif;
    }}

    .stTextInput > div > div > input:focus {{
        border-color: {tokens['neon_cyan']};
        box-shadow: 0 0 8px {tokens['neon_cyan']}40;
    }}

    /* ── Chat Input ───────────────────────────────────────── */
    .stChatInput > div {{
        background: {tokens['input_bg']};
        border: 1px solid {tokens['input_border']};
        border-radius: 12px;
    }}

    .stChatInput > div:focus-within {{
        border-color: {tokens['neon_cyan']};
        box-shadow: 0 0 12px {tokens['neon_cyan']}30;
    }}

    /* ── Sidebar ──────────────────────────────────────────── */
    .stSidebar {{
        background: {tokens['panel']};
        border-right: 1px solid {tokens['panel_border']};
    }}

    .stSidebar .stMarkdown {{
        color: {tokens['text']};
    }}

    /* ── Metrics ──────────────────────────────────────────── */
    .stMetric {{
        background: rgba(0, 240, 255, 0.03);
        border: 1px solid {tokens['neon_cyan']}22;
        border-radius: 8px;
        padding: 12px;
    }}

    .stMetric label {{
        color: {tokens['muted']};
        font-size: 0.75rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }}

    .stMetric value {{
        color: {tokens['neon_cyan']};
        font-family: 'Orbitron', monospace;
    }}

    /* ── Expanders ────────────────────────────────────────── */
    .stExpander {{
        border: 1px solid {tokens['panel_border']};
        border-radius: 8px;
        background: rgba(0, 0, 0, 0.2);
    }}

    .stExpander summary {{
        color: {tokens['text']};
        font-weight: 600;
    }}

    /* ── Progress Bars ────────────────────────────────────── */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {tokens['neon_cyan']}, {tokens['neon_purple']});
    }}

    /* ── Alerts / Info Boxes ──────────────────────────────── */
    .stAlert {{
        border-radius: 8px;
        border: 1px solid {tokens['panel_border']};
    }}

    .stAlert[data-baseweb="notification"][kind="info"] {{
        background: rgba(0, 240, 255, 0.08);
        border-color: {tokens['neon_cyan']}44;
    }}

    .stAlert[data-baseweb="notification"][kind="warning"] {{
        background: rgba(255, 215, 0, 0.08);
        border-color: {tokens['neon_gold']}44;
    }}

    .stAlert[data-baseweb="notification"][kind="error"] {{
        background: rgba(255, 68, 68, 0.08);
        border-color: {tokens['error']}44;
    }}

    .stAlert[data-baseweb="notification"][kind="success"] {{
        background: rgba(0, 255, 136, 0.08);
        border-color: {tokens['success']}44;
    }}

    /* ── Selectbox / Dropdown ─────────────────────────────── */
    .stSelectbox > div > div {{
        background: {tokens['input_bg']};
        border: 1px solid {tokens['input_border']};
        border-radius: 8px;
    }}

    .stSelectbox > div > div:focus-within {{
        border-color: {tokens['neon_cyan']};
        box-shadow: 0 0 8px {tokens['neon_cyan']}40;
    }}

    /* ── Download Button ──────────────────────────────────── */
    .stDownloadButton > button {{
        background: linear-gradient(135deg, rgba(0, 240, 255, 0.1), rgba(123, 47, 255, 0.08));
        border: 1px solid {tokens['neon_cyan']}44;
        color: {tokens['text']};
        border-radius: 8px;
        font-family: 'Rajdhani', 'Noto Sans SC', sans-serif;
        transition: all 0.2s ease;
    }}

    .stDownloadButton > button:hover {{
        border-color: {tokens['neon_cyan']};
        box-shadow: 0 0 12px {tokens['neon_cyan']}40;
    }}

    /* ── Focus Outlines (Accessibility) ───────────────────── */
    *:focus-visible {{
        outline: 2px solid {tokens['neon_cyan']};
        outline-offset: 2px;
    }}

    /* ── Chat Messages ────────────────────────────────────── */
    .stChatMessage {{
        background: rgba(0, 240, 255, 0.02);
        border: 1px solid {tokens['panel_border']};
        border-radius: 12px;
        margin-bottom: 8px;
    }}

    /* ── Columns Gap ──────────────────────────────────────── */
    .stColumn {{
        padding: 0 8px;
    }}

    /* ── Spinner ──────────────────────────────────────────── */
    .stSpinner > div {{
        border-color: {tokens['neon_cyan']} transparent transparent transparent;
    }}

    /* ── Dialog / Modal ───────────────────────────────────── */
    [role="dialog"],
    dialog,
    .stModal {{
        background: {tokens['panel']};
        border: 1px solid {tokens['panel_border']};
        border-radius: 16px;
        box-shadow: 0 0 20px {tokens['neon_cyan']}44, 0 0 40px {tokens['neon_purple']}22 !important;
    }}

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
"""


def inject_styles(mode: ThemeMode = "dark") -> None:
    """Inject all theme styles into the Streamlit app.
    
    将所有主题样式注入 Streamlit 应用。
    应在应用启动时调用一次。
    
    Args:
        mode: Theme mode ("dark" or "light")
    """
    tokens = get_tokens(mode)
    css = _build_css(tokens)
    st.markdown(css, unsafe_allow_html=True)


def get_css(mode: ThemeMode = "dark") -> str:
    """Get the complete CSS string without injecting.
    
    获取完整的 CSS 字符串（不注入）。
    用于测试或自定义注入场景。
    
    Args:
        mode: Theme mode
        
    Returns:
        Complete CSS string
    """
    tokens = get_tokens(mode)
    return _build_css(tokens)
