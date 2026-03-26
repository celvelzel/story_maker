"""Theme and CSS styling for StoryWeaver.

Contains cyberpunk neon color palette and comprehensive CSS styles.
"""

import streamlit as st


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


def load_theme() -> None:
    """Inject CSS theme into Streamlit app (cached per session to avoid re-injection on every rerun)."""
    # 缓存：同一 session 只注入一次 CSS，避免每次 rerun 重复处理 ~780 行样式
    _cache_key = "_theme_injected"
    if st.session_state.get(_cache_key):
        return

    tokens = _theme_tokens(st.session_state.ui_mode)

    st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;800;900&family=Rajdhani:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;700;800&display=swap');

    /* ── Keyframe Animations ─────────────────────────────── */
    /* 优化：减少无限循环动画的频率以降低 CPU 占用 */
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
            box-shadow: 0 0 8px {tokens['neon_cyan']}30, 0 0 16px {tokens['neon_cyan']}10;
        }}
        50% {{
            border-color: {tokens['neon_cyan']}aa;
            box-shadow: 0 0 12px {tokens['neon_cyan']}50, 0 0 24px {tokens['neon_purple']}20;
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

    /* Grid overlay on main area - 优化：静态网格，不使用动画以减少 CPU 占用 */
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
    /* 优化：移除无限动画，改为静态样式以减少 CPU 占用 */
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
    /* Neon breathing border — always visible - 优化：延长动画周期减少 CPU 占用 */
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

    # 标记 CSS 已注入，后续 rerun 跳过
    st.session_state[_cache_key] = True
