"""StoryWeaver – Streamlit chat-based text adventure UI.

StoryWeaver Streamlit 前端界面。
提供赛博朋克风格的交互式故事生成界面。

布局（Streamlit）:
    侧边栏:    知识图谱可视化 · 一致性趋势 · 调试信息 · 下载功能
    主区域:    控制面板 · 故事聊天 · 选项按钮 · 评估仪表板
"""
from __future__ import annotations

import os
import sys
import signal
import atexit
import logging
from pathlib import Path

import streamlit as st

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.runtime_session import (
    load_runtime_session,
    remove_runtime_files,
    runtime_engine_path,
)
from src.ui.style_injector import inject_styles
from src.ui.session_contract import initialize_session, validate_session_state
from src.ui.layout.sidebar_view import render_sidebar
from src.ui.layout.main_view import render_main_area, _persist_runtime_session
from src.ui.sections.evaluation_section import render_evaluation_section
from config import settings

logger = logging.getLogger(__name__)

_RUNTIME_CLEANUP_REGISTERED = False
_RUNTIME_CLEANED = False


# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StoryWeaver - AI Story Generator",
    page_icon="🎭",
    layout="wide",
)

# Lock UI to dark mode to avoid readability issues from theme switching.
st.session_state.ui_mode = "dark"

# Inject centralized theme styles (replaces inline CSS)
inject_styles(st.session_state.ui_mode)

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

# Initialize session state using centralized contract
initialize_session()

# Override defaults with settings values
st.session_state.intent_model_path = str(settings.INTENT_MODEL_PATH)
st.session_state.kg_conflict_resolution = settings.KG_CONFLICT_RESOLUTION
st.session_state.kg_extraction_mode = settings.KG_EXTRACTION_MODE
st.session_state.kg_importance_mode = settings.KG_IMPORTANCE_MODE
st.session_state.kg_summary_mode = settings.KG_SUMMARY_MODE

# Validate session state integrity
_session_issues = validate_session_state()
if _session_issues:
    logger.warning(f"Session state issues detected: {_session_issues}")


# ── Runtime session management ───────────────────────────────────────────

def _runtime_save_dir() -> Path:
    return Path(settings.KG_SAVE_DIR)


def _restore_runtime_session_once() -> None:
    """Restore runtime session from disk if available.
    
    从磁盘恢复运行时会话（如果可用）。
    """
    if st.session_state.engine is not None:
        return

    data = load_runtime_session(_runtime_save_dir())
    if not data:
        return

    engine_file = str(data.get("engine_file", "")).strip()
    if not engine_file or not Path(engine_file).exists():
        return

    from src.engine.game_engine import GameEngine
    from src.engine.runtime_session import deserialize_options

    intent_model_path = str(data.get("intent_model_path", "")).strip() or None
    engine = GameEngine(
        genre=str(data.get("genre", "fantasy")) or "fantasy",
        intent_model_path=intent_model_path,
        conflict_resolution=str(data.get("kg_conflict_resolution", settings.KG_CONFLICT_RESOLUTION)),
        extraction_mode=str(data.get("kg_extraction_mode", settings.KG_EXTRACTION_MODE)),
        importance_mode=str(data.get("kg_importance_mode", settings.KG_IMPORTANCE_MODE)),
        summary_mode=str(data.get("kg_summary_mode", settings.KG_SUMMARY_MODE)),
    )
    engine.load_game(engine_file)

    st.session_state.engine = engine
    st.session_state.history = data.get("history", []) if isinstance(data.get("history"), list) else []
    st.session_state.consistency_history = (
        data.get("consistency_history", [])
        if isinstance(data.get("consistency_history"), list)
        else []
    )
    st.session_state.kg_html = str(data.get("kg_html", ""))
    st.session_state.options = deserialize_options(data.get("options", []))
    st.session_state.nlu_debug = data.get("nlu_debug", {}) if isinstance(data.get("nlu_debug"), dict) else {}
    st.session_state.chat_fold_mode = bool(data.get("chat_fold_mode", False))
    st.session_state.last_elapsed = float(data.get("last_elapsed", 0.0))
    st.session_state.intent_model_path = str(data.get("intent_model_path", st.session_state.intent_model_path))
    st.session_state.kg_conflict_resolution = str(
        data.get("kg_conflict_resolution", st.session_state.kg_conflict_resolution)
    )
    st.session_state.kg_extraction_mode = str(data.get("kg_extraction_mode", st.session_state.kg_extraction_mode))
    st.session_state.kg_importance_mode = str(data.get("kg_importance_mode", st.session_state.kg_importance_mode))
    st.session_state.kg_summary_mode = str(data.get("kg_summary_mode", st.session_state.kg_summary_mode))
    st.session_state.eval_result = str(data.get("eval_result", ""))
    st.session_state.eval_auto = data.get("eval_auto", {}) if isinstance(data.get("eval_auto"), dict) else {}
    st.session_state.eval_llm = data.get("eval_llm", {}) if isinstance(data.get("eval_llm"), dict) else {}
    st.session_state.eval_prev_auto = (
        data.get("eval_prev_auto", {}) if isinstance(data.get("eval_prev_auto"), dict) else {}
    )
    st.session_state.eval_prev_llm = (
        data.get("eval_prev_llm", {}) if isinstance(data.get("eval_prev_llm"), dict) else {}
    )
    st.session_state.eval_at = str(data.get("eval_at", ""))


def _cleanup_runtime_files() -> None:
    global _RUNTIME_CLEANED
    if _RUNTIME_CLEANED:
        return
    try:
        remove_runtime_files(_runtime_save_dir())
    finally:
        _RUNTIME_CLEANED = True


def _register_runtime_cleanup() -> None:
    global _RUNTIME_CLEANUP_REGISTERED
    if _RUNTIME_CLEANUP_REGISTERED or getattr(signal, "_storyweaver_runtime_cleanup_registered", False):
        return
    _RUNTIME_CLEANUP_REGISTERED = True
    setattr(signal, "_storyweaver_runtime_cleanup_registered", True)
    atexit.register(_cleanup_runtime_files)

    previous_handler = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):
        _cleanup_runtime_files()
        if callable(previous_handler):
            previous_handler(signum, frame)
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)


_register_runtime_cleanup()
_restore_runtime_session_once()


# ── Render UI ────────────────────────────────────────────────────────────

# Render sidebar
render_sidebar()

# Render main area
render_main_area()

# Render evaluation section
render_evaluation_section(_persist_runtime_session)
