"""StoryWeaver – Streamlit chat-based text adventure UI.

StoryWeaver Streamlit 前端界面。
提供赛博朋克风格的交互式故事生成界面。

布局（Streamlit）:
    侧边栏:    知识图谱可视化 · 一致性趋势 · 调试信息 · 下载功能
    主区域:    控制面板 · 故事聊天 · 选项按钮 · 评估仪表板
"""
# pyright: reportMissingImports=false
from __future__ import annotations

import os
import sys
import time
import signal
import atexit
import logging
from pathlib import Path

import streamlit as st

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.game_engine import GameEngine, TurnResult
from src.engine.runtime_session import (
    mark_session_inactive,
    remove_runtime_files,
    runtime_engine_path,
    runtime_session_path,
    save_runtime_session,
    serialize_options,
)
from src.ui.layout import load_layout
from src.ui.sections.evaluation import render_evaluation
from src.ui.sections.sidebar import render_sidebar
from src.ui.sections.chat import render_chat_history, render_chat_input
from src.ui.state_manager import (
    cleanup_session_state,
    initialize_state,
    is_runtime_session_active,
    restore_runtime_session,
)
from config import settings

logger = logging.getLogger(__name__)

_RUNTIME_CLEANUP_REGISTERED = False
_RUNTIME_CLEANED = False
_SAVE_META_EXCLUDE = {"runtime_session.json", "runtime_engine.json"}


# ── Page config ──────────────────────────────────────────────────────────

# Lock UI to dark mode to avoid readability issues from theme switching.
st.session_state.ui_mode = "dark"


# ── Load theme and layout ────────────────────────────────────────────────────

load_layout()


# ── Session State initialisation ─────────────────────────────────────────

initialize_state()
if "show_load_checkpoint_btn" not in st.session_state:
    st.session_state.show_load_checkpoint_btn = True
if "archive_filename" not in st.session_state:
    st.session_state.archive_filename = ""


def _runtime_save_dir() -> Path:
    return Path(settings.KG_SAVE_DIR)


def _persist_runtime_session() -> None:
    engine: GameEngine | None = st.session_state.engine
    if engine is None:
        return

    save_dir = _runtime_save_dir()
    engine_file = runtime_engine_path(save_dir)
    engine.save_game(str(engine_file))

    payload = {
        "version": 1,
        "genre": engine.genre,
        "history": st.session_state.history,
        "consistency_history": st.session_state.consistency_history,
        "kg_html": st.session_state.kg_html,
        "options": serialize_options(st.session_state.options),
        "nlu_debug": st.session_state.nlu_debug,
        "chat_fold_mode": st.session_state.chat_fold_mode,
        "last_elapsed": st.session_state.last_elapsed,
        "intent_model_path": st.session_state.intent_model_path,
        "kg_conflict_resolution": st.session_state.kg_conflict_resolution,
        "kg_extraction_mode": st.session_state.kg_extraction_mode,
        "kg_importance_mode": st.session_state.kg_importance_mode,
        "kg_summary_mode": st.session_state.kg_summary_mode,
        "eval_result": st.session_state.eval_result,
        "eval_auto": st.session_state.eval_auto,
        "eval_llm": st.session_state.eval_llm,
        "eval_prev_auto": st.session_state.eval_prev_auto,
        "eval_prev_llm": st.session_state.eval_prev_llm,
        "eval_at": st.session_state.eval_at,
        "engine_file": str(engine_file),
    }
    save_runtime_session(save_dir, payload)


def _cleanup_runtime_files() -> None:
    global _RUNTIME_CLEANED
    if _RUNTIME_CLEANED:
        return
    try:
        mark_session_inactive(_runtime_save_dir())
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

    # signal.signal() can only be called from the main thread.
    # Streamlit runs the script in a worker thread, so guard against that.
    import threading
    if threading.current_thread() is not threading.main_thread():
        return

    previous_handler = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):
        _cleanup_runtime_files()
        if callable(previous_handler):
            previous_handler(signum, frame)
        else:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)


_register_runtime_cleanup()
# 默认不自动恢复存档，只有点击"加载存档"按钮时才恢复
restore_runtime_session(_runtime_save_dir(), auto_restore=True)

_runtime_save_path = runtime_session_path(_runtime_save_dir())
st.session_state.show_load_checkpoint_btn = (
    st.session_state.engine is None
    and _runtime_save_path.exists()
    and not is_runtime_session_active(_runtime_save_dir())
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _process_action(action: str) -> None:
    """Run a player action through the engine and update session state.
    
    处理玩家行动：将玩家输入传递给游戏引擎，更新会话状态。
    包括：处理回合、更新聊天历史、更新知识图谱、跟踪一致性。
    """
    # 双重检查防止竞态条件
    if st.session_state.get("processing", False):
        return
    engine: GameEngine | None = st.session_state.engine
    if engine is None:
        st.warning("Please start a new game before entering an action.")
        return

    # 使用原子操作设置 processing 状态
    st.session_state.processing = True
    start = time.time()
    st.session_state.history.append({"role": "user", "content": action})
    
    try:
        with st.spinner("Processing your action..."):
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

        
    except Exception as e:
        logger.exception("Failed to process player action")
        st.error(f"Action processing failed: {e}")
        st.session_state.history.append(
            {
                "role": "assistant",
                "content": "⚠️ Failed to process this action due to an internal error. Please try again.",
            }
        )
    finally:
        # 确保 elapsed 和 processing 状态正确更新
        st.session_state.last_elapsed = time.time() - start
        st.session_state.processing = False
        _persist_runtime_session()

    st.rerun()


# ── Main area – Game controls ────────────────────────────────────────────

col_genre, col_btn = st.columns([2, 1.2])
with col_genre:
    genre = st.text_input(
        "Genre",
        value="fantasy",
        placeholder="Genre, e.g. fantasy / sci-fi / mystery",
        label_visibility="collapsed",
    )
with col_btn:
    new_game_clicked = st.button(
        "🎮 Start New Game", type="primary", width="stretch"
    )

load_checkpoint_clicked = False
if st.session_state.show_load_checkpoint_btn and st.session_state.engine is None:
    load_checkpoint_clicked = st.button("📦 Load Checkpoint", width="stretch")
    if load_checkpoint_clicked:
        restore_runtime_session(_runtime_save_dir(), auto_restore=False)
        st.session_state.show_load_checkpoint_btn = st.session_state.engine is None

if new_game_clicked:
    try:
        with st.spinner("Initializing the adventure world…"):
            intent_model_path_raw = st.session_state.intent_model_path or ""
            intent_model_path = intent_model_path_raw.strip() or None
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

            save_path = engine.save_game()
            archive_filename = Path(save_path).name
            st.session_state.archive_filename = archive_filename
            
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
            st.session_state.processing = False
            # 清理旧的 session state 数据
            cleanup_session_state()
            _persist_runtime_session()
    except Exception as e:
        st.error(f"Failed to start game: {e}")

if st.session_state.engine is None:
    st.info("Click \"Start New Game\" above to begin the interactive story.")
else:
    if st.session_state.get("archive_filename"):
        st.info(f"📁 Archive: `{st.session_state.archive_filename}`")


# ── Chat history ─────────────────────────────────────────────────────────

render_chat_history()


# ── Option buttons ───────────────────────────────────────────────────────

if st.session_state.options:
    st.markdown("<div class='section-title'>&#x1F9ED; Branch Options</div>", unsafe_allow_html=True)
    st.caption("You can click an option directly, or type a free-form action below.")
    _is_busy = st.session_state.processing
    
    opt_cols = st.columns(len(st.session_state.options))
    for idx, opt in enumerate(st.session_state.options):
        with opt_cols[idx]:
            st.markdown(
                f"<div class='option-meta-center'>Intent: {opt.intent_hint} | Risk: {opt.risk_level}</div>",
                unsafe_allow_html=True,
            )
            btn_key = f"opt_{idx}_{len(st.session_state.history)}"
            if st.button(
                f"{idx + 1}. {opt.text}",
                key=btn_key,
                width="stretch",
                disabled=_is_busy,
            ):
                _process_action(opt.text)
            st.markdown("</div>", unsafe_allow_html=True)


# ── Chat input ───────────────────────────────────────────────────────────

st.session_state.process_action = _process_action
render_chat_input()


# ── Performance footer ──────────────────────────────────────────────────

if st.session_state.last_elapsed > 0:
    st.caption(f"✅ Generation time for this turn: {st.session_state.last_elapsed:.2f}s")


render_evaluation()

render_sidebar()
