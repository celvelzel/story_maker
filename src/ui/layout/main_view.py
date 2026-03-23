"""Main view component for StoryWeaver.

StoryWeaver 主区域视图组件。
提供主区域的渲染函数，包括游戏控制、聊天历史、选项按钮、聊天输入等。
"""
from __future__ import annotations

import time

import streamlit as st

from src.engine.game_engine import GameEngine, TurnResult
from src.engine.runtime_session import save_runtime_session, runtime_engine_path, serialize_options
from src.ui.feedback import show_info, show_warning
from config import settings
from pathlib import Path


def _runtime_save_dir() -> Path:
    """Get runtime save directory.
    
    获取运行时保存目录。
    """
    return Path(settings.KG_SAVE_DIR)


def _persist_runtime_session() -> None:
    """Persist current session to disk.
    
    将当前会话持久化到磁盘。
    """
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


def _process_action(action: str) -> None:
    """Run a player action through the engine and update session state.
    
    处理玩家行动：将玩家输入传递给游戏引擎，更新会话状态。
    包括：处理回合、更新聊天历史、更新知识图谱、跟踪一致性。
    """
    engine: GameEngine | None = st.session_state.engine
    if engine is None:
        show_warning(
            "Please start a new game before entering an action.",
            action_hint="Click 'Start New Game' above to begin your adventure",
        )
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
    _persist_runtime_session()


def _build_turn_pairs(history: list[dict]) -> tuple[str, list[tuple[int, str, str]]]:
    """Return opening narration and user-assistant turn pairs.
    
    构建回合配对：从聊天历史中提取开场白和用户-助手对话配对。
    返回 (开场白, [(回合号, 用户输入, 助手回复), ...])
    """
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


def render_main_area() -> None:
    """Render the complete main area.
    
    渲染完整的主区域。
    包括：游戏控制、聊天历史、选项按钮、聊天输入、性能页脚。
    """
    _render_game_controls()
    _render_chat_history()
    _render_option_buttons()
    _render_chat_input()
    _render_performance_footer()


def _render_game_controls() -> None:
    """Render game control section.
    
    渲染游戏控制区域。
    """
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
            intent_model_path = str(st.session_state.intent_model_path or "").strip() or None
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
            _persist_runtime_session()
        st.rerun()

    if st.session_state.engine is None:
        show_info("Click \"Start New Game\" above to begin the interactive story.")


def _render_chat_history() -> None:
    """Render chat history section.
    
    渲染聊天历史区域。
    """
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


def _render_option_buttons() -> None:
    """Render option buttons section.
    
    渲染选项按钮区域。
    """
    if st.session_state.options:
        st.markdown("<div class='section-title'>&#x1F9ED; Branch Options</div>", unsafe_allow_html=True)
        st.caption("You can click an option directly, or type a free-form action below.")
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
                    use_container_width=True,
                ):
                    _process_action(opt.text)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)


def _render_chat_input() -> None:
    """Render chat input section.
    
    渲染聊天输入区域。
    """
    user_input = st.chat_input("Enter your action (e.g., investigate the runes in the ruins)…")
    if user_input:
        _process_action(user_input)
        st.rerun()


def _render_performance_footer() -> None:
    """Render performance footer section.
    
    渲染性能页脚区域。
    """
    if st.session_state.last_elapsed > 0:
        st.caption(f"✅ Generation time for this turn: {st.session_state.last_elapsed:.2f}s")
