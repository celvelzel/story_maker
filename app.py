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
import time
import signal
import atexit
import logging
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.game_engine import GameEngine, TurnResult
from src.engine.runtime_session import (
    remove_runtime_files,
    runtime_engine_path,
    save_runtime_session,
    serialize_options,
)
from src.nlg.option_generator import StoryOption
from src.evaluation.metrics import full_evaluation
from src.evaluation.llm_judge import judge as llm_judge
from src.knowledge_graph.visualizer import render_kg_html
from src.ui.layout import load_layout
from src.ui.sections.sidebar import render_sidebar
from src.ui.state_manager import initialize_state, restore_runtime_session
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
restore_runtime_session(_runtime_save_dir())


# ── Helpers ──────────────────────────────────────────────────────────────

def _process_action(action: str) -> None:
    """Run a player action through the engine and update session state.
    
    处理玩家行动：将玩家输入传递给游戏引擎，更新会话状态。
    包括：处理回合、更新聊天历史、更新知识图谱、跟踪一致性。
    """
    if st.session_state.processing:
        return
    engine: GameEngine | None = st.session_state.engine
    if engine is None:
        st.warning("Please start a new game before entering an action.")
        return

    st.session_state.processing = True
    start = time.time()
    st.session_state.history.append({"role": "user", "content": action})
    try:
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
    except Exception:
        logger.exception("Failed to process player action")
        st.error("Action processing failed. Please try again.")
        st.session_state.history.append(
            {
                "role": "assistant",
                "content": "⚠️ Failed to process this action due to an internal error. Please try again.",
            }
        )
    finally:
        st.session_state.last_elapsed = time.time() - start
        st.session_state.processing = False
        _persist_runtime_session()


def _run_evaluation() -> tuple[str, dict, dict]:
    """Collect session data from the engine and run all evaluations.
    
    收集会话数据并运行所有评估：
    1. 自动指标（Distinct-n、Self-BLEU、实体覆盖率、一致性率）
    2. LLM 评分（叙事质量、一致性、玩家代理、创意、节奏）
    """
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


def _delta_str(current: float, previous: dict, key: str, fmt: str = ".4f") -> str | None:
    """Format metric delta string for st.metric.
    
    格式化指标变化字符串：计算当前值与前值的差值。
    用于 Streamlit 的 st.metric 组件显示变化量。
    """
    if not previous or key not in previous:
        return None
    diff = float(current) - float(previous.get(key, 0))
    return f"{diff:+{fmt}}"


def _delta_pct(current: float, previous: dict, key: str) -> str | None:
    """Format percentage-point delta for st.metric.
    
    格式化百分点变化：计算当前值与前值的百分点差值。
    返回格式如 "+5.23pp"。
    """
    if not previous or key not in previous:
        return None
    diff = float(current) - float(previous.get(key, 0))
    return f"{diff * 100:+.2f}pp"


render_sidebar()


# ── Main area – Game controls ────────────────────────────────────────────

col_genre, col_btn = st.columns([3, 1])
with col_genre:
    genre = st.text_input(
        "Genre",
        key="genre_input",
        placeholder="Genre, e.g. fantasy / sci-fi / mystery",
        label_visibility="collapsed",
    )
with col_btn:
    new_game_clicked = st.button(
        "🎮 Start New Game", type="primary", width="stretch"
    )

if new_game_clicked:
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
        _persist_runtime_session()
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
                with st.spinner("Processing your action…"):
                    _process_action(opt.text)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


# ── Chat input ───────────────────────────────────────────────────────────

user_input = st.chat_input(
    "Enter your action (e.g., investigate the runes in the ruins)…",
    disabled=st.session_state.processing,
)
if user_input:
    with st.spinner("Processing your action…"):
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
    run_eval = st.button(
        "Run Evaluation",
        width="stretch",
        disabled=st.session_state.processing,
    )
with col_eval_hint:
    st.markdown("<div class='muted-note'>Evaluation computes automatic metrics from the current session and calls the LLM Judge for scoring.</div>", unsafe_allow_html=True)

if run_eval:
    with st.spinner("⏳ Calculating evaluation results… Please wait."):
        report_md, auto_scores, llm_scores = _run_evaluation()

        if st.session_state.eval_auto and st.session_state.eval_llm:
            st.session_state.eval_prev_auto = st.session_state.eval_auto.copy()
            st.session_state.eval_prev_llm = st.session_state.eval_llm.copy()

        st.session_state.eval_result = report_md
        st.session_state.eval_auto = auto_scores
        st.session_state.eval_llm = llm_scores
        st.session_state.eval_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _persist_runtime_session()

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
