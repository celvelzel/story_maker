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
import streamlit.components.v1 as components

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.game_engine import GameEngine, TurnResult
from src.engine.runtime_session import (
    deserialize_options,
    load_runtime_session,
    remove_runtime_files,
    runtime_engine_path,
    save_runtime_session,
    serialize_options,
)
from src.nlg.option_generator import StoryOption
from src.knowledge_graph.visualizer import render_kg_html
from src.evaluation.metrics import full_evaluation
from src.evaluation.llm_judge import judge as llm_judge
from src.ui.layout import load_layout
from src.ui.state_manager import initialize_state, restore_runtime_session, _DEFAULTS
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


def _list_save_slots(save_dir: Path) -> list[dict[str, str]]:
    """Scan save directory and return sorted save-slot metadata for UI."""
    slots: list[dict[str, str]] = []
    if not save_dir.exists():
        return slots

    for file in save_dir.glob("*.json"):
        if file.name in _SAVE_META_EXCLUDE:
            continue

        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        state = data.get("state", {}) if isinstance(data.get("state"), dict) else {}
        try:
            turn_id = int(state.get("turn_id", 0))
        except (TypeError, ValueError):
            turn_id = 0
        genre = str(data.get("genre", state.get("genre", "unknown")))
        save_title = str(data.get("save_title", "")).strip() or "Untitled Snapshot"
        mtime = file.stat().st_mtime
        updated_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

        label = f"{save_title} | {genre} · Turn {turn_id} · {updated_at}"
        slots.append(
            {
                "label": label,
                "path": str(file),
                "filename": file.name,
                "mtime": str(mtime),
            }
        )

    slots.sort(key=lambda item: float(item.get("mtime", "0")), reverse=True)
    return slots


def _restore_from_save_file(filepath: str) -> tuple[bool, str]:
    """Load selected save file into current Streamlit session state."""
    path = Path(filepath)
    if not path.exists():
        return False, f"Save file not found: {path.name}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.exception("Failed to read save file: %s", path)
        return False, f"Failed to read save file: {exc}"

    if not isinstance(data, dict):
        return False, "Invalid save format."

    state_data = data.get("state", {}) if isinstance(data.get("state"), dict) else {}
    saved_genre = str(data.get("genre", state_data.get("genre", "fantasy"))) or "fantasy"

    try:
        intent_model_path_raw = st.session_state.intent_model_path or ""
        intent_model_path = intent_model_path_raw.strip() or None
        engine = GameEngine(
            genre=saved_genre,
            intent_model_path=intent_model_path,
            conflict_resolution=str(data.get("conflict_resolution", settings.KG_CONFLICT_RESOLUTION)),
            extraction_mode=str(data.get("extraction_mode", settings.KG_EXTRACTION_MODE)),
            importance_mode=str(data.get("importance_mode", settings.KG_IMPORTANCE_MODE)),
            summary_mode=str(data.get("summary_mode", settings.KG_SUMMARY_MODE)),
        )
        engine.load_game(str(path))

        restored_history: list[dict[str, str]] = []
        for item in engine.state.story_history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            if role == "player":
                restored_history.append({"role": "user", "content": text})
            elif role == "narrator":
                restored_history.append({"role": "assistant", "content": text})

        last_story = ""
        for item in reversed(engine.state.story_history):
            if isinstance(item, dict) and item.get("role") == "narrator":
                last_story = str(item.get("text", "")).strip()
                if last_story:
                    break

        options = engine.option_gen.generate(last_story, engine.kg.to_summary()) if last_story else []
        kg_html = render_kg_html(engine.kg.graph)

        st.session_state.engine = engine
        st.session_state.genre_input = saved_genre
        st.session_state.history = restored_history
        st.session_state.kg_html = kg_html
        st.session_state.options = options
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
        st.session_state.kg_conflict_resolution = engine.conflict_resolution
        st.session_state.kg_extraction_mode = engine.extraction_mode
        st.session_state.kg_importance_mode = engine.importance_mode
        st.session_state.kg_summary_mode = engine.summary_mode

        _persist_runtime_session()
    except Exception as exc:
        logger.exception("Failed to restore save file: %s", path)
        return False, f"Failed to load save: {exc}"

    return True, f"Loaded save: {path.name}"


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


def _story_turn_count() -> int:
    """Return number of user turns in current chat history.
    
    返回当前聊天历史中用户回合的数量。
    """
    return sum(1 for m in st.session_state.history if m["role"] == "user")


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


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='section-title'>&#x1F39B;&#xFE0F; Dashboard</div>", unsafe_allow_html=True)
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

    st.markdown("<div class='section-title'>&#x1F4CA; Story World Knowledge Graph</div>", unsafe_allow_html=True)
    if st.session_state.kg_html:
        st.markdown("<div class='kg-frame'>", unsafe_allow_html=True)
        components.html(st.session_state.kg_html, height=480, scrolling=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("The knowledge graph will appear after starting a game.")

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
    
    st.markdown("<div class='section-title'>&#x2699; Settings</div>", unsafe_allow_html=True)
    with st.expander("💾 Save / Load", expanded=True):
        save_slots = _list_save_slots(_runtime_save_dir())
        if save_slots:
            selected_idx = st.selectbox(
                "Choose a save slot",
                options=list(range(len(save_slots))),
                format_func=lambda i: save_slots[i]["label"],
                help="Snapshot title is generated by LLM during periodic turn snapshots.",
            )
            st.caption(f"Available saves: {len(save_slots)}")
            if st.button("📂 Load Selected Save", width="stretch", disabled=st.session_state.processing):
                with st.spinner("Loading selected save..."):
                    ok, message = _restore_from_save_file(save_slots[selected_idx]["path"])
                if ok:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        else:
            st.caption("No save files found yet. Play a few turns to generate snapshots.")

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
