"""StoryWeaver – Streamlit chat-based text adventure UI.

Layout (Streamlit):
  Sidebar:    KG visualisation · consistency scores · NLU debug · evaluation
  Main area:  chat history · option buttons · free-text chat input
"""
from __future__ import annotations

import os
import sys
import time
import logging

import streamlit as st
import streamlit.components.v1 as components

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.game_engine import GameEngine, TurnResult
from src.nlg.option_generator import StoryOption
from src.evaluation.metrics import full_evaluation
from src.evaluation.llm_judge import judge as llm_judge

logger = logging.getLogger(__name__)


# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StoryWeaver – AI Text Adventure",
    page_icon="🎭",
    layout="wide",
)

st.title("🎭 StoryWeaver: AI-Powered Text Adventure")
st.markdown(
    "*Hybrid NLU + LLM story engine with live knowledge-graph tracking* "
    "| Multi-turn generation + Real-time KG"
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
    "last_elapsed": 0.0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ──────────────────────────────────────────────────────────────

def _process_action(action: str) -> None:
    """Run a player action through the engine and update session state."""
    engine: GameEngine | None = st.session_state.engine
    if engine is None:
        st.warning("Please start a new game first.")
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


def _run_evaluation() -> str:
    """Collect session data from the engine and run all evaluations."""
    engine: GameEngine | None = st.session_state.engine
    if engine is None or not engine.all_story_texts:
        return "*No story content to evaluate yet. Start a game first!*"

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

    # Format as Markdown table
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
    return "\n".join(lines)


# ── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    # ── KG Visualisation ──
    st.header("📊 Story World KG")
    if st.session_state.kg_html:
        components.html(st.session_state.kg_html, height=480, scrolling=True)
    else:
        st.info("Start a game to see the Knowledge Graph")

    # ── Consistency Scores ──
    st.header("📈 Consistency Scores")
    if st.session_state.consistency_history:
        recent = st.session_state.consistency_history[-5:]
        offset = max(0, len(st.session_state.consistency_history) - 5)
        for i, score in enumerate(recent):
            st.progress(
                max(0.0, min(1.0, score)),
                text=f"Turn {offset + i + 1}: {score:.2f}",
            )
    else:
        st.caption("Scores will appear after your first action")

    # ── NLU Debug ──
    with st.expander("🔍 NLU Debug", expanded=False):
        nlu = st.session_state.nlu_debug
        if nlu:
            st.markdown(f"**Resolved input:** {nlu.get('resolved_input', '')}")
            st.markdown(
                f"**Intent:** {nlu.get('intent', '?')}  "
                f"(conf {nlu.get('confidence', 0):.2f})"
            )
            st.markdown(f"**Entities:** {nlu.get('entities', [])}")
        else:
            st.caption("NLU info will appear after your first action")

    st.markdown("---")

    # ── Download ──
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

    st.markdown("---")

    # ── Evaluation ──
    st.header("📊 Session Evaluation")
    if st.button("Evaluate Session", use_container_width=True):
        with st.spinner("Running evaluation (metrics + LLM judge)…"):
            st.session_state.eval_result = _run_evaluation()
    if st.session_state.eval_result:
        st.markdown(st.session_state.eval_result)


# ── Main area – Game controls ────────────────────────────────────────────

col_genre, col_btn = st.columns([3, 1])
with col_genre:
    genre = st.text_input(
        "Genre",
        value="fantasy",
        placeholder="Genre (e.g. fantasy, sci-fi, mystery)",
        label_visibility="collapsed",
    )
with col_btn:
    new_game_clicked = st.button(
        "🎮 New Game", type="primary", use_container_width=True
    )

if new_game_clicked:
    with st.spinner("Initialising a new adventure…"):
        engine = GameEngine(genre=genre or "fantasy")
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
        st.session_state.last_elapsed = 0.0
    st.rerun()


# ── Chat history ─────────────────────────────────────────────────────────

for msg in st.session_state.history:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])


# ── Option buttons ───────────────────────────────────────────────────────

if st.session_state.options:
    st.markdown("**Choose an option or type your own action below:**")
    opt_cols = st.columns(len(st.session_state.options))
    for idx, opt in enumerate(st.session_state.options):
        with opt_cols[idx]:
            # Include history length in key to avoid stale-button collisions
            btn_key = f"opt_{idx}_{len(st.session_state.history)}"
            if st.button(
                f"{idx + 1}. {opt.text}",
                key=btn_key,
                use_container_width=True,
            ):
                _process_action(opt.text)
                st.rerun()


# ── Chat input ───────────────────────────────────────────────────────────

user_input = st.chat_input("Or type a free-form action…")
if user_input:
    _process_action(user_input)
    st.rerun()


# ── Performance footer ──────────────────────────────────────────────────

if st.session_state.last_elapsed > 0:
    st.caption(f"✅ Last turn completed in {st.session_state.last_elapsed:.2f}s")
