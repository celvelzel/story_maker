"""Evaluation dashboard rendering for StoryWeaver."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import settings
from src.evaluation.llm_judge import judge as llm_judge
from src.evaluation.metrics import full_evaluation
from src.engine.runtime_session import runtime_engine_path, save_runtime_session, serialize_options

logger = logging.getLogger(__name__)

__all__ = ["render_evaluation"]


def _persist_runtime_session() -> None:
    """保存运行时会话（与 app.py 中的版本保持一致）。
    
    注意：此函数与 app.py._persist_runtime_session 功能重复，
    但因 Streamlit 的模块导入限制无法直接复用。
    两处修改需同步维护。
    """
    engine = st.session_state.engine
    if engine is None:
        return

    save_dir = Path(settings.KG_SAVE_DIR)
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


def _run_evaluation() -> tuple[str, dict, dict]:
    engine = st.session_state.engine
    if engine is None or not engine.all_story_texts:
        return "*No evaluable content yet. Please start and progress the story first.*", {}, {}

    texts = engine.all_story_texts

    auto = full_evaluation(
        texts=texts,
        entity_names=engine.kg_entity_names,
        turn_conflict_counts=engine.turn_conflict_counts,
    )
    transcript = "\n".join(texts)
    llm_scores = llm_judge(transcript)

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


def _delta_str(current: float, previous: dict, key: str, fmt: str = ".4f") -> str | None:
    if not previous or key not in previous:
        return None
    diff = float(current) - float(previous.get(key, 0))
    return f"{diff:+{fmt}}"


def _delta_pct(current: float, previous: dict, key: str) -> str | None:
    if not previous or key not in previous:
        return None
    diff = float(current) - float(previous.get(key, 0))
    return f"{diff * 100:+.2f}pp"


def render_evaluation() -> None:
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
            a1.metric("Distinct-1", f"{auto.get('distinct_1', 0):.4f}", _delta_str(auto.get("distinct_1", 0), prev_auto, "distinct_1"))
            a2.metric("Distinct-2", f"{auto.get('distinct_2', 0):.4f}", _delta_str(auto.get("distinct_2", 0), prev_auto, "distinct_2"))
            a3.metric("Distinct-3", f"{auto.get('distinct_3', 0):.4f}", _delta_str(auto.get("distinct_3", 0), prev_auto, "distinct_3"))

            a4, a5, a6 = st.columns(3)
            a4.metric("Self-BLEU", f"{auto.get('self_bleu', 0):.4f}", _delta_str(auto.get("self_bleu", 0), prev_auto, "self_bleu"))
            a5.metric("Entity Coverage", f"{auto.get('entity_coverage', 0):.2%}", _delta_pct(auto.get("entity_coverage", 0), prev_auto, "entity_coverage"))
            a6.metric("Consistency Rate", f"{auto.get('consistency_rate', 0):.2%}", _delta_pct(auto.get("consistency_rate", 0), prev_auto, "consistency_rate"))

            st.markdown("**LLM Judge Dimension Scores**")
            j1, j2, j3 = st.columns(3)
            j1.metric("Narrative", llm.get("narrative_quality", 0), _delta_str(llm.get("narrative_quality", 0), prev_llm, "narrative_quality", ".2f"))
            j2.metric("Consistency", llm.get("consistency", 0), _delta_str(llm.get("consistency", 0), prev_llm, "consistency", ".2f"))
            j3.metric("Agency", llm.get("player_agency", 0), _delta_str(llm.get("player_agency", 0), prev_llm, "player_agency", ".2f"))

            j4, j5, j6 = st.columns(3)
            j4.metric("Creativity", llm.get("creativity", 0), _delta_str(llm.get("creativity", 0), prev_llm, "creativity", ".2f"))
            j5.metric("Pacing", llm.get("pacing", 0), _delta_str(llm.get("pacing", 0), prev_llm, "pacing", ".2f"))
            j6.metric("Average", llm.get("average", 0), _delta_str(llm.get("average", 0), prev_llm, "average", ".2f"))

            with st.expander("View Raw Evaluation Report (Markdown)", expanded=False):
                st.markdown(st.session_state.eval_result)
        else:
            st.markdown(st.session_state.eval_result)
