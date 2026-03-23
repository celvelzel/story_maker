"""Evaluation section component for StoryWeaver.

StoryWeaver 评估区域组件。
提供评估仪表板的渲染函数，包括自动指标和LLM评分。
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from src.engine.game_engine import GameEngine
from src.evaluation.metrics import full_evaluation
from src.evaluation.llm_judge import judge as llm_judge


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


def render_evaluation_section(persist_callback) -> None:
    """Render the evaluation dashboard section.
    
    渲染评估仪表板区域。
    
    Args:
        persist_callback: Function to call to persist session state
    """
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
            persist_callback()

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
