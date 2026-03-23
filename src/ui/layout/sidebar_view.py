"""Sidebar view component for StoryWeaver.

StoryWeaver 侧边栏视图组件。
提供侧边栏的渲染函数，包括NLU配置、KG设置、知识图谱可视化等。
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from src.engine.game_engine import GameEngine
from src.ui.feedback import show_empty


def _story_turn_count() -> int:
    """Return number of user turns in current chat history.
    
    返回当前聊天历史中用户回合的数量。
    """
    return sum(1 for m in st.session_state.history if m["role"] == "user")


def render_sidebar() -> None:
    """Render the complete sidebar.
    
    渲染完整的侧边栏。
    包括：NLU模型配置、KG策略设置、知识图谱可视化、一致性趋势、NLU解析详情、统计信息、下载功能。
    """
    with st.sidebar:
        _render_nlu_config()
        _render_kg_settings()
        _render_knowledge_graph()
        _render_consistency_trend()
        _render_nlu_debug()
        _render_statistics()
        _render_download()


def _render_nlu_config() -> None:
    """Render NLU model configuration section.
    
    渲染NLU模型配置区域。
    """
    with st.expander("🧠 NLU Model Path", expanded=False):
        st.session_state.intent_model_path = st.text_input(
            "Intent model directory",
            value=st.session_state.intent_model_path,
            help="Leave empty to use the default directory; if it doesn't exist, it falls back to rule_fallback.",
        )


def _render_kg_settings() -> None:
    """Render KG strategy settings section.
    
    渲染KG策略设置区域。
    """
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


def _render_knowledge_graph() -> None:
    """Render knowledge graph visualization section.
    
    渲染知识图谱可视化区域。
    """
    st.markdown("<div class='section-title'>&#x1F4CA; Story World Knowledge Graph</div>", unsafe_allow_html=True)
    if st.session_state.kg_html:
        st.markdown("<div class='kg-frame'>", unsafe_allow_html=True)
        components.html(st.session_state.kg_html, height=480, scrolling=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        show_empty(
            "No knowledge graph yet",
            hint="Start a game to see the story world's entities and relationships",
        )


def _render_consistency_trend() -> None:
    """Render consistency trend section.
    
    渲染一致性趋势区域。
    """
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


def _render_nlu_debug() -> None:
    """Render NLU parsing details section.
    
    渲染NLU解析详情区域。
    """
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
            stage_metrics = nlu.get("stage_metrics", {})
            if isinstance(stage_metrics, dict) and stage_metrics:
                st.markdown("**Stage Metrics (ms/tokens/cost):**")
                for stage_name, metric in stage_metrics.items():
                    if not isinstance(metric, dict):
                        continue
                    elapsed = float(metric.get("elapsed_ms", 0.0))
                    in_tok = float(metric.get("input_tokens", 0.0))
                    out_tok = float(metric.get("output_tokens", 0.0))
                    cost = float(metric.get("cost_usd", 0.0))
                    st.markdown(
                        f"- `{stage_name}`: {elapsed:.1f} ms, "
                        f"in={in_tok:.0f}, out={out_tok:.0f}, cost=${cost:.6f}"
                    )
        else:
            st.caption("NLU parsing details will appear after each action")


def _render_statistics() -> None:
    """Render statistics section.
    
    渲染统计信息区域。
    """
    st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)

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


def _render_download() -> None:
    """Render download section.
    
    渲染下载区域。
    """
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
