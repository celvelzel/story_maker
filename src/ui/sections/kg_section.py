"""KG section component for StoryWeaver.

StoryWeaver 知识图谱区域组件。
提供知识图谱面板的渲染函数，包括可视化、状态显示、加载状态等。
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from src.ui.feedback import show_empty, show_loading, show_error


def render_kg_section() -> None:
    """Render the knowledge graph section.
    
    渲染知识图谱区域。
    包括：标题、状态、可视化、加载/错误/空状态。
    """
    st.markdown("<div class='section-title'>&#x1F4CA; Story World Knowledge Graph</div>", unsafe_allow_html=True)
    
    # Get KG state
    kg_html = st.session_state.get("kg_html", "")
    engine = st.session_state.get("engine")
    
    # Render based on state
    if engine is None:
        # No game started - show empty state
        show_empty(
            "No knowledge graph yet",
            hint="Start a game to see the story world's entities and relationships",
        )
    elif not kg_html:
        # Game started but no KG data yet - show loading state
        show_loading("Building knowledge graph...")
    else:
        # KG data available - render visualization
        _render_kg_visualization(kg_html)
        _render_kg_status(engine)


def _render_kg_visualization(kg_html: str) -> None:
    """Render the KG visualization.
    
    渲染知识图谱可视化。
    """
    st.markdown("<div class='kg-frame'>", unsafe_allow_html=True)
    components.html(kg_html, height=480, scrolling=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_kg_status(engine) -> None:
    """Render KG status information.
    
    渲染知识图谱状态信息。
    """
    entity_count = len(engine.kg_entity_names) if hasattr(engine, 'kg_entity_names') else 0
    conflict_total = sum(engine.turn_conflict_counts) if hasattr(engine, 'turn_conflict_counts') else 0
    
    st.caption(f"Entities: {entity_count} | Conflicts: {conflict_total}")


def render_kg_section_compact() -> None:
    """Render a compact version of the KG section for sidebar.
    
    渲染紧凑版知识图谱区域（用于侧边栏）。
    """
    st.markdown("<div class='section-title'>&#x1F4CA; Knowledge Graph</div>", unsafe_allow_html=True)
    
    kg_html = st.session_state.get("kg_html", "")
    engine = st.session_state.get("engine")
    
    if engine is None:
        show_empty(
            "No graph yet",
            hint="Start a game to populate",
        )
    elif not kg_html:
        show_loading("Building graph...")
    else:
        st.markdown("<div class='kg-frame'>", unsafe_allow_html=True)
        components.html(kg_html, height=300, scrolling=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Compact status
        entity_count = len(engine.kg_entity_names) if hasattr(engine, 'kg_entity_names') else 0
        st.caption(f"{entity_count} entities")
