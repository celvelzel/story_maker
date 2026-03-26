"""UI Layout module for StoryWeaver."""

import streamlit as st
from .theme import load_theme


def load_layout() -> None:
    """Initialize all UI layout components (theme, hero banner)."""
    load_theme()
    
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


__all__ = ["load_layout", "load_theme"]
