"""Chat history and input rendering for StoryWeaver."""

from __future__ import annotations

from typing import Any

import streamlit as st

__all__ = ["render_chat_history", "render_chat_input"]

def _build_turn_pairs(history: list[dict[str, Any]]) -> tuple[str, list[tuple[int, str, str]]]:
    """Return opening narration and user-assistant turn pairs."""
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


def render_chat_history() -> None:
    """Display the chat message history."""
    if "chat_fold_mode" not in st.session_state:
        st.session_state.chat_fold_mode = False

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


def render_chat_input() -> None:
    """Show the chat input field and submit action."""
    user_input = st.chat_input(
        "Enter your action (e.g., investigate the runes in the ruins)…",
        disabled=st.session_state.processing,
    )
    if user_input:
        action_handler = st.session_state.get("process_action")
        if callable(action_handler):
            action_handler(user_input)
