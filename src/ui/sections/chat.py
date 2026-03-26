"""Chat history and input rendering for StoryWeaver."""

from __future__ import annotations

from typing import Any

import streamlit as st

__all__ = ["render_chat_history", "render_chat_input"]

# 分页配置
_CHAT_PAGE_SIZE = 10  # 每页显示的消息数


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
    """Display the chat message history with pagination."""
    # 优化：确保 session_state 中有默认值，避免首次渲染时状态不一致
    if "chat_fold_mode" not in st.session_state:
        st.session_state.chat_fold_mode = False
    # 初始化分页状态
    if "chat_page" not in st.session_state:
        st.session_state.chat_page = 0
    
    def _on_toggle_change():
        """Toggle 状态变更回调"""
        st.session_state.chat_fold_mode = st.session_state._chat_fold_toggle
        # 切换模式时重置到第一页
        st.session_state.chat_page = 0
    
    # 使用 key 来确保状态一致性
    chat_fold_mode = st.toggle(
        "Fold history by turn",
        value=st.session_state.chat_fold_mode,
        key="_chat_fold_toggle",
        help='When enabled, each turn is folded as "user input + system response" for easier long-session reading.',
        on_change=_on_toggle_change,
    )

    history = st.session_state.history
    total_messages = len(history)
    
    # 计算分页
    page_size = _CHAT_PAGE_SIZE
    total_pages = max(1, (total_messages + page_size - 1) // page_size)
    current_page = min(st.session_state.chat_page, total_pages - 1)
    
    # 分页导航控件（仅在消息较多时显示）
    if total_messages > page_size:
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("⬅️ Previous", key="chat_prev", disabled=(current_page == 0)):
                st.session_state.chat_page = current_page - 1
        with col_info:
            st.caption(f"📄 Page {current_page + 1} of {total_pages} ({total_messages} messages)")
        with col_next:
            if st.button("Next ➡️", key="chat_next", disabled=(current_page >= total_pages - 1)):
                st.session_state.chat_page = current_page + 1
    
    # 计算当前页的切片范围
    start_idx = current_page * page_size
    end_idx = min(start_idx + page_size, total_messages)
    page_history = history[start_idx:end_idx]

    if chat_fold_mode:
        # 对于折叠模式，需要显示完整的 turn pairs，但只显示当前页的
        opening, turn_pairs = _build_turn_pairs(history)
        if opening and current_page == 0:
            with st.chat_message("assistant"):
                st.markdown(opening)

        # 计算当前页对应的 turn pairs 范围
        page_turns = turn_pairs[start_idx:end_idx]
        for turn_idx, (turn, user_text, ai_text) in enumerate(page_turns):
            global_turn = start_idx + turn_idx + 1
            preview = user_text[:28] + ("..." if len(user_text) > 28 else "")
            title = f"Turn {global_turn} | {preview or '(empty input)'}"
            with st.expander(title, expanded=(global_turn == len(turn_pairs))):
                if user_text:
                    st.markdown("**You:**")
                    st.markdown(user_text)
                if ai_text:
                    st.markdown("**System:**")
                    st.markdown(ai_text)
    else:
        for msg in page_history:
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
