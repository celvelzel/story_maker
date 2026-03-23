"""Chat section component for StoryWeaver.

StoryWeaver 聊天区域组件。
提供聊天历史和选项按钮的渲染函数。
"""
from __future__ import annotations

import streamlit as st


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


def render_chat_section() -> None:
    """Render the chat history section.
    
    渲染聊天历史区域。
    """
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


def render_option_buttons(action_callback) -> None:
    """Render option buttons section.
    
    渲染选项按钮区域。
    
    Args:
        action_callback: Function to call when an option is selected
    """
    if st.session_state.options:
        st.markdown("<div class='section-title'>&#x1F9ED; Branch Options</div>", unsafe_allow_html=True)
        st.caption("You can click an option directly, or type a free-form action below.")
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
                    use_container_width=True,
                ):
                    action_callback(opt.text)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
