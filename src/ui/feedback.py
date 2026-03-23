"""Feedback state helpers for StoryWeaver UI.

StoryWeaver 反馈状态助手。
提供统一的加载/错误/成功/空状态渲染函数。

Usage:
    from src.ui.feedback import show_loading, show_error, show_success, show_empty
    
    show_loading("Generating story...")
    show_error("Failed to generate", retry_hint="Please try again")
    show_success("Story generated successfully!")
    show_empty("No entities yet", hint="Start a game to populate the knowledge graph")
"""
from __future__ import annotations

import streamlit as st


def show_loading(message: str = "Processing...") -> None:
    """Show a loading state with spinner.
    
    显示加载状态和旋转器。
    
    Args:
        message: Loading message to display
    """
    st.spinner(message)


def show_error(
    message: str,
    *,
    retry_hint: str | None = None,
    details: str | None = None,
) -> None:
    """Show an error state with optional retry guidance.
    
    显示错误状态和可选的重试指导。
    
    Args:
        message: Error message to display
        retry_hint: Optional hint for how to retry
        details: Optional technical details (shown in expander)
    """
    st.error(message)
    if retry_hint:
        st.caption(f"💡 {retry_hint}")
    if details:
        with st.expander("Technical Details"):
            st.code(details)


def show_success(message: str, *, duration: int | None = None) -> None:
    """Show a success state.
    
    显示成功状态。
    
    Args:
        message: Success message to display
        duration: Optional auto-dismiss duration in seconds (not supported in all Streamlit versions)
    """
    st.success(message)


def show_info(message: str) -> None:
    """Show an informational state.
    
    显示信息状态。
    """
    st.info(message)


def show_warning(message: str, *, action_hint: str | None = None) -> None:
    """Show a warning state with optional action guidance.
    
    显示警告状态和可选的操作指导。
    
    Args:
        message: Warning message to display
        action_hint: Optional hint for what to do
    """
    st.warning(message)
    if action_hint:
        st.caption(f"💡 {action_hint}")


def show_empty(
    title: str = "No data yet",
    *,
    hint: str | None = None,
    icon: str = "📭",
) -> None:
    """Show an empty/no-data state.
    
    显示空状态/无数据状态。
    
    Args:
        title: Empty state title
        hint: Optional hint for how to populate
        icon: Optional icon to display
    """
    st.markdown(f"**{icon} {title}**")
    if hint:
        st.caption(hint)


def show_feedback(
    state: str,
    message: str,
    *,
    retry_hint: str | None = None,
    details: str | None = None,
) -> None:
    """Unified feedback dispatcher.
    
    统一反馈分发器。
    根据状态类型调用相应的反馈函数。
    
    Args:
        state: One of "loading", "error", "success", "info", "warning", "empty"
        message: Message to display
        retry_hint: Optional retry hint (for error/warning states)
        details: Optional technical details (for error state)
    """
    state_lower = state.lower()
    
    if state_lower == "loading":
        show_loading(message)
    elif state_lower == "error":
        show_error(message, retry_hint=retry_hint, details=details)
    elif state_lower == "success":
        show_success(message)
    elif state_lower == "info":
        show_info(message)
    elif state_lower == "warning":
        show_warning(message, action_hint=retry_hint)
    elif state_lower == "empty":
        show_empty(message, hint=retry_hint)
    else:
        # Fallback to info for unknown states
        show_info(message)
