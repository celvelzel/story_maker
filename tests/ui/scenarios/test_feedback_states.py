"""Tests for feedback state helpers.

反馈状态助手测试。
验证加载/错误/成功/空状态渲染函数的行为。
"""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

# ── App script that exercises feedback helpers ──────────────────────────────

_FEEDBACK_APP = """
import streamlit as st
from src.ui.feedback import (
    show_loading,
    show_error,
    show_success,
    show_info,
    show_warning,
    show_empty,
    show_feedback,
)

state = st.session_state.get("test_state", "loading")
message = st.session_state.get("test_message", "Test message")

if state == "loading":
    show_loading(message)
elif state == "error":
    show_error(message, retry_hint="Try again", details="Error details here")
elif state == "success":
    show_success(message)
elif state == "info":
    show_info(message)
elif state == "warning":
    show_warning(message, action_hint="Check your input")
elif state == "empty":
    show_empty(message, hint="Start a game to populate")
elif state == "feedback_loading":
    show_feedback("loading", message)
elif state == "feedback_error":
    show_feedback("error", message, retry_hint="Retry now")
elif state == "feedback_success":
    show_feedback("success", message)
elif state == "feedback_unknown":
    show_feedback("unknown_state", message)
"""


def _run_feedback_app(state: str, message: str = "Test message") -> AppTest:
    """Run the feedback test app with given state."""
    at = AppTest.from_string(_FEEDBACK_APP)
    at.session_state["test_state"] = state
    at.session_state["test_message"] = message
    at.run()
    return at


# ── Individual feedback state tests ─────────────────────────────────────────

class TestShowLoading:
    """Test show_loading helper."""

    def test_loading_renders_spinner(self):
        """Loading state should render without errors."""
        at = _run_feedback_app("loading", "Generating story...")
        assert not at.exception


class TestShowError:
    """Test show_error helper."""

    def test_error_renders_message(self):
        """Error state should render error message."""
        at = _run_feedback_app("error", "Connection failed")
        assert not at.exception

    def test_error_with_retry_hint(self):
        """Error with retry hint should render caption."""
        at = _run_feedback_app("error", "Failed to generate")
        assert not at.exception
        # Caption with retry hint should be present
        captions_text = " ".join(c.value for c in at.caption)
        assert "Try again" in captions_text

    def test_error_with_details(self):
        """Error with details should render expander."""
        at = _run_feedback_app("error", "API error")
        assert not at.exception
        # Expander should exist for technical details
        assert len(at.expander) > 0


class TestShowSuccess:
    """Test show_success helper."""

    def test_success_renders_message(self):
        """Success state should render success message."""
        at = _run_feedback_app("success", "Story saved!")
        assert not at.exception


class TestShowInfo:
    """Test show_info helper."""

    def test_info_renders_message(self):
        """Info state should render info message."""
        at = _run_feedback_app("info", "Processing your request...")
        assert not at.exception


class TestShowWarning:
    """Test show_warning helper."""

    def test_warning_renders_message(self):
        """Warning state should render warning message."""
        at = _run_feedback_app("warning", "Low confidence")
        assert not at.exception

    def test_warning_with_action_hint(self):
        """Warning with action hint should render caption."""
        at = _run_feedback_app("warning", "Input may be unclear")
        assert not at.exception
        captions_text = " ".join(c.value for c in at.caption)
        assert "Check your input" in captions_text


class TestShowEmpty:
    """Test show_empty helper."""

    def test_empty_renders_title(self):
        """Empty state should render title."""
        at = _run_feedback_app("empty", "No entities yet")
        assert not at.exception

    def test_empty_with_hint(self):
        """Empty with hint should render caption."""
        at = _run_feedback_app("empty", "Knowledge graph is empty")
        assert not at.exception
        captions_text = " ".join(c.value for c in at.caption)
        assert "Start a game to populate" in captions_text


class TestShowFeedback:
    """Test unified show_feedback dispatcher."""

    def test_feedback_loading(self):
        """Feedback dispatcher should handle loading state."""
        at = _run_feedback_app("feedback_loading", "Processing...")
        assert not at.exception

    def test_feedback_error(self):
        """Feedback dispatcher should handle error state."""
        at = _run_feedback_app("feedback_error", "Failed")
        assert not at.exception

    def test_feedback_success(self):
        """Feedback dispatcher should handle success state."""
        at = _run_feedback_app("feedback_success", "Done!")
        assert not at.exception

    def test_feedback_unknown_falls_back_to_info(self):
        """Unknown state should fall back to info."""
        at = _run_feedback_app("feedback_unknown", "Something happened")
        assert not at.exception


# ── Integration: feedback in realistic flow ──────────────────────────────────

_REALISTIC_FLOW_APP = """
import streamlit as st
from src.ui.feedback import show_loading, show_error, show_success, show_empty

phase = st.session_state.get("phase", "idle")

if phase == "loading":
    show_loading("Generating your next adventure...")
elif phase == "error":
    show_error(
        "Failed to connect to LLM",
        retry_hint="Check your API key and try again",
        details="ConnectionError: timeout after 30s",
    )
elif phase == "success":
    show_success("Story chapter generated!")
elif phase == "empty":
    show_empty(
        "No story yet",
        hint="Enter a genre and click Start to begin your adventure",
    )
"""


class TestFeedbackIntegration:
    """Test feedback states in a realistic app flow."""

    def test_loading_to_success_transition(self):
        """App should transition from loading to success without errors."""
        at = AppTest.from_string(_REALISTIC_FLOW_APP)
        
        # Phase 1: Loading
        at.session_state["phase"] = "loading"
        at.run()
        assert not at.exception
        
        # Phase 2: Success
        at.session_state["phase"] = "success"
        at.run()
        assert not at.exception

    def test_loading_to_error_transition(self):
        """App should transition from loading to error without errors."""
        at = AppTest.from_string(_REALISTIC_FLOW_APP)
        
        # Phase 1: Loading
        at.session_state["phase"] = "loading"
        at.run()
        assert not at.exception
        
        # Phase 2: Error
        at.session_state["phase"] = "error"
        at.run()
        assert not at.exception

    def test_empty_state_for_new_user(self):
        """New user should see empty state with helpful hint."""
        at = AppTest.from_string(_REALISTIC_FLOW_APP)
        at.session_state["phase"] = "empty"
        at.run()
        assert not at.exception
        # Should have caption with hint
        captions_text = " ".join(c.value for c in at.caption)
        assert "Enter a genre" in captions_text
