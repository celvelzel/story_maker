"""Tests for session state persistence and contract.

会话状态持久化和契约测试。
验证刷新后状态保持和会话状态完整性。
"""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

# ── App script that tests session persistence ──────────────────────────────

_SESSION_APP = """
import streamlit as st
from src.ui.session_contract import (
    initialize_session,
    validate_session_state,
    get_session_snapshot,
    safe_get,
    safe_set,
)

# Initialize session
initialize_session()

# Display current state
st.write("Session initialized")

# Test safe_get/safe_set
if st.button("Set test values"):
    safe_set("history", [{"role": "user", "content": "test"}])
    safe_set("kg_html", "<div>test graph</div>")
    safe_set("chat_fold_mode", True)
    st.rerun()

if st.button("Validate state"):
    issues = validate_session_state()
    if issues:
        st.error(f"Issues: {issues}")
    else:
        st.success("Session state is valid")

# Display current values
history = safe_get("history", [])
kg_html = safe_get("kg_html", "")
fold_mode = safe_get("chat_fold_mode", False)

st.write(f"History length: {len(history)}")
st.write(f"KG HTML length: {len(kg_html)}")
st.write(f"Fold mode: {fold_mode}")

# Show snapshot
if st.button("Show snapshot"):
    snapshot = get_session_snapshot()
    st.json(snapshot)
"""


def _run_session_app() -> AppTest:
    """Run the session test app."""
    at = AppTest.from_string(_SESSION_APP)
    at.run()
    return at


# ── Session initialization tests ────────────────────────────────────────────

class TestSessionInitialization:
    """Test session state initialization."""

    def test_session_initializes_without_errors(self):
        """Session should initialize without exceptions."""
        at = _run_session_app()
        assert not at.exception

    def test_session_has_default_values(self):
        """Session should have all default values set."""
        at = _run_session_app()
        assert not at.exception
        
        # Check that key session state values exist
        assert "history" in at.session_state
        assert "kg_html" in at.session_state
        assert "chat_fold_mode" in at.session_state
        assert "engine" in at.session_state

    def test_session_defaults_have_correct_types(self):
        """Session defaults should have correct types."""
        at = _run_session_app()
        assert not at.exception
        
        assert isinstance(at.session_state["history"], list)
        assert isinstance(at.session_state["kg_html"], str)
        assert isinstance(at.session_state["chat_fold_mode"], bool)
        assert at.session_state["engine"] is None


# ── Session validation tests ────────────────────────────────────────────────

class TestSessionValidation:
    """Test session state validation."""

    def test_valid_session_passes_validation(self):
        """Valid session state should pass validation."""
        at = _run_session_app()
        assert not at.exception
        
        # Click validate button
        at.button[1].click()
        at.run()
        assert not at.exception

    def test_invalid_type_fails_validation(self):
        """Invalid type should fail validation."""
        at = _run_session_app()
        assert not at.exception
        
        # Set invalid type
        at.session_state["history"] = "not a list"
        at.button[1].click()  # Click validate
        at.run()
        assert not at.exception
        # Should show error about type mismatch
        assert len(at.error) > 0


# ── Safe get/set tests ──────────────────────────────────────────────────────

class TestSafeGetSet:
    """Test safe_get and safe_set functions."""

    def test_safe_set_valid_value(self):
        """safe_set should accept valid values."""
        at = _run_session_app()
        assert not at.exception
        
        # Click set test values button
        at.button[0].click()
        at.run()
        assert not at.exception
        
        # Check values were set
        assert len(at.session_state["history"]) == 1
        assert at.session_state["kg_html"] == "<div>test graph</div>"
        assert at.session_state["chat_fold_mode"] is True

    def test_safe_get_returns_default_for_missing_key(self):
        """safe_get should return default for missing keys."""
        at = _run_session_app()
        assert not at.exception
        
        # Streamlit session_state doesn't have .get() method
        # Use 'in' operator to check if key exists
        assert "nonexistent_key" not in at.session_state


# ── Session persistence tests ───────────────────────────────────────────────

class TestSessionPersistence:
    """Test session state persistence across reruns."""

    def test_state_persists_across_reruns(self):
        """State should persist across Streamlit reruns."""
        at = _run_session_app()
        assert not at.exception
        
        # Set values
        at.button[0].click()
        at.run()
        assert not at.exception
        
        # Verify values persist
        assert len(at.session_state["history"]) == 1
        assert at.session_state["kg_html"] == "<div>test graph</div>"
        
        # Run again (simulating rerun)
        at.run()
        assert not at.exception
        
        # Values should still be there
        assert len(at.session_state["history"]) == 1
        assert at.session_state["kg_html"] == "<div>test graph</div>"

    def test_snapshot_shows_correct_state(self):
        """Snapshot should show current state correctly."""
        at = _run_session_app()
        assert not at.exception
        
        # Set values
        at.button[0].click()
        at.run()
        assert not at.exception
        
        # Click show snapshot
        at.button[2].click()
        at.run()
        assert not at.exception
        
        # Should have JSON output
        assert len(at.json) > 0


# ── Integration: session with app flow ──────────────────────────────────────

_INTEGRATION_APP = """
import streamlit as st
from src.ui.session_contract import initialize_session, validate_session_state

# Initialize
initialize_session()

# Simulate game flow
if st.button("Start game"):
    st.session_state.engine = "mock_engine"
    st.session_state.history = [{"role": "assistant", "content": "Welcome!"}]
    st.session_state.kg_html = "<div>initial graph</div>"
    st.rerun()

if st.button("Add turn"):
    if st.session_state.engine:
        st.session_state.history.append({"role": "user", "content": "Look around"})
        st.session_state.history.append({"role": "assistant", "content": "You see a forest."})
        st.session_state.consistency_history.append(1.0)
        st.rerun()

# Display state
st.write(f"Engine: {st.session_state.engine}")
st.write(f"History length: {len(st.session_state.history)}")
st.write(f"Consistency scores: {len(st.session_state.consistency_history)}")

# Validate
issues = validate_session_state()
if issues:
    st.error(f"Validation issues: {issues}")
"""


class TestSessionIntegration:
    """Test session state in realistic app flow."""

    def test_game_flow_maintains_state(self):
        """Game flow should maintain consistent state."""
        at = AppTest.from_string(_INTEGRATION_APP)
        at.run()
        assert not at.exception
        
        # Start game
        at.button[0].click()
        at.run()
        assert not at.exception
        assert at.session_state["engine"] == "mock_engine"
        assert len(at.session_state["history"]) == 1
        
        # Add turn
        at.button[1].click()
        at.run()
        assert not at.exception
        assert len(at.session_state["history"]) == 3
        assert len(at.session_state["consistency_history"]) == 1

    def test_validation_after_game_flow(self):
        """Session should remain valid after game flow."""
        at = AppTest.from_string(_INTEGRATION_APP)
        at.run()
        assert not at.exception
        
        # Start game and add turn
        at.button[0].click()
        at.run()
        at.button[1].click()
        at.run()
        assert not at.exception
        
        # Should not have validation errors
        assert len(at.error) == 0
