"""Session state contract for StoryWeaver UI.

StoryWeaver 会话状态契约。
定义前端关键状态键的模式、默认值和验证助手。

Usage:
    from src.ui.session_contract import SESSION_DEFAULTS, validate_session_state, initialize_session
    
    # Initialize all session state keys with defaults
    initialize_session()
    
    # Validate session state integrity
    issues = validate_session_state()
    if issues:
        st.warning(f"Session state issues: {issues}")
"""
from __future__ import annotations

from typing import Any

import streamlit as st


# ── Session State Schema ────────────────────────────────────────────────────

SESSION_DEFAULTS: dict[str, Any] = {
    # Core game state
    "engine": None,                    # GameEngine instance
    "history": [],                     # list[dict] with role/content
    "consistency_history": [],         # float per turn
    "kg_html": "",                     # rendered KG HTML string
    "options": [],                     # list[StoryOption]
    "nlu_debug": {},                   # NLU parsing debug dict
    
    # Evaluation state
    "eval_result": "",                 # evaluation report markdown
    "eval_auto": {},                   # automatic metrics dict
    "eval_llm": {},                    # LLM judge scores dict
    "eval_prev_auto": {},              # previous automatic metrics
    "eval_prev_llm": {},               # previous LLM judge scores
    "eval_at": "",                     # last evaluation timestamp
    
    # UI state
    "chat_fold_mode": False,           # fold history by turn toggle
    "last_elapsed": 0.0,               # last generation time in seconds
    "ui_mode": "dark",                 # UI theme mode
    
    # Configuration (persisted)
    "intent_model_path": "",           # NLU model path
    "kg_conflict_resolution": "llm_arbitrate",
    "kg_extraction_mode": "dual_extract",
    "kg_importance_mode": "composite",
    "kg_summary_mode": "layered",
}

# Type hints for validation
_SESSION_TYPES: dict[str, type | tuple[type, ...]] = {
    "engine": (type(None), object),  # GameEngine or None
    "history": list,
    "consistency_history": list,
    "kg_html": str,
    "options": list,
    "nlu_debug": dict,
    "eval_result": str,
    "eval_auto": dict,
    "eval_llm": dict,
    "eval_prev_auto": dict,
    "eval_prev_llm": dict,
    "eval_at": str,
    "chat_fold_mode": bool,
    "last_elapsed": (int, float),
    "ui_mode": str,
    "intent_model_path": str,
    "kg_conflict_resolution": str,
    "kg_extraction_mode": str,
    "kg_importance_mode": str,
    "kg_summary_mode": str,
}


def initialize_session() -> None:
    """Initialize all session state keys with defaults.
    
    初始化所有会话状态键的默认值。
    只设置尚未存在的键，不覆盖已有值。
    """
    for key, default_value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def validate_session_state() -> list[str]:
    """Validate session state integrity and return list of issues.
    
    验证会话状态完整性，返回问题列表。
    
    Returns:
        List of issue descriptions (empty if all valid)
    """
    issues = []
    
    for key, expected_type in _SESSION_TYPES.items():
        if key not in st.session_state:
            issues.append(f"Missing key: {key}")
            continue
        
        value = st.session_state[key]
        if not isinstance(value, expected_type):
            issues.append(
                f"Type mismatch for {key}: expected {expected_type}, "
                f"got {type(value).__name__}"
            )
    
    # Validate specific constraints
    if st.session_state.get("ui_mode") not in ("dark", "light"):
        issues.append(f"Invalid ui_mode: {st.session_state.get('ui_mode')}")
    
    if st.session_state.get("kg_conflict_resolution") not in ("llm_arbitrate", "keep_latest"):
        issues.append(f"Invalid kg_conflict_resolution: {st.session_state.get('kg_conflict_resolution')}")
    
    if st.session_state.get("kg_extraction_mode") not in ("dual_extract", "story_only"):
        issues.append(f"Invalid kg_extraction_mode: {st.session_state.get('kg_extraction_mode')}")
    
    if st.session_state.get("kg_importance_mode") not in ("composite", "degree_only"):
        issues.append(f"Invalid kg_importance_mode: {st.session_state.get('kg_importance_mode')}")
    
    if st.session_state.get("kg_summary_mode") not in ("layered", "flat"):
        issues.append(f"Invalid kg_summary_mode: {st.session_state.get('kg_summary_mode')}")
    
    return issues


def get_session_snapshot() -> dict[str, Any]:
    """Get a snapshot of current session state for debugging.
    
    获取当前会话状态的快照用于调试。
    
    Returns:
        Dict with all session state keys and their types
    """
    snapshot = {}
    for key in SESSION_DEFAULTS:
        value = st.session_state.get(key)
        snapshot[key] = {
            "type": type(value).__name__,
            "value": repr(value)[:100] if value is not None else "None",
            "present": key in st.session_state,
        }
    return snapshot


def reset_session_to_defaults() -> None:
    """Reset all session state keys to their default values.
    
    将所有会话状态键重置为默认值。
    警告：这将清除所有当前状态！
    """
    for key, default_value in SESSION_DEFAULTS.items():
        st.session_state[key] = default_value


def safe_get(key: str, fallback: Any = None) -> Any:
    """Safely get a session state value with fallback.
    
    安全地获取会话状态值，带后备值。
    
    Args:
        key: Session state key
        fallback: Value to return if key is missing or invalid
        
    Returns:
        Session state value or fallback
    """
    if key not in st.session_state:
        return fallback
    
    value = st.session_state[key]
    expected_type = _SESSION_TYPES.get(key)
    
    if expected_type and not isinstance(value, expected_type):
        return fallback
    
    return value


def safe_set(key: str, value: Any) -> bool:
    """Safely set a session state value with type validation.
    
    安全地设置会话状态值，带类型验证。
    
    Args:
        key: Session state key
        value: Value to set
        
    Returns:
        True if value was set successfully, False if type mismatch
    """
    expected_type = _SESSION_TYPES.get(key)
    
    if expected_type and not isinstance(value, expected_type):
        return False
    
    st.session_state[key] = value
    return True
