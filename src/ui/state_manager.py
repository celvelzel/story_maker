"""Session state management for StoryWeaver.

Handles initialization, persistence, and restoration of session state.
"""

import streamlit as st
from pathlib import Path
from src.engine.game_engine import GameEngine
from src.engine.runtime_session import (
    deserialize_options,
    load_runtime_session,
)
from config import settings


_DEFAULTS = {
    "engine": None,
    "history": [],                # list[dict] with role / content
    "consistency_history": [],    # float per turn
    "kg_html": "",
    "options": [],                # list[StoryOption]
    "nlu_debug": {},
    "eval_result": "",
    "eval_auto": {},
    "eval_llm": {},
    "eval_prev_auto": {},
    "eval_prev_llm": {},
    "eval_at": "",
    "chat_fold_mode": False,
    "last_elapsed": 0.0,
    "intent_model_path": str(settings.INTENT_MODEL_PATH),
    # KG strategy settings
    "kg_conflict_resolution": settings.KG_CONFLICT_RESOLUTION,
    "kg_extraction_mode": settings.KG_EXTRACTION_MODE,
    "kg_importance_mode": settings.KG_IMPORTANCE_MODE,
    "kg_summary_mode": settings.KG_SUMMARY_MODE,
    "processing": False,
    "genre_input": "fantasy",
}


def initialize_state() -> None:
    """Initialize session state with default values."""
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def restore_runtime_session(save_dir: Path) -> None:
    """Restore session from runtime save files (called once at startup).
    
    Loads persisted game state if available.
    """
    if st.session_state.engine is not None:
        return

    data = load_runtime_session(save_dir)
    if not data:
        return

    engine_file = str(data.get("engine_file", "")).strip()
    if not engine_file or not Path(engine_file).exists():
        return

    intent_model_path = str(data.get("intent_model_path", "")).strip() or None
    engine = GameEngine(
        genre=str(data.get("genre", "fantasy")) or "fantasy",
        intent_model_path=intent_model_path,
        conflict_resolution=str(data.get("kg_conflict_resolution", settings.KG_CONFLICT_RESOLUTION)),
        extraction_mode=str(data.get("kg_extraction_mode", settings.KG_EXTRACTION_MODE)),
        importance_mode=str(data.get("kg_importance_mode", settings.KG_IMPORTANCE_MODE)),
        summary_mode=str(data.get("kg_summary_mode", settings.KG_SUMMARY_MODE)),
    )
    engine.load_game(engine_file)

    st.session_state.engine = engine
    st.session_state.history = data.get("history", []) if isinstance(data.get("history"), list) else []
    st.session_state.consistency_history = (
        data.get("consistency_history", [])
        if isinstance(data.get("consistency_history"), list)
        else []
    )
    st.session_state.kg_html = str(data.get("kg_html", ""))
    st.session_state.options = deserialize_options(data.get("options", []))
    st.session_state.nlu_debug = data.get("nlu_debug", {}) if isinstance(data.get("nlu_debug"), dict) else {}
    st.session_state.chat_fold_mode = bool(data.get("chat_fold_mode", False))
    st.session_state.last_elapsed = float(data.get("last_elapsed", 0.0))
    st.session_state.intent_model_path = str(data.get("intent_model_path", st.session_state.intent_model_path))
    st.session_state.kg_conflict_resolution = str(
        data.get("kg_conflict_resolution", st.session_state.kg_conflict_resolution)
    )
    st.session_state.kg_extraction_mode = str(data.get("kg_extraction_mode", st.session_state.kg_extraction_mode))
    st.session_state.kg_importance_mode = str(data.get("kg_importance_mode", st.session_state.kg_importance_mode))
    st.session_state.kg_summary_mode = str(data.get("kg_summary_mode", st.session_state.kg_summary_mode))
    st.session_state.eval_result = str(data.get("eval_result", ""))
    st.session_state.eval_auto = data.get("eval_auto", {}) if isinstance(data.get("eval_auto"), dict) else {}
    st.session_state.eval_llm = data.get("eval_llm", {}) if isinstance(data.get("eval_llm"), dict) else {}
    st.session_state.eval_prev_auto = (
        data.get("eval_prev_auto", {}) if isinstance(data.get("eval_prev_auto"), dict) else {}
    )
    st.session_state.eval_prev_llm = (
        data.get("eval_prev_llm", {}) if isinstance(data.get("eval_prev_llm"), dict) else {}
    )
    st.session_state.eval_at = str(data.get("eval_at", ""))


__all__ = ["initialize_state", "restore_runtime_session", "_DEFAULTS"]
