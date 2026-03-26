"""Session state management for StoryWeaver.

Handles initialization, persistence, and restoration of session state.
"""

import logging
import streamlit as st
from pathlib import Path
from src.engine.game_engine import GameEngine
from src.engine.runtime_session import (
    deserialize_options,
    load_runtime_session,
)
from config import settings


logger = logging.getLogger(__name__)

# 内存优化配置
_MAX_HISTORY_SIZE = 100  # 保留的最大历史记录数
_MAX_CONSISTENCY_HISTORY = 50  # 保留的最大一致性历史数
_MAX_NLU_DEBUG_SIZE = 20  # 保留的最大 NLU 调试信息条数


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


def is_runtime_session_active(save_dir: Path) -> bool:
    """Return whether the persisted runtime session is marked active."""
    data = load_runtime_session(save_dir)
    return bool(data and data.get("is_active", False))


def restore_runtime_session(save_dir: Path, auto_restore: bool = True) -> None:
    """Restore session from runtime save files (called once at startup).
    
    Loads persisted game state if available.
    """
    if st.session_state.engine is not None:
        logger.info("Skip runtime restore: engine already initialized.")
        return

    data = load_runtime_session(save_dir)
    if not data:
        logger.info("Skip runtime restore: no runtime session data found.")
        return

    is_active = bool(data.get("is_active", False))
    logger.info("Runtime restore decision: auto_restore=%s, is_active=%s", auto_restore, is_active)
    if auto_restore and not is_active:
        logger.info("Skip runtime restore: session is not active and auto_restore is enabled.")
        return

    engine_file = str(data.get("engine_file", "")).strip()
    if not engine_file or not Path(engine_file).exists():
        logger.info("Skip runtime restore: engine file missing or invalid: %s", engine_file)
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
    logger.info("Runtime session restored successfully.")


__all__ = [
    "initialize_state",
    "restore_runtime_session",
    "is_runtime_session_active",
    "_DEFAULTS",
    "cleanup_session_state",
]


def cleanup_session_state() -> None:
    """清理过期的 session state 数据以释放内存。
    
    在以下情况调用：
    - 开始新游戏后
    - 加载存档后
    - 每隔一定次数的回合后
    
    清理策略：
    - history: 保留最近 MAX_HISTORY_SIZE 条记录
    - consistency_history: 保留最近 MAX_CONSISTENCY_HISTORY 条记录
    - nlu_debug: 保留最近 MAX_NLU_DEBUG_SIZE 条记录
    """
    # 清理 history（保留最近的）
    history = st.session_state.get("history", [])
    if len(history) > _MAX_HISTORY_SIZE:
        # 保留开场白（第一条 assistant 消息）和最近的消息
        if history and history[0].get("role") == "assistant":
            # 保留第一条和最近的消息
            keep_count = _MAX_HISTORY_SIZE - 1
            st.session_state.history = [history[0]] + history[-(keep_count):]
        else:
            st.session_state.history = history[-_MAX_HISTORY_SIZE:]
    
    # 清理 consistency_history
    consistency = st.session_state.get("consistency_history", [])
    if len(consistency) > _MAX_CONSISTENCY_HISTORY:
        st.session_state.consistency_history = consistency[-_MAX_CONSISTENCY_HISTORY:]
    
    # 注意：不清理 kg_html，因为它是有状态的
    # 注意：不清理 nlu_debug，因为它已经是字典，内存占用有限
