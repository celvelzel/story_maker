"""Runtime session persistence helpers for Streamlit refresh recovery."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.nlg.option_generator import StoryOption

RUNTIME_SESSION_FILENAME = "runtime_session.json"
RUNTIME_ENGINE_FILENAME = "runtime_engine.json"


def runtime_session_path(save_dir: str | Path) -> Path:
    """Return the metadata JSON path used for runtime session restore."""
    return Path(save_dir) / RUNTIME_SESSION_FILENAME


def runtime_engine_path(save_dir: str | Path) -> Path:
    """Return the engine snapshot path used for runtime session restore."""
    return Path(save_dir) / RUNTIME_ENGINE_FILENAME


def serialize_options(options: Iterable[StoryOption]) -> List[Dict[str, str]]:
    """Serialize StoryOption objects to JSON-friendly dicts."""
    return [
        {
            "text": o.text,
            "intent_hint": o.intent_hint,
            "risk_level": o.risk_level,
        }
        for o in options
    ]


def deserialize_options(raw_options: Iterable[Dict[str, Any]]) -> List[StoryOption]:
    """Deserialize option dicts to StoryOption objects."""
    options: List[StoryOption] = []
    for item in raw_options:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        options.append(
            StoryOption(
                text=text,
                intent_hint=str(item.get("intent_hint", "other")),
                risk_level=str(item.get("risk_level", "medium")),
            )
        )
    return options


def save_runtime_session(save_dir: str | Path, payload: Dict[str, Any]) -> str:
    """Persist runtime session metadata JSON and return file path."""
    path = runtime_session_path(save_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)


def load_runtime_session(save_dir: str | Path) -> Dict[str, Any] | None:
    """Load runtime session metadata JSON if it exists."""
    path = runtime_session_path(save_dir)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def remove_runtime_files(save_dir: str | Path) -> List[str]:
    """Delete runtime session files and return deleted file paths."""
    removed: List[str] = []
    for path in (runtime_session_path(save_dir), runtime_engine_path(save_dir)):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed
