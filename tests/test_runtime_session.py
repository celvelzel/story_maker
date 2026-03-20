"""Tests for runtime session persistence helpers."""
from pathlib import Path

from src.engine.runtime_session import (
    deserialize_options,
    load_runtime_session,
    remove_runtime_files,
    runtime_engine_path,
    runtime_session_path,
    save_runtime_session,
    serialize_options,
)
from src.nlg.option_generator import StoryOption


def test_runtime_paths(tmp_path: Path):
    assert runtime_session_path(tmp_path).name == "runtime_session.json"
    assert runtime_engine_path(tmp_path).name == "runtime_engine.json"


def test_save_and_load_runtime_session(tmp_path: Path):
    payload = {
        "version": 1,
        "genre": "fantasy",
        "history": [{"role": "assistant", "content": "Opening."}],
    }
    saved = save_runtime_session(tmp_path, payload)
    assert Path(saved).exists()

    loaded = load_runtime_session(tmp_path)
    assert loaded is not None
    assert loaded["version"] == 1
    assert loaded["genre"] == "fantasy"
    assert loaded["history"][0]["content"] == "Opening."


def test_serialize_deserialize_options_roundtrip():
    options = [
        StoryOption("Investigate the ruins", "explore", "low"),
        StoryOption("Confront the guard", "action", "high"),
    ]
    raw = serialize_options(options)
    restored = deserialize_options(raw)

    assert len(restored) == 2
    assert restored[0].text == "Investigate the ruins"
    assert restored[0].intent_hint == "explore"
    assert restored[1].risk_level == "high"


def test_deserialize_options_skips_invalid_items():
    restored = deserialize_options([
        {"text": "", "intent_hint": "other", "risk_level": "low"},
        "invalid",
        {"text": "Valid option"},
    ])
    assert len(restored) == 1
    assert restored[0].text == "Valid option"
    assert restored[0].intent_hint == "other"
    assert restored[0].risk_level == "medium"


def test_remove_runtime_files(tmp_path: Path):
    save_runtime_session(tmp_path, {"version": 1})
    runtime_engine_path(tmp_path).write_text("{}", encoding="utf-8")

    removed = remove_runtime_files(tmp_path)
    assert len(removed) == 2
    assert not runtime_session_path(tmp_path).exists()
    assert not runtime_engine_path(tmp_path).exists()
