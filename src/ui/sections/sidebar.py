"""Sidebar rendering for StoryWeaver."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import streamlit as st
import streamlit.components.v1 as components

from config import settings
from src.engine.game_engine import GameEngine
from src.knowledge_graph.visualizer import render_kg_html


def _runtime_save_dir() -> Path:
    return Path(settings.KG_SAVE_DIR)


def _story_turn_count() -> int:
    return sum(1 for m in st.session_state.history if m["role"] == "user")


def _list_save_slots(save_dir: Path) -> list[dict[str, str]]:
    """Scan save directory and return sorted save-slot metadata for UI."""
    slots: list[dict[str, str]] = []
    if not save_dir.exists():
        return slots

    for file in save_dir.glob("*.json"):
        if file.name == "runtime_session.json" or file.name == "runtime_engine.json":
            continue

        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        state = data.get("state", {}) if isinstance(data.get("state"), dict) else {}
        try:
            turn_id = int(state.get("turn_id", 0))
        except (TypeError, ValueError):
            turn_id = 0
        genre = str(data.get("genre", state.get("genre", "unknown")))
        save_title = str(data.get("save_title", "")).strip() or "Untitled Snapshot"
        mtime = file.stat().st_mtime
        updated_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

        label = f"{save_title} | {genre} · Turn {turn_id} · {updated_at}"
        slots.append(
            {
                "label": label,
                "path": str(file),
                "filename": file.name,
                "mtime": str(mtime),
            }
        )

    slots.sort(key=lambda item: float(item.get("mtime", "0")), reverse=True)
    return slots


def _restore_from_save_file(filepath: str) -> tuple[bool, str]:
    """Load selected save file into current Streamlit session state."""
    path = Path(filepath)
    if not path.exists():
        return False, f"Save file not found: {path.name}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return False, f"Failed to read save file: {exc}"

    if not isinstance(data, dict):
        return False, "Invalid save format."

    state_data = data.get("state", {}) if isinstance(data.get("state"), dict) else {}
    saved_genre = str(data.get("genre", state_data.get("genre", "fantasy"))) or "fantasy"

    try:
        intent_model_path_raw = st.session_state.intent_model_path or ""
        intent_model_path = intent_model_path_raw.strip() or None
        engine = GameEngine(
            genre=saved_genre,
            intent_model_path=intent_model_path,
            conflict_resolution=str(data.get("conflict_resolution", settings.KG_CONFLICT_RESOLUTION)),
            extraction_mode=str(data.get("extraction_mode", settings.KG_EXTRACTION_MODE)),
            importance_mode=str(data.get("importance_mode", settings.KG_IMPORTANCE_MODE)),
            summary_mode=str(data.get("summary_mode", settings.KG_SUMMARY_MODE)),
        )
        engine.load_game(str(path))

        restored_history: list[dict[str, str]] = []
        for item in engine.state.story_history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            if role == "player":
                restored_history.append({"role": "user", "content": text})
            elif role == "narrator":
                restored_history.append({"role": "assistant", "content": text})

        last_story = ""
        for item in reversed(engine.state.story_history):
            if isinstance(item, dict) and item.get("role") == "narrator":
                last_story = str(item.get("text", "")).strip()
                if last_story:
                    break

        options = engine.option_gen.generate(last_story, engine.kg.to_summary()) if last_story else []
        kg_html = render_kg_html(engine.kg.graph)

        st.session_state.engine = engine
        st.session_state.genre_input = saved_genre
        st.session_state.history = restored_history
        st.session_state.kg_html = kg_html
        st.session_state.options = options
        st.session_state.consistency_history = []
        st.session_state.nlu_debug = {}
        st.session_state.eval_result = ""
        st.session_state.eval_auto = {}
        st.session_state.eval_llm = {}
        st.session_state.eval_prev_auto = {}
        st.session_state.eval_prev_llm = {}
        st.session_state.eval_at = ""
        st.session_state.last_elapsed = 0.0
        st.session_state.processing = False
        st.session_state.kg_conflict_resolution = engine.conflict_resolution
        st.session_state.kg_extraction_mode = engine.extraction_mode
        st.session_state.kg_importance_mode = engine.importance_mode
        st.session_state.kg_summary_mode = engine.summary_mode

    except Exception as exc:
        return False, f"Failed to load save: {exc}"

    return True, f"Loaded save: {path.name}"


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("<div class='section-title'>&#x1F39B;&#xFE0F; Dashboard</div>", unsafe_allow_html=True)
        turn_count = _story_turn_count()
        engine: GameEngine | None = st.session_state.engine
        entity_count = len(engine.kg_entity_names) if engine else 0
        conflict_total = sum(engine.turn_conflict_counts) if engine else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Turns", turn_count)
        c2.metric("Entities", entity_count)
        c3.metric("Conflicts", conflict_total)

        if engine:
            st.caption(
                f"Strategy: {engine.conflict_resolution} | {engine.extraction_mode} | "
                f"{engine.summary_mode} | {engine.importance_mode}"
            )

        st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>&#x1F4CA; Story World Knowledge Graph</div>", unsafe_allow_html=True)
        if st.session_state.kg_html:
            st.markdown("<div class='kg-frame'>", unsafe_allow_html=True)
            components.html(st.session_state.kg_html, height=480, scrolling=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("The knowledge graph will appear after starting a game.")

        with st.expander("🔍 NLU Parsing Details", expanded=False):
            nlu = st.session_state.nlu_debug
            if nlu:
                st.markdown(f"**Resolved Input:** {nlu.get('resolved_input', '')}")
                st.markdown(
                    f"**Intent:** {nlu.get('intent', '?')}  "
                    f"(confidence {nlu.get('confidence', 0):.2f})"
                )
                st.markdown(f"**Backend:** {nlu.get('intent_backend', 'rule_fallback')}")
                st.markdown(f"**Model Loaded:** {nlu.get('intent_model_loaded', False)}")
                st.markdown(f"**Entities:** {nlu.get('entities', [])}")
            else:
                st.caption("NLU parsing details will appear after each action")

        st.markdown("<div class='section-title'>&#x1F4C8; Consistency Trend</div>", unsafe_allow_html=True)
        if st.session_state.consistency_history:
            recent = st.session_state.consistency_history[-5:]
            offset = max(0, len(st.session_state.consistency_history) - 5)
            for i, score in enumerate(recent):
                st.progress(max(0.0, min(1.0, score)), text=f"Turn {offset + i + 1}: {score:.2f}")
            st.line_chart(st.session_state.consistency_history, height=120, use_container_width=True)
        else:
            st.caption("Consistency scores appear after the first action")

        st.markdown("<div class='section-title'>&#x2699; Settings</div>", unsafe_allow_html=True)
        with st.expander("💾 Save / Load", expanded=True):
            save_slots = _list_save_slots(_runtime_save_dir())
            if save_slots:
                selected_idx = st.selectbox(
                    "Choose a save slot",
                    options=list(range(len(save_slots))),
                    format_func=lambda i: save_slots[i]["label"],
                    help="Snapshot title is generated by LLM during periodic turn snapshots.",
                )
                st.caption(f"Available saves: {len(save_slots)}")
                if st.button("📂 Load Selected Save", width="stretch", disabled=st.session_state.processing):
                    with st.spinner("Loading selected save..."):
                        ok, message = _restore_from_save_file(save_slots[selected_idx]["path"])
                    if ok:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.caption("No save files found yet. Play a few turns to generate snapshots.")

        with st.expander("⚙ KG Strategy Settings", expanded=False):
            st.caption('Strategy changes take effect after the next "Start New Game".')

            st.session_state.kg_conflict_resolution = st.selectbox(
                "Conflict resolution strategy",
                ["llm_arbitrate", "keep_latest"],
                index=0 if st.session_state.kg_conflict_resolution == "llm_arbitrate" else 1,
                help=(
                    "llm_arbitrate: LLM decides which information to keep (best quality)\n"
                    "keep_latest: Keep the most recently updated information (no LLM call required)"
                ),
            )

            st.session_state.kg_extraction_mode = st.selectbox(
                "Entity extraction mode",
                ["dual_extract", "story_only"],
                index=0 if st.session_state.kg_extraction_mode == "dual_extract" else 1,
                help=(
                    "dual_extract: Extract from both player input and story text for fuller context\n"
                    "story_only: Extract only from story text (backward compatible)"
                ),
            )

            st.session_state.kg_summary_mode = st.selectbox(
                "KG summary format",
                ["layered", "flat"],
                index=0 if st.session_state.kg_summary_mode == "layered" else 1,
                help=(
                    "layered: Group by importance (core/secondary/background), with descriptions and timeline\n"
                    "flat: Simple list format (backward compatible)"
                ),
            )

            st.session_state.kg_importance_mode = st.selectbox(
                "Entity pruning strategy",
                ["composite", "degree_only"],
                index=0 if st.session_state.kg_importance_mode == "composite" else 1,
                help=(
                    "composite: Score by degree + recency + mention_count\n"
                    "degree_only: Degree-only pruning (backward compatible)"
                ),
            )

        st.markdown("<hr class='neon-divider'>", unsafe_allow_html=True)

        if st.session_state.history:
            full_story = "\n\n".join(
                f"{'[Player]' if m['role'] == 'user' else '[Story]'}: {m['content']}"
                for m in st.session_state.history
            )
            st.download_button(
                "📥 Download Full Story",
                full_story,
                file_name="full_story.txt",
                mime="text/plain",
                use_container_width=True,
            )


__all__ = ["render_sidebar"]
