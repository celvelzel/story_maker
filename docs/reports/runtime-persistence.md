# Runtime Persistence Report

**Date**: 2026-03-31  
**Scope**: Session persistence across browser refreshes; runtime data cleanup only on terminal stop (Ctrl+C).

## 1. Problem and Objectives

### Original Issues
Streamlit's `st.session_state` is only valid for the current browser session. Refreshing the page resets the session, causing:
- Current story dialogue to disappear.
- Knowledge Graph visualization to reset.
- Evaluation metrics and generated options to be lost.

### Objectives
- Automatically restore the current session (story, KG, options) after a browser refresh.
- Delete runtime session files only when the service is stopped via `Ctrl+C` in the terminal.
- Maintain existing `saves/` capability for permanent snapshots without automatic deletion.
- Ensure `saves/` remains excluded from Git tracking.

## 2. Implementation Details

### 2.1 Runtime Session Persistence Module
**New File**: `src/engine/runtime_session.py`

**Key Features**:
- **Path Management**: Handles `runtime_session.json` and `runtime_engine.json`.
- **Serialization**: Supports `StoryOption <-> dict` conversion for complex UI states.
- **I/O Operations**: Provides `save_runtime_session(...)` and `load_runtime_session(...)`.
- **Cleanup**: `remove_runtime_files(...)` specifically targets runtime artifacts.

### 2.2 Integration in Streamlit App
**Modified File**: `app.py`

**Core Logic**:
- `_persist_runtime_session()`: Saves engine state and UI-related states (chat history, KG HTML, options, evaluation data).
- `_restore_runtime_session_once()`: Attempt restoration on app startup. It loads the engine state first, then populates `st.session_state`.
- `_cleanup_runtime_files()` + `_register_runtime_cleanup()`: Uses `atexit` and `SIGINT` (Ctrl+C) handlers to ensure cleanup only happens on server shutdown.

**Persistence Triggers**:
- After clicking "Start New Game".
- After every action processing cycle (`_process_action`).
- After running evaluation (ensures results persist across refreshes).

## 3. Git Tracking Strategy
`.gitignore` already includes the `saves/` directory. Runtime files and standard save files remain untracked by Git, maintaining a clean repository while allowing local persistence.

## 4. Testing and Verification

### 4.1 New Unit Tests
**New File**: `tests/test_runtime_session.py`

**Coverage**:
- Correct runtime file path generation.
- Runtime metadata save/load integrity.
- `StoryOption` serialization/deserialization.
- Targeted deletion of runtime files (ensuring permanent saves are untouched).

### 4.2 Regression Testing
Validated against existing persistence tests:
- `tests/test_kg_persistence.py`

### 4.3 Results
- `pytest tests/test_runtime_session.py -q` → **5 passed**
- `pytest tests/test_kg_persistence.py -q` → **15 passed**

## 5. Behavior Summary
- **Browser Refresh**: Automatically restores the active session and KG.
- **Terminal Stop (Ctrl+C)**: Deletes only `runtime_session.json` and `runtime_engine.json`.
- **Manual/Auto Saves**: Permanent snapshots in `saves/` are preserved.

## 6. Affected Files
- `app.py`: Integrated restoration, persistence, and cleanup logic.
- `src/engine/runtime_session.py`: Core persistence logic.
- `tests/test_runtime_session.py`: Verification suite.
