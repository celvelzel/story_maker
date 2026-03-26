# Work Plan: app.py Code Modularization

## Objective
Slim down `app.py` (currently 1600+ lines) by extracting CSS, state management, and UI sections into modular files under `src/ui/`. This will significantly improve maintainability and separate concerns.

## Scope
- IN: Moving CSS, theme definitions, session state initialization, and major UI sections (sidebar, chat, evaluation) to separate modules.
- IN: Updating `app.py` to import and use these new modules.
- OUT: Changing business logic, game engine mechanics, or any backend models.
- OUT: Rewriting the UI components to work differently (pure extraction).

## Design & Architecture
- **Theme/CSS**: `src/ui/layout/theme.py` will contain a function `load_theme()` which injects the Streamlit markdown.
- **State Management**: `src/ui/state_manager.py` will handle initializing `st.session_state` variables and persisting/loading save states.
- **Sections**:
  - `src/ui/sections/sidebar.py`: Will contain `render_sidebar()`.
  - `src/ui/sections/chat.py`: Will contain `render_chat_history()` and `render_chat_input()`.
  - `src/ui/sections/evaluation.py`: Will contain `render_evaluation_dashboard()`.

## TODOs
- [x] [app.py: Move theme to `src/ui/layout/theme.py`] Extract 700+ lines of CSS to `load_theme()` — expect `app.py` to become significantly smaller.
- [x] [app.py: Move state initialization to `src/ui/state_manager.py`] Create `initialize_state()` and move `_DEFAULTS` and `_restore_runtime_session_once` logic — expect cleaner setup phase.
- [x] [app.py: Move sidebar to `src/ui/sections/sidebar.py`] Extract the sidebar rendering block into a function — expect `app.py` to import `render_sidebar`.
- [x] [app.py: Move chat history & input to `src/ui/sections/chat.py`] Extract chat rendering and input processing — expect `app.py` to import chat functions.
- [x] [app.py: Move evaluation dashboard to `src/ui/sections/evaluation.py`] Extract evaluation UI — expect `app.py` to import `render_evaluation`.
- [x] [app.py: Clean up and test] Verify `app.py` runs without errors — expect `streamlit run app.py` to render the exact same UI as before.

## Final Verification Wave
- [ ] F1: App starts correctly (`streamlit run app.py`).
- [ ] F2: CSS styles are correctly applied after modularization.
- [ ] F3: Session state works (chat, actions, eval panel).
- [ ] F4: Modularization scope respected (no business-logic behavior changes).
