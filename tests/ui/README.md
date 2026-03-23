# UI Smoke Test Scenarios

## Overview

This directory contains Streamlit AppTest-based smoke tests for the StoryWeaver frontend.

本目录包含基于 Streamlit AppTest 的 StoryWeaver 前端冒烟测试。

## Scenario Matrix

### Scenario 1: Start Game from Landing State
**ID**: `S1-start-game`
**Description**: User starts a new game from the initial landing page.
**Steps**:
1. Open app (landing state)
2. Enter genre in text input
3. Click "Start" button
4. Observe game engine initialization

**Expected Behavior**:
- Loading: `st.spinner` appears during initialization
- Success: Chat history shows initial story message
- Error: If engine fails, `st.error` with retry guidance
- Empty: N/A (always produces initial content)

**Test File**: `test_chat_primary_flow.py::test_start_game_landing`

---

### Scenario 2: Submit Free-Text Action
**ID**: `S2-free-text-action`
**Description**: User types a custom action and submits via chat input.
**Steps**:
1. Game already started (post S1)
2. Type action in `st.chat_input`
3. Submit action
4. Observe narrative response in chat

**Expected Behavior**:
- Loading: `st.spinner` during LLM generation
- Success: New assistant message appears in chat history
- Error: If LLM fails, error feedback with retry guidance
- Empty: N/A (always produces response)

**Test File**: `test_chat_primary_flow.py::test_submit_free_text_action`

---

### Scenario 3: Select Generated Option Button
**ID**: `S3-option-selection`
**Description**: User clicks one of the LLM-generated option buttons.
**Steps**:
1. Game in progress with options visible
2. Click one option button
3. Observe state transition and new narrative

**Expected Behavior**:
- Loading: `st.spinner` during turn processing
- Success: Selected option action processed, new story text appears
- Error: If processing fails, error feedback with context
- Empty: If no options available, informational message

**Test File**: `test_chat_primary_flow.py::test_select_option_button`

---

### Scenario 4: KG Panel Content Refresh
**ID**: `S4-kg-panel-refresh`
**Description**: Knowledge graph panel updates after a story turn.
**Steps**:
1. Complete a story turn (via S2 or S3)
2. Observe KG sidebar panel
3. Verify graph visualization updated

**Expected Behavior**:
- Loading: Panel shows loading state during KG update
- Success: PyVis HTML renders with updated entities/relations
- Error: If visualization fails, fallback message shown
- Empty: If no entities yet, informational empty state

**Test File**: `test_kg_sidebar_flow.py::test_kg_panel_refresh_after_turn`

---

## Baseline Evidence

**Last Run**: 2026-03-22
**Results**: 2 passed, 6 skipped (LLM not available)

| Scenario | Status | Notes |
|----------|--------|-------|
| S1-start-game | ⚠️ SKIPPED | Signal handler issue in AppTest thread (known limitation) |
| S1-landing-controls | ✅ PASSED | Genre input and Start button render correctly |
| S2-free-text-action | ⚠️ SKIPPED | Requires OPENAI_API_KEY |
| S3-option-selection | ⚠️ SKIPPED | Requires OPENAI_API_KEY |
| S4-kg-panel-empty | ✅ PASSED | KG panel shows empty state before game starts |
| S4-kg-panel-refresh | ⚠️ SKIPPED | Requires OPENAI_API_KEY |

### Known Issues
1. **Signal Handler**: `app.py` uses `signal.signal()` which fails in AppTest's non-main thread. Tests skip gracefully.
2. **LLM Dependency**: Most interaction tests require `OPENAI_API_KEY` environment variable.

## Running Tests

```bash
# Run all UI smoke tests
pytest tests/ui/scenarios -v

# Run specific scenario
pytest tests/ui/scenarios/test_chat_primary_flow.py -v
```

## Dependencies

- `pytest`
- `streamlit` (AppTest API available in streamlit >= 1.28.0)
