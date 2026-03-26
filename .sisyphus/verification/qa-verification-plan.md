# QA VERIFICATION PLAN — Task 1: KG Rendering Fix

## Overview

This document outlines the manual QA testing strategy for the Knowledge Graph rendering fix in `src/ui/sections/sidebar.py`.

**Objective**: Verify that after "Start New Game" is clicked, the KG visualization and Dashboard metrics appear immediately (without requiring a user action first).

**Tool**: Playwright browser automation
**Test Environment**: Local Streamlit app at `http://127.0.0.1:7860`

---

## Pre-QA Setup

### 1. Start the Streamlit App
```bash
streamlit run app.py --logger.level=debug
# Expected: App loads at http://127.0.0.1:7860
```

### 2. Environment Check
- [x] Streamlit available
- [x] Playwright/browser available
- [x] Python environment activated
- [x] `.env` configured with API key

---

## QA Scenarios

### Scenario 1: Start New Game → KG Renders Immediately

**Objective**: Verify KG appears in sidebar immediately after "Start New Game" is clicked

**Preconditions**:
- App is running at `http://127.0.0.1:7860`
- No active game session (fresh page load)
- Browser console is open for error checking

**Steps**:

1. Navigate to the app
   ```
   URL: http://127.0.0.1:7860
   ```

2. Wait for genre input to be visible
   ```
   Wait for selector: input with placeholder "Genre, e.g. fantasy / sci-fi / mystery"
   Timeout: 5000ms
   ```

3. Fill genre input with "fantasy"
   ```
   Input value: "fantasy"
   ```

4. Click "🎮 Start New Game" button
   ```
   Button text: "Start New Game"
   Type: "primary" button
   ```

5. Wait for spinner to disappear
   ```
   Wait for: .stSpinner to be hidden
   Timeout: 15000ms
   ```

6. Take screenshot for evidence
   ```
   Path: .sisyphus/evidence/qa-scenario1-kg-display.png
   Description: Screenshot showing KG and Dashboard after game start
   ```

7. Verify KG frame is visible
   ```
   Check CSS class: "kg-frame"
   Expected: Element is visible (not hidden, not empty)
   ```

8. Verify iframe contains graph
   ```
   Check: .kg-frame iframe
   Expected: At least 1 iframe element present
   Expected: iframe not empty (has content)
   ```

9. Verify Dashboard metrics are visible
   ```
   Check for text: "Turns"
   Expected: Visible in sidebar
   
   Check for text: "Entities"
   Expected: Visible in sidebar
   
   Check for text: "Conflicts"
   Expected: Visible in sidebar
   ```

10. Check browser console for errors
    ```
    Expected: No error messages related to:
    - render_kg_html
    - kg_html
    - sidebar rendering
    ```

**Expected Result**:
- ✅ KG visualization is visible in sidebar (not loading state)
- ✅ Dashboard shows Turns: 0, Entities: 2+, Conflicts: 0
- ✅ No console errors
- ✅ Screenshot shows clean UI with KG and Dashboard

**Failure Indicators**:
- ❌ KG frame is empty or still shows loading message
- ❌ Dashboard metrics not visible or show "?" 
- ❌ Console contains error messages
- ❌ Screenshot shows broken UI

---

### Scenario 2: First Action → KG Updates Without Breaking

**Objective**: Verify KG continues to render correctly after first player action

**Preconditions**:
- Scenario 1 completed successfully
- Game is running and showing initial story
- KG and Dashboard are visible
- Options/buttons are available

**Steps**:

1. Wait for "Branch Options" section
   ```
   Wait for text: "Branch Options"
   Timeout: 5000ms
   ```

2. Click the first branch option button
   ```
   Button: First option (usually "1. ...")
   Index: First button in options section
   ```

3. Wait for spinner to disappear
   ```
   Wait for: .stSpinner to be hidden
   Timeout: 20000ms
   ```

4. Take screenshot for evidence
   ```
   Path: .sisyphus/evidence/qa-scenario2-kg-update.png
   Description: Screenshot showing KG after first action
   ```

5. Verify KG frame still visible
   ```
   Check CSS class: "kg-frame"
   Expected: Still visible (not disappeared)
   ```

6. Verify Dashboard metrics updated
   ```
   Check Turns metric
   Expected: Changed from 0 to 1
   
   Check Entities metric
   Expected: Same or increased
   
   Check Conflicts metric
   Expected: Updated (may be 0 or >0)
   ```

7. Check browser console
   ```
   Expected: No new error messages
   ```

**Expected Result**:
- ✅ KG visualization remains visible
- ✅ Dashboard metrics update correctly (Turns: 1)
- ✅ Story progresses normally
- ✅ No console errors

**Failure Indicators**:
- ❌ KG disappears after action
- ❌ Dashboard metrics don't update
- ❌ Story content doesn't progress
- ❌ Console errors appear

---

### Scenario 3: Load Saved Game → KG Renders for Loaded State

**Objective**: Verify KG renders correctly when loading a saved game

**Preconditions**:
- A previous game exists with at least one save file
- App is running
- (May need to run Scenario 1+2 first to create a save)

**Steps**:

1. Click sidebar expander "💾 Save / Load"
   ```
   Button text: "Save / Load"
   Type: Expander header
   ```

2. Wait for save slots dropdown to appear
   ```
   Wait for selector: dropdown/selectbox
   Timeout: 5000ms
   Expected: At least 1 save slot listed
   ```

3. Select the first save slot
   ```
   Click dropdown
   Select: First option in list
   ```

4. Click "📂 Load Selected Save" button
   ```
   Button text: "Load Selected Save"
   ```

5. Wait for spinner to disappear
   ```
   Wait for: .stSpinner to be hidden
   Timeout: 15000ms
   ```

6. Take screenshot for evidence
   ```
   Path: .sisyphus/evidence/qa-scenario3-kg-loaded.png
   Description: Screenshot showing KG after loading save
   ```

7. Verify KG is visible
   ```
   Check CSS class: "kg-frame"
   Expected: Visible
   ```

8. Verify Dashboard shows loaded game state
   ```
   Check Turns metric
   Expected: Greater than 0 (shows loaded turn count)
   
   Check Entities metric
   Expected: Greater than 2
   
   Check Conflicts metric
   Expected: Some value (may be 0)
   ```

**Expected Result**:
- ✅ KG displays for loaded game
- ✅ Dashboard shows correct turn count from save
- ✅ Story content from save is visible
- ✅ No errors

**Failure Indicators**:
- ❌ KG doesn't appear after load
- ❌ Dashboard metrics are zero or missing
- ❌ Wrong save data loaded
- ❌ Error messages in console

---

### Scenario 4: Fresh App → Fallback Message Shows

**Objective**: Verify fallback message appears when no game is started

**Preconditions**:
- Fresh app load or multiple browser sessions
- User hasn't clicked "Start New Game"

**Steps**:

1. Navigate to app (or clear session)
   ```
   URL: http://127.0.0.1:7860
   Or: Open in new private/incognito window
   ```

2. Wait for page to load
   ```
   Wait for: genre input to be visible
   Timeout: 5000ms
   ```

3. Take screenshot for evidence
   ```
   Path: .sisyphus/evidence/qa-scenario4-no-game.png
   Description: Screenshot showing fallback when no game started
   ```

4. Verify fallback message appears
   ```
   Check for text: "The knowledge graph will appear after starting a game"
   Expected: This text is visible in sidebar
   Expected: Not an error message
   ```

5. Verify Dashboard is either hidden or shows zeros
   ```
   Check: Either Dashboard section is not visible
   OR Dashboard shows all zeros (Turns: 0, Entities: 0, Conflicts: 0)
   ```

6. Check console
   ```
   Expected: No error messages
   ```

**Expected Result**:
- ✅ Fallback message is displayed clearly
- ✅ Dashboard either hidden or shows safe default values
- ✅ No error messages
- ✅ Layout doesn't break

**Failure Indicators**:
- ❌ Fallback message missing or broken
- ❌ Error message displayed instead
- ❌ Dashboard throws error trying to render
- ❌ UI layout broken

---

## Evidence Collection

### Required Screenshots
- [ ] `qa-scenario1-kg-display.png` — KG visible after Start New Game
- [ ] `qa-scenario2-kg-update.png` — KG visible after first action
- [ ] `qa-scenario3-kg-loaded.png` — KG visible for loaded game
- [ ] `qa-scenario4-no-game.png` — Fallback message on fresh load

### Console Log Check
- [ ] No errors in browser console during any scenario
- [ ] No warnings related to rendering or state management
- [ ] No undefined reference errors

### Performance Observation
- [ ] Game startup completes in reasonable time (<10 seconds)
- [ ] Page remains responsive during rendering
- [ ] No apparent lag or stuttering

---

## Pass/Fail Criteria

### PASS Criteria (ALL must be true)
- ✅ Scenario 1: KG appears immediately after game start
- ✅ Scenario 2: KG continues to work after first action
- ✅ Scenario 3: KG renders for loaded games (if applicable)
- ✅ Scenario 4: Fallback message works for no-game state
- ✅ Dashboard metrics visible and updating correctly
- ✅ No console errors in any scenario
- ✅ UI layout remains clean and responsive

### FAIL Criteria (ANY of these is true → FAIL)
- ❌ KG doesn't appear immediately after game start
- ❌ KG disappears after user action
- ❌ Dashboard metrics don't update or show errors
- ❌ Browser console contains error messages
- ❌ Fallback message doesn't appear for no-game state
- ❌ UI breaks or becomes unresponsive

---

## Test Execution Log

### Date: [TO BE FILLED]
### Tester: [TO BE FILLED]
### Environment: Streamlit at http://127.0.0.1:7860

| Scenario | Start Time | End Time | Result | Notes |
|----------|-----------|----------|--------|-------|
| 1. Start Game | | | | |
| 2. First Action | | | | |
| 3. Load Save | | | | |
| 4. No Game | | | | |

### Overall Result
[ ] PASS — All scenarios passed, ready for release
[ ] FAIL — Issues found, needs debugging

### Issues Found (if any)
```
[List any failures or unexpected behaviors here]
```

---

## Sign-Off

**Verified By**: [Name]
**Date**: [Date]
**Status**: [APPROVED / REJECTED]

---

## Notes for Next Steps

If PASS:
- Code is ready for merge to main
- Deploy to production
- No further work needed

If FAIL:
- Document issues in detail
- Create bug report with screenshots
- Review code changes
- Iterate on fix

