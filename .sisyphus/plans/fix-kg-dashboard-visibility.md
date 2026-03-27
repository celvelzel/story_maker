# Fix KG and Dashboard Visibility Issue

## TL;DR

> **Quick Summary**: Fix a frontend bug where the Knowledge Graph and dashboard do not appear immediately after clicking "Start New Game" by correcting the rendering order in Streamlit.
> 
> **Deliverables**: 
> - Updated `app.py` to render the sidebar after state mutation logic
> 
> **Estimated Effort**: Quick
> **Parallel Execution**: NO - sequential
> **Critical Path**: Task 1

---

## Context

### Original Request
Fix the bug where the Knowledge Graph and dashboard do not appear immediately after clicking "Start New Game" (they only appear after the first user choice). The user wants them to appear immediately after the first story text is generated.

### Interview Summary
**Key Discussions**:
- The bug is a Streamlit rendering lifecycle issue. `render_sidebar()` in `app.py` is called at line 194, before the "Start New Game" button click handler. 
- During the 10-15s generation time, the sidebar shows the old (empty) state because the UI renders before the state mutations.
- Even after `st.rerun()`, Streamlit sometimes retains the state of components rendered before a button click handler within the same execution path. Moving the rendering function to the bottom ensures it always accesses the most updated `st.session_state`.

### Metis Review
**Identified Gaps**:
- Need to ensure no duplicate renders, flicker, or empty sidebars occur.
- Must preserve current sidebar content and styling without adding new UI behaviors.

---

## Work Objectives

### Core Objective
Ensure that the sidebar containing the Knowledge Graph and Dashboard updates and renders immediately after the "Start New Game" generation finishes.

### Concrete Deliverables
- Modify `app.py` so that `render_sidebar()` is called at the end of the script instead of the middle.

### Definition of Done
- [ ] `render_sidebar()` is successfully moved in `app.py` and the sidebar updates reliably on game start.

### Must Have
- The sidebar render call MUST be placed at the end of the script (e.g., after `render_evaluation()`).

### Must NOT Have (Guardrails)
- Do not modify the game initialization logic in `src/engine/game_engine.py`.
- Do not modify `src/ui/sections/sidebar.py`.

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES
- **Automated tests**: None (UI visual change, difficult to test with simple unit tests)
- **Agent-Executed QA**: MANDATORY.

### QA Policy
The executing agent will verify the fix by running the Streamlit app and simulating the user interaction.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1:
├── Task 1: Move render_sidebar() call in app.py [quick]

Wave FINAL:
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
```

---

## TODOs

- [x] 1. Move `render_sidebar()` in `app.py`

  **What to do**:
  - In `app.py`, locate the call to `render_sidebar()` around line 194.
  - Cut this line.
  - Paste it at the very bottom of the file (e.g., around line 292, just after the `render_evaluation()` call or `render_chat_input()`).
  - Ensure the indentation is at the top level.

  **Must NOT do**:
  - Do not change the internal logic of `render_sidebar()` itself.
  - Do not remove the `st.rerun()` from the "Start New Game" button block.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file logic order change.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocked By**: None

  **References**:
  - `app.py:194` - The current location of `render_sidebar()`
  - `app.py:291` - The target location (end of file)

  **Acceptance Criteria**:
  - [x] `render_sidebar()` is at the bottom of `app.py`.
  - [x] `render_sidebar()` is no longer present in the middle of the file.

  **QA Scenarios**:
  ```
  Scenario: Verify render_sidebar position
    Tool: Bash (grep/cat)
    Preconditions: None
    Steps:
      1. run `tail -n 10 app.py`
      2. Verify that `render_sidebar()` appears near the end of the output.
      3. run `sed -n '180,200p' app.py`
      4. Verify `render_sidebar()` is NOT present in the middle section.
    Expected Result: The render call is correctly placed at the bottom.
    Failure Indicators: `render_sidebar()` is found in the middle of the file.
    Evidence: .sisyphus/evidence/task-1-position.txt
  ```

---

## Final Verification Wave

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": verify compliance.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `flake8` or linter if available. Review changed files for AI slop and syntax correctness.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Run `streamlit run app.py` (mocked or headless if possible) or verify the script logic. Since testing Streamlit headless is complex, carefully inspect the updated `app.py` logic to ensure `render_sidebar()` appears sequentially after all `st.button` and `st.rerun()` handlers.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  Verify 1:1 — everything in spec was built, nothing beyond spec was built.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **1**: `fix(ui): update sidebar rendering order to display KG on game start` — app.py

---

## Success Criteria

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] `app.py` correctly modified
