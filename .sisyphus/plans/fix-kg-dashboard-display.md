# 修复：Start New Game 后立即显示知识图谱和仪表板

## TL;DR

> **问题**：点击 "Start New Game" 后，知识图谱(KG)和仪表板(Dashboard)不会立即显示，需要用户进行第一次选择才能出现。
> 
> **根本原因**：`st.rerun()` 后，侧边栏在 `kg_html` 尚未被正式保存前执行条件判断，导致跳过渲染。
>
> **方案**：
> 1. 修改 `sidebar.py` 中的 KG 渲染逻辑 — 当 engine 存在时立即渲染，不依赖 `kg_html` state
> 2. 确保 dashboard metrics 在 engine 初始化后立即显示
>
> **Deliverables**:
> - ✅ 修改后的 `src/ui/sections/sidebar.py` — KG 条件判断和渲染逻辑
> - ✅ 验证 "Start New Game" → KG 和 Dashboard 立即显示（无需额外操作）

---

## Context

### 当前行为
1. 用户点击 "🎮 Start New Game"
2. 引擎初始化，生成初始故事和 KG HTML
3. 页面调用 `st.rerun()`
4. 重新运行时，侧边栏执行 `if st.session_state.kg_html or engine:`
5. **问题**：此时 `kg_html` 为空或尚未被正式注册，条件判断为假
6. 只显示 "The knowledge graph will appear after starting a game" 信息
7. 需要用户进行第一次选择（触发 `_process_action`）后，KG 才会显示

### 代码路径
- **主入口**：`app.py:212-242` — "Start New Game" 逻辑
- **侧边栏**：`src/ui/sections/sidebar.py:148-180` — Dashboard 和 KG 渲染
- **仪表板渲染**：`sidebar.py:150-165` — Dashboard metrics

### 根本原因
侧边栏的 KG 渲染条件（第171行）过于保守：
```python
if st.session_state.kg_html or engine:
    if st.session_state.kg_html:
        # 显示 kg_html
    elif engine:
        # 仅显示加载状态 ⚠️ 问题在这里
        st.info("⏳ Knowledge graph is being generated...")
else:
    # 显示"尚未开始"
    st.info("The knowledge graph will appear after starting a game.")
```

**问题**：`elif engine:` 分支只显示加载状态，不进行渲染。

---

## Work Objectives

### Core Objective
修改 `sidebar.py` 中的知识图谱和仪表板渲染逻辑，确保：
- ✅ 当 engine 初始化后立即显示 KG（不必等 `kg_html` state 被持久化）
- ✅ Dashboard metrics 在 engine 初始化时就可见
- ✅ 无需用户额外操作即可看到这两个组件

### Concrete Deliverables
- 修改后的 `src/ui/sections/sidebar.py`（174-178 行的 KG 渲染逻辑）
- Dashboard 立即可用（已在 150-165 行，只需确保 engine 存在时正常显示）

### Definition of Done
✅ 按 "Start New Game" 后，侧边栏立即显示：
  - Dashboard with correct Turns, Entities, Conflicts counts
  - Knowledge graph visualization (not just "loading" message)

✅ No console errors or warnings

✅ Verified with test scenarios (see QA Scenarios)

### Must Have
- When `engine` is initialized, KG visualization renders immediately (even if `kg_html` state not yet persisted)
- Dashboard metrics display on initial page load after game start
- No regression in existing functionality (load game, restore session, etc.)

### Must NOT Have (Guardrails)
- Don't remove or break the fallback to `st.info()` when engine is None
- Don't add expensive API calls in the rendering logic
- Don't change sidebar layout or structure — only the KG/Dashboard rendering flow
- Don't modify `app.py` main logic — only touch `sidebar.py`

---

## Verification Strategy

### Test Infrastructure
- No automated tests required for this UI fix
- Manual QA scenarios will verify the rendering behavior

### Agent-Executed QA Policy
Each task includes interactive browser-based QA using Playwright to verify:
- UI renders correctly after "Start New Game"
- KG visualization appears (specific selectors, not vague checks)
- Dashboard metrics display with correct values
- No console errors or warnings

---

## Execution Strategy

### Sequential Execution (Single Task)
This is a simple, focused fix — one file change, no dependencies.

```
Task 1: Modify sidebar.py KG rendering logic
  └─ Understand current code flow
  └─ Identify and fix the conditional logic
  └─ Verify behavior
  └─ QA scenarios (Playwright)
```

---

## TODOs

- [ ] 1. 修改 `sidebar.py` 中的 KG 渲染逻辑

  **What to do**:
  - 打开 `src/ui/sections/sidebar.py` 第169-180 行 KG 渲染部分
  - 修改第171行的条件判断逻辑：
    - **当前**：`if st.session_state.kg_html or engine:` 后的 `elif engine:` 只显示加载消息
    - **修改**：当 engine 存在时，直接调用 `render_kg_html(engine.kg.graph)` 渲染，而不是等待 `kg_html` state
  - 确保保存 HTML 到 `st.session_state.kg_html` 以供下次重新运行使用
  - 保留对 `kg_html` state 的检查以提高性能（避免重复渲染）

  **具体实现步骤**:
  1. 在侧边栏顶部导入 `render_kg_html`（如果还未导入）
  2. 修改第171-180行逻辑：
     ```python
     if st.session_state.kg_html or engine:
         if st.session_state.kg_html:
             # 显示已缓存的 kg_html
             st.markdown("<div class='kg-frame'>", unsafe_allow_html=True)
             components.html(st.session_state.kg_html, height=480, scrolling=True)
             st.markdown("</div>", unsafe_allow_html=True)
         elif engine:
             # 当 engine 存在但 kg_html 还未缓存时，立即渲染
             kg_html = render_kg_html(engine.kg.graph)
             st.session_state.kg_html = kg_html  # 缓存以供下次使用
             st.markdown("<div class='kg-frame'>", unsafe_allow_html=True)
             components.html(kg_html, height=480, scrolling=True)
             st.markdown("</div>", unsafe_allow_html=True)
     else:
         st.info("The knowledge graph will appear after starting a game.")
     ```
  3. 验证语法无误

  **Must NOT do**:
  - 改动 dashboard metrics 部分（150-165 行已正确）
  - 改动其他侧边栏功能（NLU debug, 一致性趋势等）
  - 改动 `app.py`

  **Recommended Agent Profile**:
  > 这是一个纯粹的 UI 修复任务，涉及一个文件的局部修改。
  - **Category**: `quick` (quick + direct file edit)
    - Reason: Single-file modification, clear logic change, no complex dependencies
  - **Skills**: [] (no special skills needed — straightforward Python Streamlit code)

  **Parallelization**:
  - **Can Run In Parallel**: NO (single task)
  - **Parallel Group**: N/A — this is the only task
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `src/ui/sections/sidebar.py:14` — `render_kg_html` import (verify it's already imported)
  - `src/ui/sections/sidebar.py:169-180` — Current KG rendering logic
  - `app.py:225-229` — Where `engine.start_game()` sets `kg_html` initially
  - `src/knowledge_graph/visualizer.py` — `render_kg_html()` function signature

  **Acceptance Criteria**:
  - [ ] File modified: `src/ui/sections/sidebar.py` lines 169-180
  - [ ] Syntax verified: No Python errors when importing the module
  - [ ] Logic verified: 
    - ✓ When `st.session_state.kg_html` exists → display it (no re-render)
    - ✓ When `st.session_state.kg_html` is None/empty but `engine` exists → render immediately
    - ✓ When `engine` is None → display fallback message

  **QA Scenarios**:

  ```
  Scenario: Start new game → KG renders immediately
    Tool: Playwright
    Preconditions:
      - App is running at http://127.0.0.1:7860
      - No active game session (first load)
    Steps:
      1. Playwright navigates to the app
      2. Wait for genre input to be visible: selector "input[key='genre_input']"
      3. Fill genre input with "fantasy": page.fill("input[key='genre_input']", "fantasy")
      4. Click "🎮 Start New Game" button: page.click("button", has_text="Start New Game")
      5. Wait for spinner to disappear: wait_for_selector(".stSpinner", state="hidden", timeout=15000)
      6. Take screenshot: page.screenshot(path=".sisyphus/evidence/task-1-kg-render.png")
      7. Verify KG frame is visible: 
         - Check CSS class "kg-frame" exists in DOM: page.locator(".kg-frame").is_visible()
         - Check iframe inside the frame contains graph: page.locator(".kg-frame iframe").count() >= 1
      8. Verify dashboard metrics are visible:
         - Check "Turns" metric: page.locator("text='Turns'").is_visible()
         - Check "Entities" metric: page.locator("text='Entities'").is_visible()
         - Check "Conflicts" metric: page.locator("text='Conflicts'").is_visible()
    Expected Result:
      - KG visualization (iframe with graph) is visible in sidebar
      - Dashboard shows correct metrics (Turns: 0, Entities: 2+, Conflicts: 0)
      - No error messages or "loading" state still visible
    Failure Indicators:
      - KG frame is empty or shows "loading" message
      - Dashboard metrics not visible
      - Console contains error messages
    Evidence: .sisyphus/evidence/task-1-kg-render.png

  Scenario: Perform an action → KG updates without breaking
    Tool: Playwright
    Preconditions:
      - Game is running and showing initial story (from previous scenario)
      - KG and Dashboard are visible
    Steps:
      1. Wait for branch options to appear: wait_for_selector("text='Branch Options'", timeout=5000)
      2. Click the first option button: page.click("button >> nth=0")  // First branch option
      3. Wait for spinner to disappear (processing): wait_for_selector(".stSpinner", state="hidden", timeout=20000)
      4. Take screenshot: page.screenshot(path=".sisyphus/evidence/task-1-kg-update.png")
      5. Verify KG still visible and updated:
         - Check KG frame is still visible: page.locator(".kg-frame").is_visible()
         - Verify no error messages appear
      6. Verify dashboard metrics updated:
         - Check "Turns" increased from 0 to 1: page.locator("text='Turns'").first.text_content() contains "1"
    Expected Result:
      - KG visualization remains visible after action
      - Dashboard metrics update correctly (Turns: 1)
      - No console errors
    Failure Indicators:
      - KG disappears after action
      - Dashboard metrics don't update
      - Error messages in console
    Evidence: .sisyphus/evidence/task-1-kg-update.png

  Scenario: Load saved game → KG renders correctly
    Tool: Playwright
    Preconditions:
      - A previous game exists with saves (need at least one save file)
      - App is running
    Steps:
      1. Click on sidebar expander "💾 Save / Load": page.click("button >> text='Save / Load'")
      2. Wait for save slots dropdown to appear: wait_for_selector("select", timeout=5000)
      3. Select the first save slot: page.select_option("select", index=0)
      4. Click "📂 Load Selected Save" button: page.click("button >> text='Load Selected Save'")
      5. Wait for spinner to disappear: wait_for_selector(".stSpinner", state="hidden", timeout=15000)
      6. Take screenshot: page.screenshot(path=".sisyphus/evidence/task-1-kg-load.png")
      7. Verify KG is visible: page.locator(".kg-frame").is_visible()
      8. Verify dashboard shows loaded game state: page.locator("text='Turns'").first.text_content() contains a number > 0
    Expected Result:
      - KG displays correctly after loading save
      - Dashboard shows loaded game's turn count
      - No error messages
    Failure Indicators:
      - KG doesn't appear after load
      - Dashboard metrics are zero or missing
      - Error in console
    Evidence: .sisyphus/evidence/task-1-kg-load.png

  Scenario: No game started → fallback message shows
    Tool: Playwright
    Preconditions:
      - Fresh app load, no active session
      - Not clicked "Start New Game" yet
    Steps:
      1. Navigate to app: page.goto("http://127.0.0.1:7860")
      2. Wait for page to load: page.wait_for_load_state("networkidle")
      3. Take screenshot: page.screenshot(path=".sisyphus/evidence/task-1-no-game.png")
      4. Verify fallback message appears: page.locator("text='The knowledge graph will appear after starting a game'").is_visible()
      5. Verify dashboard is not shown (or shows zero values): page.locator("text='Turns'").count() == 0 OR contains "0"
    Expected Result:
      - Fallback message is displayed (not broken)
      - Dashboard either hidden or showing placeholder
    Failure Indicators:
      - Fallback message is missing or broken
      - Random content shows instead
    Evidence: .sisyphus/evidence/task-1-no-game.png
  ```

  **Evidence to Capture**:
  - [ ] Screenshot after "Start New Game": `task-1-kg-render.png` (KG visible, Dashboard showing)
  - [ ] Screenshot after first action: `task-1-kg-update.png` (KG updated, Turns metric increased)
  - [ ] Screenshot after load save: `task-1-kg-load.png` (KG visible for loaded game)
  - [ ] Screenshot with no game: `task-1-no-game.png` (fallback message visible)

  **Commit**: YES
  - Type: `fix`
  - Scope: `ui/sidebar`
  - Message: `fix(ui/sidebar): render KG immediately when engine initializes, not waiting for kg_html state`
  - Files: `src/ui/sections/sidebar.py`
  - Test command: `python -m pytest tests/ -v -k sidebar` (if tests exist)

---

## Final Verification Wave (after implementation)

- [ ] F1. Code quality check
  - Verify no syntax errors: `python -m py_compile src/ui/sections/sidebar.py`
  - Check for unused imports (if any removed)
  - Verify indentation and style consistency

- [ ] F2. Integration check
  - Verify Streamlit app starts without errors: startup logs clean
  - No warnings related to sidebar rendering in console

- [ ] F3. Manual UI verification (via QA scenarios above)
  - All 4 scenarios pass (Start new game, Perform action, Load save, No game)
  - Screenshots show KG and Dashboard rendering as expected
  - No visual regressions

- [ ] F4. Edge case verification
  - Empty history scenario (no story yet)
  - Multiple save/load cycles
  - Fast consecutive actions (doesn't break KG rendering)

---

## Commit Strategy

```bash
git add src/ui/sections/sidebar.py
git commit -m "fix(ui/sidebar): render KG immediately when engine initializes, not waiting for kg_html state"
```

**Rationale**: 
- Single-file fix targeting the UI rendering logic
- Clear problem (KG not showing after Start New Game) and clear solution (render immediately when engine exists)
- No breaking changes — only improves the UX by rendering sooner

---

## Success Criteria

### Verification Commands
```bash
# 1. Syntax check
python -m py_compile src/ui/sections/sidebar.py

# 2. Import check (no side effects)
python -c "from src.ui.sections.sidebar import render_sidebar; print('✓ Import OK')"

# 3. App startup (manual, via browser at http://127.0.0.1:7860)
# Check:
#   - "Start New Game" → KG visible in sidebar immediately
#   - Dashboard shows Turns/Entities/Conflicts metrics
#   - No error messages in console or browser console
```

### Final Checklist
- [ ] File modified: `src/ui/sections/sidebar.py`
- [ ] Syntax correct: No Python compilation errors
- [ ] KG renders when engine exists (even if `kg_html` state not yet persisted)
- [ ] Dashboard metrics visible on game start
- [ ] All 4 QA scenarios pass (screenshots captured)
- [ ] No regressions in save/load or other sidebar features
- [ ] Commit created with proper message

