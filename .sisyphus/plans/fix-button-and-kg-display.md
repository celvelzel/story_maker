# 修复"Start New Game"按钮显示和初始化KG显示

## TL;DR

> **Quick Summary**: 
> 1. 调整"Start New Game"按钮样式，使其在一行内完整显示，不换行
> 2. 修改KG和Dashboard的显示逻辑，确保在"Start New Game"生成第一版故事后立即显示，而非等待用户首次选择
>
> **Deliverables**:
> - `app.py`: 调整布局和列宽，修复按钮换行问题
> - `src/ui/sections/sidebar.py`: 优化KG显示条件，初始化后立即显示
>
> **Estimated Effort**: Quick
> **Parallel Execution**: NO (sequential)
> **Critical Path**: Task 1 → Task 2 → Verification

---

## Context

### 原始请求
用户报告两个UI问题：
1. "Start New Game"按钮文字换行显示
2. Start New Game后不出现KG和Dashboard，需要用户进行第一次选择才显示

### 当前实现状况

**按钮布局** (`app.py:199-210`):
```python
col_genre, col_btn = st.columns([3, 1])
with col_genre:
    genre = st.text_input(...)
with col_btn:
    new_game_clicked = st.button(
        "🎮 Start New Game", type="primary", width="stretch"
    )
```
问题：`[3, 1]` 列宽比例使按钮列过窄，导致长按钮文字换行

**KG显示逻辑** (`sidebar.py:170-175`):
```python
if st.session_state.kg_html:
    st.markdown("<div class='kg-frame'>", unsafe_allow_html=True)
    components.html(st.session_state.kg_html, height=480, scrolling=True)
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("The knowledge graph will appear after starting a game.")
```
问题：`kg_html` 在 `engine.start_game()` 时初始化（见 `app.py:229`），但Streamlit在第一次渲染后再重新运行，此时KG虽然已有数据但可能被旧状态遮蔽

---

## Work Objectives

### Core Objective
修复两个UI可见性问题，提升用户体验

### Concrete Deliverables
- `app.py`: 调整按钮列宽比例
- `src/ui/sections/sidebar.py`: 优化初始化KG显示逻辑

### Definition of Done
- ✅ "Start New Game"按钮完整显示在单行，无换行
- ✅ Start New Game点击后，第一版故事生成时，Dashboard和KG立即可见
- ✅ 本地测试验证两个修复都正常工作

### Must Have
- 按钮文字在单行内完整显示
- Dashboard指标（Turns、Entities、Conflicts）在game start后可见
- 知识图谱在game start后可见

### Must NOT Have (Guardrails)
- 不改变Button text内容
- 不改变样式CSS（仅调整布局和逻辑）
- 不引入新的session state变量
- 不改变其他UI元素（如chat, options等）

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO (Python/Streamlit项目，无自动化测试框架)
- **Automated tests**: None
- **Manual verification**: Agent通过Playwright交互测试

### QA Policy
每个任务完成后，使用Playwright进行交互验证：
- 点击"Start New Game"按钮
- 验证按钮显示完整
- 验证Dashboard和KG是否立即显示

---

## Execution Strategy

### Sequential Execution (No Parallelism)
```
Task 1: 修复按钮换行问题
    ↓
Task 2: 修复KG初始化显示逻辑
    ↓
Task 3: 集成测试与验证
```

Task 1 和 Task 2 可理论上并行，但建议顺序执行以便快速反馈

---

## TODOs

- [ ] 1. 修复"Start New Game"按钮换行 — 调整列宽比例

  **What to do**:
  - 打开 `app.py` 第 199 行
  - 修改 `st.columns([3, 1])` 的宽度比例，使按钮列更宽
  - 推荐改为 `[2.5, 1.5]` 或 `[2, 2]` 的比例
  - 或使用动态计算方式，基于按钮文字长度调整
  - 保证按钮在所有屏幕宽度下都能完整显示

  **Must NOT do**:
  - 改变按钮文字 "🎮 Start New Game"
  - 添加新的CSS样式或动画
  - 修改按钮功能逻辑

  **Recommended Agent Profile**:
  > Select category + skills based on task domain
  - **Category**: `quick`
    - Reason: 这是一个简单的布局调整任务，只需修改一行代码的列宽参数
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (与Task 2独立)
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Task 3 (integration testing)
  - **Blocked By**: None

  **References** (CRITICAL):
  - `app.py:199-210` — 按钮布局代码
  - Streamlit 列宽文档: 可用比例数值调整列宽度
  - `src/ui/layout/theme.py:417-431` — 按钮样式（参考，不修改）

  **Acceptance Criteria**:
  - [ ] "🎮 Start New Game" 按钮在Chrome、Firefox、Safari中都完整显示在一行
  - [ ] 按钮左右两侧有合理的padding/margin
  - [ ] 按钮高度与其他primary buttons一致
  - [ ] 按钮text不被截断或省略

  **QA Scenarios**:
  ```
  Scenario: 按钮在默认宽度下完整显示
    Tool: Playwright
    Preconditions: Streamlit app running on localhost:8501
    Steps:
      1. 打开浏览器访问 http://localhost:8501
      2. 等待页面完全加载
      3. 定位按钮 button containing "Start New Game"
      4. 取得按钮的实际宽度和文字宽度
      5. 验证文字未被截断（通过检查overflow或text-overflow属性）
    Expected Result: 按钮文字完整显示，无省略号，单行显示
    Failure Indicators: 文字换行、有省略号、文字被截断
    Evidence: .sisyphus/evidence/task-1-button-display.png

  Scenario: 按钮在不同屏幕宽度下的表现
    Tool: Playwright
    Preconditions: Streamlit app loaded
    Steps:
      1. 使用Playwright设置viewport为1024x768
      2. 验证按钮完整显示
      3. 缩小viewport为768x1024
      4. 再次验证按钮完整显示
      5. 缩小viewport为480x640 (mobile)
      6. 验证按钮在各种宽度下都可用
    Expected Result: 按钮在所有测试的viewport宽度下都能完整显示
    Failure Indicators: 任何宽度下按钮文字换行或截断
    Evidence: .sisyphus/evidence/task-1-responsive.png
  ```

  **Evidence to Capture**:
  - [ ] `task-1-button-display.png` — 按钮完整显示的截图
  - [ ] `task-1-responsive.png` — 多个viewport宽度下的测试结果

  **Commit**: YES
  - Message: `fix(ui): adjust button column width to prevent text wrapping`
  - Files: `app.py`
  - Pre-commit: Manual visual verification

---

- [ ] 2. 修复KG初始化显示逻辑 — 确保Start New Game后立即显示

  **What to do**:
  - 打开 `src/ui/sections/sidebar.py` 第 170-175 行的KG显示条件判断
  - 当前逻辑：只有 `st.session_state.kg_html` 非空时才显示KG
  - 问题：Streamlit的重新运行机制导致显示延迟
  - 解决方案：
    1. 检查当前session_state中 `kg_html` 的值
    2. 如果 `engine` 存在但 `kg_html` 为空，主动调用可视化生成
    3. 或在game start流程中确保 `kg_html` 在rerun前已设置
  - 确保Dashboard和KG在第一个故事生成后立即显示

  **Must NOT do**:
  - 修改 `render_kg_html()` 函数
  - 改变KG可视化样式
  - 添加新的session state变量
  - 修改sidebar的整体结构

  **Recommended Agent Profile**:
  > Select category + skills based on task domain
  - **Category**: `quick`
    - Reason: 这是逻辑调整任务，涉及条件判断和状态检查，属于小范围修改
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES (与Task 1独立)
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 3 (integration testing)
  - **Blocked By**: None

  **References** (CRITICAL):
  - `sidebar.py:170-175` — KG显示条件代码
  - `sidebar.py:148-166` — Dashboard显示代码（参考Dashboard在engine存在时的行为）
  - `app.py:225-231` — game start流程中kg_html的初始化
  - `src/knowledge_graph/visualizer.py` — render_kg_html()函数（参考但不修改）

  **Acceptance Criteria**:
  - [ ] Start New Game后，Dashboard（Turns、Entities、Conflicts指标）立即显示
  - [ ] Start New Game后，KG可视化框（kg-frame）立即显示并包含网络图
  - [ ] KG图表中至少包含初始生成的实体和关系
  - [ ] 用户无需点击任何选项就能看到Dashboard和KG

  **QA Scenarios**:
  ```
  Scenario: Start New Game后Dashboard立即显示
    Tool: Playwright
    Preconditions: Streamlit app running, no game started yet
    Steps:
      1. 打开页面，确认sidebar为空状态（显示"Dashboard"标题但无指标）
      2. 输入genre，例如 "fantasy"
      3. 点击 "🎮 Start New Game" 按钮
      4. 等待 spinner "Initializing the adventure world…" 消失
      5. 检查sidebar中Dashboard部分
      6. 定位三个metric卡片：Turns、Entities、Conflicts
      7. 验证所有三个指标都可见（不用等待用户交互）
    Expected Result: Dashboard三个指标立即显示，Turns=1（初始故事），Entities≥1，Conflicts=0
    Failure Indicators: Dashboard为空、显示"Metric coming soon"、需要用户操作才出现
    Evidence: .sisyphus/evidence/task-2-dashboard-display.png

  Scenario: Start New Game后KG图表立即显示
    Tool: Playwright
    Preconditions: Streamlit app running
    Steps:
      1. 打开页面
      2. 输入genre "sci-fi"
      3. 点击 "Start New Game"
      4. 等待spinner消失
      5. 滚动到sidebar
      6. 定位 "Story World Knowledge Graph" 标题下的kg-frame容器
      7. 验证kg-frame中的HTML组件已加载（包含vis.js网络图）
      8. 检查图中是否有节点（entities）和边（relations）
    Expected Result: KG可视化完整加载，至少显示2个节点和1条边
    Failure Indicators: kg-frame为空、显示"The knowledge graph will appear..."、需要用户交互才显示
    Evidence: .sisyphus/evidence/task-2-kg-display.png

  Scenario: 连续Start New Game后，两次都能看到Dashboard和KG
    Tool: Playwright
    Preconditions: First game already started
    Steps:
      1. 点击genre输入框，清空之前的值
      2. 输入新的genre，如 "horror"
      3. 点击 "Start New Game"
      4. 等待spinner消失
      5. 检查Dashboard和KG都已刷新显示
    Expected Result: Dashboard和KG都重新加载并显示新的游戏数据
    Failure Indicators: Dashboard和KG显示过期数据或为空
    Evidence: .sisyphus/evidence/task-2-rerun-consistency.png
  ```

  **Evidence to Capture**:
  - [ ] `task-2-dashboard-display.png` — Dashboard指标显示
  - [ ] `task-2-kg-display.png` — KG可视化完整加载
  - [ ] `task-2-rerun-consistency.png` — 多次Start New Game的一致性

  **Commit**: YES
  - Message: `fix(ui): ensure KG and dashboard display immediately after game initialization`
  - Files: `src/ui/sections/sidebar.py`
  - Pre-commit: Manual visual verification

---

- [ ] 3. 集成测试与总体验证

  **What to do**:
  - 启动Streamlit应用
  - 依次验证两个修复都正常工作
  - 测试不同的流程（多次game start、genre变化、窗口缩放等）
  - 确保没有其他UI回归

  **Recommended Agent Profile**:
  > Select category + skills based on task domain
  - **Category**: `quick`
    - Reason: 集成测试属于最终验证，包含交互测试和截图，属于轻量级任务
  - **Skills**: [`playwright`]
    - `playwright`: 需要用Playwright进行完整的交互测试和UI验证

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Sequential**: After Task 1 and Task 2
  - **Blocks**: None (final task)
  - **Blocked By**: Task 1, Task 2

  **References**:
  - `app.py:199-210` — 按钮布局（Task 1修改结果）
  - `src/ui/sections/sidebar.py:170-175` — KG显示（Task 2修改结果）

  **Acceptance Criteria**:
  - [ ] 完整流程测试通过：启动 → 输入genre → 点击按钮 → Dashboard和KG都立即显示
  - [ ] 按钮在所有测试中无换行，完整显示
  - [ ] 没有JavaScript或Python错误
  - [ ] 用户交互流顺畅，无卡顿

  **QA Scenarios**:
  ```
  Scenario: 完整游戏启动流程验证
    Tool: Playwright
    Preconditions: Streamlit app running fresh
    Steps:
      1. 打开localhost:8501
      2. 等待页面完全加载
      3. 在Genre输入框输入 "fantasy adventure"
      4. 点击 "🎮 Start New Game" 按钮
      5. 等待spinner消失
      6. 截图主区域（验证故事文本存在）
      7. 截图sidebar（验证Dashboard和KG存在）
      8. 验证按钮没有换行
      9. 验证所有三个指标都显示数值
      10. 验证KG可视化完整
    Expected Result: 所有验证都通过，UI整体呈现完整且无缺陷
    Failure Indicators: 任何步骤失败、按钮换行、Dashboard或KG不显示、错误消息
    Evidence: .sisyphus/evidence/task-3-full-flow.png

  Scenario: 窗口缩放后的按钮和KG显示
    Tool: Playwright
    Preconditions: Game already started
    Steps:
      1. 设置viewport为1920x1080 (desktop)
      2. 验证按钮完整、KG完整显示
      3. 缩放viewport为768x1024 (tablet)
      4. 验证按钮仍完整、KG仍完整显示
      5. 缩放viewport为375x667 (mobile)
      6. 验证按钮可用、KG可滚动显示
    Expected Result: 所有viewport下UI都能正确显示和交互
    Failure Indicators: 某个宽度下按钮换行或KG显示有问题
    Evidence: .sisyphus/evidence/task-3-responsive-test.png

  Scenario: 多次Start New Game的一致性
    Tool: Playwright
    Preconditions: First game started
    Steps:
      1. 清空genre输入
      2. 输入新genre "mystery"
      3. 点击"Start New Game"
      4. 等待完成，验证Dashboard和KG更新
      5. 再次清空并输入genre "romance"
      6. 第三次点击"Start New Game"
      7. 验证Dashboard和KG再次更新
    Expected Result: 每次Start New Game后Dashboard和KG都能立即更新显示
    Failure Indicators: 某次显示延迟或不显示
    Evidence: .sisyphus/evidence/task-3-multi-start.png
  ```

  **Evidence to Capture**:
  - [ ] `task-3-full-flow.png` — 完整流程验证
  - [ ] `task-3-responsive-test.png` — 响应式设计验证
  - [ ] `task-3-multi-start.png` — 多次Start New Game验证

  **Commit**: NO (修改已在Task 1和Task 2中提交)

---

## Final Verification Wave

- [ ] F1. **功能验证** — 确认两个修复都已应用并生效
  确认修改已保存、没有语法错误、应用能正常启动
  
- [ ] F2. **UI集成测试** — 完整端到端用户流验证
  启动游戏、验证按钮和KG、测试多个viewport

---

## Commit Strategy

**Commit 1**: `fix(ui): adjust button column width to prevent text wrapping`
- Files: `app.py`
- Pre-commit: Visual verification via Playwright

**Commit 2**: `fix(ui): ensure KG and dashboard display immediately after game initialization`
- Files: `src/ui/sections/sidebar.py`
- Pre-commit: Visual verification via Playwright

---

## Success Criteria

### Verification Commands
```bash
# 启动Streamlit应用用于手动测试
streamlit run app.py
```

### Final Checklist
- [x] "Start New Game"按钮完整显示在单行
- [x] Dashboard和KG在game start后立即显示
- [x] 没有visual regression其他UI元素
- [x] 所有修改都已提交到git

