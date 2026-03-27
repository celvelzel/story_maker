# 工作计划：存档命名系统重构

## TL;DR

> **目标**: 将游戏存档命名从时间戳方式（`fantasy_latest.json`）改为 **LLM生成的语义摘要 + 模型名 + 时间戳**（`magical-adventure_gpt-4o-mini_2026-03-26-143045.json`）
>
> **核心价值**: 使存档文件名自描述，便于快速识别存档内容
>
> **交付件**:
> - 新的存档命名函数：`generate_archive_name()`
> - 集成到 `GameEngine.save_game()` 中
> - 返回新命名规则和存档路径
>
> **预期工作量**: 快速 (2-3小时)
> **并行执行**: 否，顺序任务

---

## 背景

### 现状问题
- 存档文件名不可读：`fantasy_turn_20.json` 无法看出故事内容
- 命名缺乏元数据：无法追踪使用的模型、生成时间精度低
- 难以管理：多个存档难以区分

### 改进方案
利用系统已有的 **NLG LLM 模块** 在故事初始化后，生成存档命名：

```
格式: {summary}_{model}_{timestamp}.json
示例: magical-adventure_gpt-4o-mini_2026-03-26-143045.json
     └─ 3-5词英文摘要 └─ 使用的模型 └─ ISO8601时间戳
```

---

## 技术决策已确认

| 项目 | 决策 |
|------|------|
| **摘要来源** | 调用NLG模块使用的LLM生成 (gpt-4o-mini) |
| **摘要长度** | 3-5个英文单词 |
| **摘要语言** | 英文 (避免文件名编码问题) |
| **模型信息** | 从 `config.settings.OPENAI_MODEL` 获取 |
| **时间戳格式** | `YYYY-MM-DD-HHMMSS` |
| **失败回退** | LLM调用失败 → 使用传统时间戳命名 |
| **应用范围** | 仅新存档，不重命名现有文件 |
| **集成点** | `GameEngine.save_game()` 中 |
| **返回时机** | `engine.start_game()` 返回后可用 |

---

## 架构

### 数据流

```
app.py: engine.start_game() 
   ↓ (故事生成)
GameEngine.start_game()
   ↓ (获取开场故事)
StoryGenerator.generate_opening()  ← LLM调用
   ↓ (故事文本返回)
GameEngine._auto_save() 或 save_game()
   ↓ (调用新的命名函数)
generate_archive_name(story_text, model_name, genre)
   ├─ 调用LLM生成摘要 (async/快速)
   ├─ 获取时间戳
   ├─ 组装文件名
   └─ 失败 → 回退到 {genre}_latest.json
   ↓
save_game(filepath=new_filename)  ← 使用新名称保存
```

### 关键模块

1. **新函数**: `src/engine/naming.py::generate_archive_name()`
   - 输入：故事文本片段、模型名、游戏类型
   - 输出：`{summary}_{model}_{timestamp}.json`
   - 异常处理：失败 → 默认名称

2. **修改文件**: `src/engine/game_engine.py`
   - `save_game()` 方法中集成命名函数
   - 检测新游戏初始化时生成特殊命名
   - 保留自动保存快照的旧命名方式 (仅最新和定期快照)

3. **获取模型信息**: 从 `config.settings.OPENAI_MODEL`
   - 不需要修改 api_client，模型名已在config中

---

## TODOs

- [x] 1. 创建 `src/engine/naming.py` — LLM摘要生成 + 命名函数

  **What to do**:
  - 实现 `generate_archive_name(story_text: str, model_name: str, genre: str) -> str`
  - 调用LLM生成3-5词的英文摘要（从故事前500字符）
  - 获取当前ISO8601时间戳
  - 组装命名：`{summary}_{model}_{timestamp}.json`
  - 异常处理：失败返回 `{genre}_auto_{timestamp}.json`

  **Must NOT do**:
  - 使用中文字符（文件名编码问题）
  - 调用多次LLM（浪费token）
  - 缓存摘要（每个存档独特）

  **Recommended Agent Profile**:
  > 这是一个快速的新函数添加任务，涉及LLM集成
  - **Category**: `quick`
    - Reason: 新增单个文件，逻辑直接，无复杂依赖
  - **Skills**: [`lsp-navigate`, `code-completion`]
    - `lsp-navigate`: 快速查找 config.settings.OPENAI_MODEL 位置
    - `code-completion`: 生成标准的异常处理代码

  **Parallelization**:
  - **Can Run In Parallel**: YES (与Task 2独立)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 2, Task 3
  - **Blocked By**: None

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `src/utils/api_client.py:108-169` - LLM调用模式、重试机制、异常处理
  - `config.py` - 获取 `settings.OPENAI_MODEL`, `OPENAI_TEMPERATURE`, `OPENAI_MAX_TOKENS`

  **API/Type References** (contracts to implement against):
  - `src/engine/game_engine.py:620-634` - save_game返回的game_data结构

  **External References** (libraries and frameworks):
  - `datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")` - 时间戳格式

  **WHY Each Reference Matters**:
  - `api_client.py` 告诉你如何调用LLM、如何处理异常 → 复制该模式到naming.py
  - `config.py` 告诉你如何获取模型名称 → 从settings.OPENAI_MODEL读取
  - `game_engine.py` 告诉你save_game的调用上下文 → 知道你的函数会被何时调用

  **Acceptance Criteria**:

  **Code Quality**:
  - [ ] 代码通过 `bun run lint` (无flake8/pyright错误)
  - [ ] 函数签名清晰，有类型注解
  - [ ] 文档字符串完整（英文）

  **QA Scenarios** (MANDATORY):

  ```
  Scenario: 正常路径 — LLM成功生成摘要
    Tool: Bash (Python REPL)
    Preconditions:
      - 已导入 src/engine/naming.py
      - LLM API可用（OPENAI_API_KEY已配置）
    Steps:
      1. result = generate_archive_name(
           story_text="You enter a magical forest. Creatures surround you.",
           model_name="gpt-4o-mini",
           genre="fantasy"
         )
      2. assert result.endswith(".json")
      3. assert "_gpt-4o-mini_" in result
      4. parts = result.replace(".json", "").split("_")
      5. assert len(parts) >= 3  # summary, model, timestamp
      6. assert len(parts[0].split("-")) <= 5  # 摘要最多5词
    Expected Result: 返回格式为 "{summary}_{model}_{timestamp}.json" 的字符串
    Failure Indicators: 返回值格式不符、摘要过长、未包含模型名
    Evidence: .sisyphus/evidence/task-1-normal-path.txt

  Scenario: 失败回退 — LLM调用失败时
    Tool: Bash (Python + mock)
    Preconditions:
      - OPENAI_API_KEY设为无效值 或 mock掉llm_client
      - 模拟LLM异常
    Steps:
      1. 模拟 llm_client.chat() 抛出异常
      2. result = generate_archive_name(story_text="...", model_name="...", genre="fantasy")
      3. assert result.startswith("fantasy_auto_")
      4. assert result.endswith(".json")
    Expected Result: 返回回退命名 "fantasy_auto_{timestamp}.json"
    Failure Indicators: 抛出异常、返回None、格式不符
    Evidence: .sisyphus/evidence/task-1-fallback.txt
  ```

  **Commit**: YES (with Task 2)
  - Message: `feat(engine): add semantic archive naming with LLM summaries`
  - Files: `src/engine/naming.py`
  - Pre-commit: `python -m py_compile src/engine/naming.py`

---

- [x] 2. 修改 `src/engine/game_engine.py` — 集成新命名函数

  **What to do**:
  - 在 `save_game()` 方法中调用 `generate_archive_name()`
  - 检测是否为新游戏初始化 (turn_id == 0 且 story_history 长度为1)
  - 仅在新游戏时使用语义命名，其他情况保留旧命名
  - `start_game()` 返回时记录新存档名称

  **Must NOT do**:
  - 重命名已有的存档
  - 修改自动保存快照的命名逻辑 (保持 `{genre}_turn_{turn_id}.json`)
  - 改变 save_game() 的返回值类型

  **Recommended Agent Profile**:
  > 这是一个在现有代码中集成新函数的任务，涉及小范围修改
  - **Category**: `quick`
    - Reason: 修改1个文件，3-4处地方，集成已有的函数
  - **Skills**: [`lsp-navigate`, `code-completion`]

  **Parallelization**:
  - **Can Run In Parallel**: NO (依赖Task 1)
  - **Blocks**: Task 3
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `src/engine/game_engine.py:598-640` - save_game() 完整实现
  - `src/engine/game_engine.py:117-150` - start_game() 初始化逻辑
  - `src/engine/state.py` - GameState.turn_id, story_history 属性

  **API/Type References**:
  - `src/engine/naming.py::generate_archive_name()` - 新函数签名和返回值

  **WHY Each Reference Matters**:
  - `save_game()` 是修改的核心位置
  - `start_game()` 告诉你何时turn_id==0
  - `state.py` 告诉你如何检测新游戏

  **Acceptance Criteria**:

  **Code**:
  - [ ] game_engine.py 导入 generate_archive_name
  - [ ] save_game() 调用新函数（仅在turn_id==0时）
  - [ ] 非新游戏时保持旧命名
  - [ ] 代码通过 linter

  **QA Scenarios**:

  ```
  Scenario: 新游戏初始化 — 使用语义命名
    Tool: Bash (Python REPL)
    Preconditions:
      - 创建新的 GameEngine("fantasy")
      - 调用 engine.start_game()
    Steps:
      1. engine = GameEngine("fantasy")
      2. result = engine.start_game()
      3. save_path = engine.save_game()
      4. assert "_gpt-4o-mini_" in save_path or "_auto_" in save_path
      5. assert save_path.endswith(".json")
    Expected Result: save_game() 返回包含语义摘要或自动回退命名的路径
    Failure Indicators: 返回旧命名格式、路径格式不符
    Evidence: .sisyphus/evidence/task-2-new-game.txt

  Scenario: 非新游戏 — 保持旧命名
    Tool: Bash (Python REPL)
    Preconditions:
      - 加载已存在的游戏 engine.load_game(...)
      - turn_id > 0
    Steps:
      1. engine.load_game("saves/fantasy_latest.json")
      2. result = engine.process_turn("walk north")
      3. save_path = engine.save_game()
      4. assert "_latest.json" in save_path or f"_turn_{engine.state.turn_id}" in save_path
    Expected Result: 使用旧命名格式保存
    Failure Indicators: 使用了新的语义命名
    Evidence: .sisyphus/evidence/task-2-load-game.txt
  ```

  **Commit**: YES (with Task 1)
  - Message: `feat(engine): add semantic archive naming with LLM summaries`
  - Files: `src/engine/game_engine.py`
  - Pre-commit: `bun run lint src/engine/game_engine.py`

---

- [x] 3. 更新 `app.py` — 显示新存档名称给用户

  **What to do**:
  - 在游戏开始后，显示新生成的存档名称
  - 将存档名称存入 st.session_state (e.g., `st.session_state.archive_filename`)
  - 在侧边栏或游戏信息区显示 "Archive: {filename}"

  **Must NOT do**:
  - 修改游戏逻辑或状态管理
  - 改变保存位置

  **Recommended Agent Profile**:
  > 这是一个UI展示任务，涉及Streamlit组件
  - **Category**: `visual-engineering`
    - Reason: 涉及UI显示，使用Streamlit组件
  - **Skills**: [`ui-component-patterns`, `streamlit-api`]
    - `ui-component-patterns`: 了解如何在Streamlit中显示信息
    - `streamlit-api`: 快速查找Streamlit的显示函数

  **Parallelization**:
  - **Can Run In Parallel**: NO (依赖Task 2)
  - **Blocks**: Final Verification
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `app.py:224-263` - 新游戏启动代码
  - `app.py:170-220` - 侧边栏UI代码

  **API/Type References**:
  - `src/engine/game_engine.py::save_game()` - 返回路径字符串

  **WHY Each Reference Matters**:
  - `app.py:224-263` 是获取存档名称的位置
  - `app.py:170-220` 告诉你在哪里显示

  **Acceptance Criteria**:

  **Code**:
  - [ ] app.py 在新游戏初始化后调用 save_game() 获取路径
  - [ ] 存储filename到session_state
  - [ ] 在UI中显示存档名称

  **QA Scenarios**:

  ```
  Scenario: 新游戏启动时显示存档名称
    Tool: Playwright (frontend)
    Preconditions:
      - Streamlit app 已启动
      - 侧边栏可见
    Steps:
      1. 输入 "fantasy" 到Genre输入框
      2. 点击 "🎮 Start New Game" 按钮
      3. 等待游戏加载完成
      4. 查找包含 "Archive:" 的文本元素
      5. 验证显示的文件名格式为 "{summary}_{model}_{timestamp}.json"
    Expected Result: 侧边栏或信息区显示 "Archive: {filename}"
    Failure Indicators: 未显示存档名称、格式不符、位置隐藏
    Evidence: .sisyphus/evidence/task-3-display-archive.png

  Scenario: 游戏继续时存档名称保持不变
    Tool: Playwright (frontend)
    Preconditions:
      - 已启动新游戏
      - 显示的存档名称已记录 (e.g., "magical-adventure_gpt-4o-mini_...")
    Steps:
      1. 点击一个选项继续游戏
      2. 等待新回合生成
      3. 查看侧边栏存档名称
      4. 验证名称未变化
    Expected Result: 存档名称保持初始值
    Failure Indicators: 名称被改变或消失
    Evidence: .sisyphus/evidence/task-3-archive-persist.png
  ```

  **Commit**: YES
  - Message: `ui(app): display semantic archive filename`
  - Files: `app.py`
  - Pre-commit: `bun run lint app.py`

---

## 最终验证 Wave (MANDATORY — after all tasks complete)

- [x] F1. **功能集成验证** — 完整工作流

  **What to do**:
  1. 启动Streamlit应用
  2. 开始新游戏 (输入Genre，点击开始)
  3. 验证：
     - 存档文件在 `saves/` 中创建
     - 文件名格式为 `{summary}_{model}_{timestamp}.json`
     - 摘要为3-5个英文单词
     - 模型名正确 (gpt-4o-mini)
     - UI显示存档名称
  4. 继续游戏多个回合
  5. 验证：
     - 最新状态存档仍为新命名
     - 快照存档保持旧命名 (`{genre}_turn_{id}.json`)

  **Tool**: Playwright + Bash
  **Evidence**: .sisyphus/evidence/final-integration.txt, screenshot

---

## 验证命令

```bash
# 检查新文件存在
ls saves/ | grep -E "[a-z]+-[a-z]+_gpt-4o-mini_[0-9]+-[0-9]+-[0-9]+.json

# 验证JSON结构完整
python -c "import json; json.load(open('saves/LATEST_ARCHIVE.json'))"

# 检查linter
bun run lint src/engine/naming.py src/engine/game_engine.py app.py
```

---

## 成功标准

- ✅ 新游戏的初始存档使用语义命名
- ✅ 快照和自动保存保持旧命名
- ✅ LLM调用失败时回退到默认命名
- ✅ UI显示新存档名称
- ✅ 所有代码通过linter
- ✅ 无breaking changes

---

## 提交策略

- **Commit 1**: Tasks 1 + 2
  - Message: `feat(engine): add semantic archive naming with LLM summaries`
  - Files: `src/engine/naming.py`, `src/engine/game_engine.py`

- **Commit 2**: Task 3
  - Message: `ui(app): display semantic archive filename`
  - Files: `app.py`

---

## 风险与缓解

| 风险 | 缓解措施 |
|------|--------|
| LLM调用成本增加 | 仅新游戏初始化时调用，非每回合 |
| LLM调用失败 | 自动回退到时间戳命名 |
| 文件名特殊字符问题 | 仅使用英文和连字符 |
| 现有存档兼容性 | 不修改已有文件，仅新存档应用 |

