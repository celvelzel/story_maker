# 改进运行时持久化：区分浏览器刷新 vs 完全重启

## TL;DR

> **快速总结**: 当前系统无法区分 Streamlit 页面刷新与完全重启。修复后：
> - 浏览器刷新（F5）→ **自动恢复** 之前的状态
> - Ctrl+C 后重启 → **不自动恢复**，显示"Load Checkpoint"和"Start New Game"两个选项
> 
> **关键机制**: 在 `runtime_session.json` 中添加"活跃"标记，启动时根据标记判断是否自动恢复。
>
> **Estimated Effort**: Short  
> **Parallel Execution**: NO (顺序执行，影响启动逻辑)  
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4 → Test

---

## Context

### 原始需求（用户描述）

```
当浏览器刷新时，程序应该恢复之前的状态。
但是当我使用 Ctrl+C 关闭程序再重启后，程序应该默认不恢复状态，
只有明确点击"加载存档"才恢复。
```

### 当前实现方式

**文件**: `app.py` 第 139 行
```python
restore_runtime_session(_runtime_save_dir())
```

**行为**: 无条件恢复 `saves/runtime_session.json` 中的所有状态（无区分）

**问题**: 无法区分是浏览器刷新还是完全重启

### 根本原因

Streamlit 的 `st.session_state` 在**浏览器刷新**时**保留**，但在**重启 Python 进程**时**丢失**。
当前代码无论何时都调用 `restore_runtime_session()`，因此无法区分两种场景。

### 技术选择

**区分方案：标记"活跃会话"**

1. 在 `runtime_session.json` 中添加字段: `is_active: boolean`
2. **启动时**: 
   - `is_active == true` → **浏览器刷新场景** → 恢复状态
   - `is_active == false` → **完全重启场景** → 不恢复，显示选项
3. **运行时**: 
   - 游戏进行中 → `is_active = true`（在 `_persist_runtime_session()` 中更新）
   - 主动点击"Start New Game" → `is_active = false`（清空标记）
4. **关闭时**: 
   - Ctrl+C / 进程终止 → `is_active = false`（通过 `atexit` 钩子）

---

## Work Objectives

### 核心目标
区分浏览器刷新与完全重启，根据场景决定是否自动恢复游戏状态。

### 具体目标
1. **启动时**: 根据 `is_active` 标记判断是否自动恢复
2. **运行时**: 每次持久化时更新 `is_active = true`
3. **关闭时**: 进程退出时标记 `is_active = false`
4. **UI 展示**: 
   - 完全重启后 → 显示"Load Checkpoint"和"Start New Game"两个选项
   - 恢复后 → 继续游戏（不显示选项）

### 可交付成果
- [ ] `saves/runtime_session.json` 中有 `is_active` 字段
- [ ] `app.py` 启动逻辑能正确区分两种场景
- [ ] UI 显示正确的选项或继续游戏

### 必要条件
1. 不破坏现有的浏览器刷新恢复功能
2. 不丢失已有的存档数据

### 禁止项（Guardrails）
- ❌ 删除或清空 `runtime_session.json`（应该只修改标记）
- ❌ 改变 `start_game()` 的行为（只改启动逻辑）
- ❌ 影响正常游戏进行中的持久化逻辑

---

## Verification Strategy

### 测试基础设施
- **测试框架**: pytest + 手动集成测试（因为涉及 Streamlit lifecycle）
- **策略**: 不依赖单元测试，手动验证三个场景

### 场景验证
1. **场景 A: 浏览器刷新** (F5)
   - 启动游戏 → 玩两回合 → 按 F5 刷新
   - ✅ 应该看到完整的历史记录 + 之前生成的选项

2. **场景 B: 完全重启（存档存在）**
   - 启动游戏 → 玩两回合 → Ctrl+C 关闭
   - 再次运行 `python app.py` 或 `streamlit run app.py`
   - ✅ 应该看到空白界面 + "Load Checkpoint" 和 "Start New Game" 两个按钮

3. **场景 C: 完全重启后加载存档**
   - 场景 B 的基础上 → 点击 "Load Checkpoint"
   - ✅ 应该恢复之前的游戏状态

### Agent-Executed QA（检查清单）
每项改动完成后，执行的验证：

| 场景 | 验证命令 | 预期结果 |
|-----|--------|---------|
| runtime_session.py 修改 | 手动检查 JSON 结构 | `is_active` 字段存在 |
| app.py 启动逻辑 | 打开浏览器，访问 URL | 正确判断是否恢复 |
| UI 选择逻辑 | 在 Ctrl+C 后重启，查看按钮 | "Load Checkpoint" + "New Game" |

---

## Execution Strategy

### 顺序执行（非并行）
因为每个任务影响启动流程，必须按顺序进行。

```
Task 1: 修改 runtime_session.py
  ↓
Task 2: 更新 app.py 启动逻辑
  ↓
Task 3: 添加 UI 选项组件
  ↓
Task 4: 集成测试
```

---

## TODOs

- [ ] 1. 扩展 runtime_session.json 结构 — 添加生命周期标记

  **What to do**:
  - 修改 `src/engine/runtime_session.py`
  - 在 `save_runtime_session()` 中添加参数 `is_active: bool`（默认 `True`）
  - 更新 `load_runtime_session()` 返回值包含 `is_active` 字段
  - 添加新函数 `mark_session_inactive(save_dir)` 清空标记

  **References**:
  - `src/engine/runtime_session.py:55-61` — 现有 `save_runtime_session()` 函数
  - `src/engine/runtime_session.py:64-76` — 现有 `load_runtime_session()` 函数

  **Acceptance Criteria**:
  - [ ] `runtime_session.json` 包含字段 `"is_active": true`（游戏进行中）
  - [ ] `mark_session_inactive()` 能将 `is_active` 改为 `false`
  - [ ] 现有存档加载时默认 `is_active=false`（安全默认值）

- [ ] 2. 更新 app.py 启动逻辑 — 根据 is_active 决定是否恢复

  **What to do**:
  - 修改 `src/ui/state_manager.py` 的 `restore_runtime_session()` 函数
  - 添加参数 `only_if_active: bool = False`（默认不恢复）
  - 仅当 `is_active == true` 时才加载状态
  - 修改 `app.py` 第 139 行调用：`restore_runtime_session(_runtime_save_dir(), only_if_active=True)`

  **Must NOT do**:
  - 不要改变 `_cleanup_runtime_files()` 的行为
  - 不要删除 `runtime_session.json`（只修改字段）

  **References**:
  - `src/ui/state_manager.py:54-110` — 现有 `restore_runtime_session()` 函数
  - `app.py:139` — 启动时的恢复调用

  **Acceptance Criteria**:
  - [ ] 运行 `streamlit run app.py` 时，检查日志确认是否恢复
  - [ ] 若 `is_active=false`，不恢复任何状态（history/options/kg_html 为空）
  - [ ] 若 `is_active=true`，完整恢复所有字段

- [ ] 3. 添加"加载存档"UI 组件 — 当启动时不恢复时显示

  **What to do**:
  - 在 `app.py` 中添加状态：`show_load_checkpoint_btn`
  - 当启动不恢复时设置 `st.session_state.show_load_checkpoint_btn = True`
  - 在 UI 顶部（genre 输入框和 "Start New Game" 按钮之间）添加条件渲染：
    - 如果有可加载的存档 → 显示 "📂 Load Previous Checkpoint" 按钮
    - 点击后触发 `restore_runtime_session(only_if_active=False)`（强制加载）
  - 成功加载后隐藏按钮

  **References**:
  - `app.py:210-221` — 现有 "Start New Game" 按钮位置
  - `src/engine/runtime_session.py` — 检查存档是否存在的逻辑

  **Acceptance Criteria**:
  - [ ] Ctrl+C 后重启时，显示 "📂 Load Previous Checkpoint" 按钮
  - [ ] 点击按钮后正确加载存档
  - [ ] 加载后隐藏按钮，游戏继续进行

- [ ] 4. 更新关闭钩子 — 进程退出时标记 is_active=false

  **What to do**:
  - 修改 `app.py` 中的 `_cleanup_runtime_files()` 函数
  - 在删除文件之前，先调用 `mark_session_inactive(save_dir)` 标记为不活跃
  - 确保 SIGINT 和 atexit 钩子都调用这个新逻辑

  **References**:
  - `app.py:102-118` — 现有 `_cleanup_runtime_files()` 和关闭钩子

  **Acceptance Criteria**:
  - [ ] Ctrl+C 后，`runtime_session.json` 中 `is_active` 改为 `false`
  - [ ] 文件仍然存在（未被删除）
  - [ ] 重启时不自动恢复

- [ ] 5. 集成测试 — 验证三个场景

  **What to do**:
  - 手动测试（因为 Streamlit lifecycle 难以单元测试）：
    1. 启动游戏 → 玩两回合 → 浏览器 F5 → 验证恢复
    2. 启动游戏 → 玩两回合 → Ctrl+C 关闭 → 再启动 → 验证不恢复，显示按钮
    3. 点击 "Load Checkpoint" → 验证恢复

  **Acceptance Criteria**:
  - [ ] 场景 A 通过：浏览器刷新后完整恢复（history、options、kg_html）
  - [ ] 场景 B 通过：Ctrl+C 重启后看到空白界面 + "Load Checkpoint" 按钮
  - [ ] 场景 C 通过：点击加载后恢复之前的游戏状态

---

## Final Verification Wave

- [ ] **Code Quality Review** — 检查逻辑一致性
  - `is_active` 字段更新时机正确
  - 没有硬编码值，所有 magic strings 已提取
  - 日志记录充分（便于调试）

- [ ] **Backward Compatibility** — 检查现有存档
  - 旧的 `runtime_session.json`（无 `is_active` 字段）加载时安全默认为 `false`
  - 不会因为丢失字段而报错

---

## Success Criteria

### 功能验证
- ✅ 浏览器刷新后自动恢复
- ✅ Ctrl+C 重启后不自动恢复
- ✅ 用户能手动加载存档
- ✅ 点击"Start New Game"开始新游戏

### 代码质量
- ✅ 无 `as any` / `@ts-ignore`（如果有 TS）
- ✅ 错误处理健壮（如存档损坏）
- ✅ 日志清晰，便于调试

