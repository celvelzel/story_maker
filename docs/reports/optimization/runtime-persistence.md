# 运行时持久化报告

**日期**：2026-03-31  
**范围**：浏览器刷新时的会话持久化；仅在终端停止（Ctrl+C）时清理运行时数据

## 1. 问题与目标

### 原有问题
Streamlit 的 `st.session_state` 仅在当前浏览器会话中有效。刷新页面会重置会话，导致：
- 当前故事对话消失
- 知识图谱可视化重置
- 评估指标和生成的选项丢失

### 目标
- 浏览器刷新后自动恢复当前会话（故事、KG、选项）
- 仅在终端通过 `Ctrl+C` 停止服务时删除运行时会话文件
- 保持现有 `saves/` 目录的永久快照功能，不自动删除
- 确保 `saves/` 继续被 Git 忽略

## 2. 实现细节

### 2.1 运行时会话持久化模块
**新文件**：`src/engine/runtime_session.py`

**核心功能**：
- **路径管理**：处理 `runtime_session.json` 和 `runtime_engine.json`
- **序列化**：支持 `StoryOption <-> dict` 转换，用于复杂 UI 状态
- **I/O 操作**：提供 `save_runtime_session(...)` 和 `load_runtime_session(...)`
- **清理**：`remove_runtime_files(...)` 专门针对运行时产物

### 2.2 Streamlit 应用集成
**修改文件**：`app.py`

**核心逻辑**：
- `_persist_runtime_session()`：保存引擎状态和 UI 相关状态（聊天历史、KG HTML、选项、评估数据）
- `_restore_runtime_session_once()`：应用启动时尝试恢复。先加载引擎状态，再填充 `st.session_state`
- `_cleanup_runtime_files()` + `_register_runtime_cleanup()`：使用 `atexit` 和 `SIGINT`（Ctrl+C）处理器，确保仅在服务器关闭时清理

**持久化触发时机**：
- 点击"开始新游戏"后
- 每次行动处理周期（`_process_action`）后
- 运行评估后（确保结果在刷新后保留）

## 3. Git 追踪策略
`.gitignore` 已包含 `saves/` 目录。运行时文件和标准保存文件均不被 Git 追踪，保持仓库整洁的同时允许本地持久化。

## 4. 测试与验证

### 4.1 新增单元测试
**新文件**：`tests/test_runtime_session.py`

**覆盖范围**：
- 正确的运行时文件路径生成
- 运行时元数据保存/加载完整性
- `StoryOption` 序列化/反序列化
- 运行时文件的定向删除（确保永久保存不受影响）

### 4.2 回归测试
针对现有持久化测试进行验证：
- `tests/test_kg_persistence.py`

### 4.3 结果
- `pytest tests/test_runtime_session.py -q` → **5 通过**
- `pytest tests/test_kg_persistence.py -q` → **15 通过**

## 5. 行为总结
- **浏览器刷新**：自动恢复活跃会话和 KG
- **终端停止（Ctrl+C）**：仅删除 `runtime_session.json` 和 `runtime_engine.json`
- **手动/自动保存**：`saves/` 中的永久快照保持不变

## 6. 受影响文件
- `app.py`：集成恢复、持久化和清理逻辑
- `src/engine/runtime_session.py`：核心持久化逻辑
- `tests/test_runtime_session.py`：验证套件
