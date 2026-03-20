# 浏览器刷新持久化改造更新报告

日期：2026-03-20  
范围：实现“浏览器刷新不丢失会话；仅在终端 Ctrl+C 停止程序时清理运行时数据”

## 1. 问题与目标

### 原问题
- Streamlit 的 `st.session_state` 仅在当前浏览器会话有效；刷新页面后会话重建，导致：
  - 当前对话故事消失
  - 知识图谱显示消失

### 目标
- 浏览器刷新后自动恢复当前会话（故事对话 + KG + 选项等）
- 仅在终端使用 `Ctrl+C` 停止服务时，删除运行时会话文件
- 保留已有 `saves/` 的普通存档能力，不删除历史快照
- `saves/` 目录继续保持不纳入 Git 跟踪

## 2. 实施内容

### 2.1 新增运行时会话持久化模块
新增文件：`src/engine/runtime_session.py`

提供能力：
- 运行时元数据路径管理：
  - `runtime_session.json`
  - `runtime_engine.json`
- 选项序列化/反序列化：`StoryOption <-> dict`
- 运行时元数据读写：
  - `save_runtime_session(...)`
  - `load_runtime_session(...)`
- 运行时文件删除：
  - `remove_runtime_files(...)`（只删除上述两个运行时文件）

### 2.2 在 Streamlit 应用中接入恢复与清理
修改文件：`app.py`

新增核心逻辑：
- `_persist_runtime_session()`
  - 保存引擎状态到 `runtime_engine.json`
  - 保存 UI 相关状态到 `runtime_session.json`（对话历史、KG HTML、选项、评估信息等）
- `_restore_runtime_session_once()`
  - 页面启动时自动尝试恢复
  - 先加载 `runtime_session.json`，再通过 `engine.load_game(runtime_engine.json)` 恢复引擎态
  - 回填 `st.session_state` 的聊天/KG/选项等显示字段
- `_cleanup_runtime_files()` + `_register_runtime_cleanup()`
  - 使用 `atexit` 与 `SIGINT`（Ctrl+C）处理清理
  - 只删除运行时文件，不影响 `fantasy_latest.json` 等常规存档

触发持久化时机：
- 点击 `Start New Game` 后
- 每次 `_process_action(...)` 回合处理后
- 每次运行评估后（确保评估结果也可刷新恢复）

## 3. Git 跟踪策略确认

- 已检查 `.gitignore`，其中已包含 `saves/` 忽略规则（`saves/`）。
- 本次未修改该规则；因此运行时文件与普通存档文件都不会被 Git 跟踪。

## 4. 测试与验证

### 4.1 新增测试
新增文件：`tests/test_runtime_session.py`

覆盖点：
- 运行时文件路径命名正确
- 运行时元数据保存/加载
- `StoryOption` 序列化与反序列化
- 非法选项数据过滤
- 运行时文件删除（仅运行时文件）

### 4.2 回归测试
执行了与持久化相关的既有测试：
- `tests/test_kg_persistence.py`

### 4.3 测试结果
- `pytest tests/test_runtime_session.py -q` → **5 passed**
- `pytest tests/test_kg_persistence.py -q` → **15 passed**

## 5. 行为变化总结

改造后：
- 浏览器刷新：会自动恢复上次进行中的会话内容与 KG 展示
- 终端 `Ctrl+C` 停止：仅删除运行时会话文件（`runtime_session.json`、`runtime_engine.json`）
- 历史快照与普通自动保存文件保留不变

## 6. 影响文件清单

- `app.py`（新增恢复/持久化/清理接入）
- `src/engine/runtime_session.py`（新增）
- `tests/test_runtime_session.py`（新增）
