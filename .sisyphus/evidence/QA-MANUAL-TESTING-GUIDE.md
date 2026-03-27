# 手动 QA 测试执行指南

## 前置条件

确保以下条件已满足：
1. ✓ Python 环境已激活 (`.venv` 或系统 Python)
2. ✓ `.env` 文件已配置 (API key 已设置)
3. ✓ 项目依赖已安装 (`pip install -r requirements.txt`)

## 启动应用程序

### 步骤 1: 启动 Streamlit 应用

打开终端，运行：

```bash
cd C:\Develop\python_projects\COMP5423_NLP\story_maker
streamlit run app.py --logger.level=debug
```

**预期输出**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:7860
  Network URL: http://<your-ip>:7860

  For better performance, install Watchdog.
```

**打开浏览器** 访问: `http://127.0.0.1:7860`

---

## 场景 1: "Start New Game" → KG 立即呈现

### 执行步骤

1. **等待页面加载**
   - 看到 "Genre" 输入框
   - 看到 "Start New Game" 按钮

2. **填充流派输入**
   - 在输入框中输入: `fantasy`

3. **点击 "Start New Game" 按钮**
   - 绿色按钮，文字为 "🎮 Start New Game"
   - 点击它

4. **观察加载动画**
   - 看到圆形加载指示器
   - 等待加载完成（通常 5-15 秒）

### 验证清单

当加载完成后，检查以下内容：

- [ ] **左侧边栏中是否可见知识图谱（KG）？**
  - 应该看到一个网络图可视化（节点和连接）
  - 不应该看到 "⏳ Knowledge graph is being generated..." 信息
  - 不应该看到 "The knowledge graph will appear after starting a game" 信息
  
- [ ] **左侧边栏中是否可见仪表板（Dashboard）？**
  - 标题: "🎛️ Dashboard"
  - 应该看到三个指标:
    - Turns: 0
    - Entities: 2 或更多
    - Conflicts: 0

- [ ] **主区域是否显示故事？**
  - 看到初始故事文本
  - 看到 "Branch Options" 部分
  - 看到 3 个按钮（1. 2. 3.）

- [ ] **浏览器控制台中是否有错误？**
  - 按 F12 打开开发工具
  - 查看 "Console" 标签
  - 不应该有红色错误信息

### 成功标志

✓ **PASS** 如果：
- KG 可视化立即显示（不是加载消息）
- Dashboard 显示正确的指标
- 故事内容显示
- 没有控制台错误

✗ **FAIL** 如果：
- KG 显示加载消息而不是实际的图
- Dashboard 指标缺失或显示错误
- 控制台有错误信息

### 截图

**截图名称**: `qa-scenario1-kg-display.png`

**如何截图**:
1. 按 Windows Key + Shift + S
2. 选择整个浏览器窗口
3. 保存到: `.sisyphus/evidence/qa-scenario1-kg-display.png`

---

## 场景 2: 第一步操作 → KG 继续工作

### 执行步骤

1. **前置条件**：完成场景 1
   - 游戏已启动
   - KG 和 Dashboard 可见

2. **等待 "Branch Options" 出现**
   - 看到文字 "🧭 Branch Options"
   - 看到 3 个按钮，例如：
     - "1. Option A"
     - "2. Option B"
     - "3. Option C"

3. **点击第一个选项按钮**
   - 点击标号为 "1." 的按钮
   - 选择任何看起来合理的选项

4. **等待处理完成**
   - 看到加载指示器
   - 故事继续进展
   - 新的选项出现

### 验证清单

- [ ] **KG 是否仍然可见？**
  - 左侧边栏中仍然显示网络图
  - 不应该消失或被隐藏

- [ ] **Dashboard 指标是否更新？**
  - Turns 应该从 0 变成 1
  - Entities 可能保持相同或增加
  - Conflicts 可能是 0 或增加

- [ ] **故事是否继续进展？**
  - 看到新的故事文本
  - 看到新的选项

- [ ] **是否有错误？**
  - 控制台中没有红色错误
  - 没有 "failed" 或 "error" 信息

### 成功标志

✓ **PASS** 如果：
- KG 继续显示
- Turns 指标增加到 1
- 故事继续进展
- 没有错误

✗ **FAIL** 如果：
- KG 消失
- Dashboard 不更新
- 故事卡住
- 有错误消息

### 截图

**截图名称**: `qa-scenario2-kg-update.png`

---

## 场景 3: 加载存档 → KG 为加载的游戏呈现

### 执行步骤

1. **打开浏览器新标签或窗口**
   - 打开新标签页
   - 访问 `http://127.0.0.1:7860`
   - 等待应用加载

2. **点击侧边栏的 "💾 Save / Load" 展开器**
   - 在左侧边栏中查找 "设置" 部分
   - 找到 "💾 Save / Load" 按钮
   - 点击它展开

3. **选择第一个存档**
   - 应该看到下拉菜单或选项列表
   - 选择第一个保存的游戏
   - 显示类似这样: "Untitled Snapshot | fantasy · Turn 3 · 2025-03-26 14:30"

4. **点击 "📂 Load Selected Save" 按钮**
   - 蓝色按钮
   - 等待加载完成

### 验证清单

- [ ] **加载成功后 KG 是否显示？**
  - 左侧边栏显示网络图
  - 图应该包含已加载游戏的实体

- [ ] **Dashboard 是否显示已加载游戏的状态？**
  - Turns 应该大于 0（显示加载的回合数）
  - Entities 应该大于 2
  - Conflicts 应该显示某个值

- [ ] **故事内容是否正确？**
  - 聊天历史显示已加载游戏的故事
  - 显示之前的玩家选择和故事响应

- [ ] **是否有错误？**
  - 没有红色错误信息
  - 加载成功完成

### 成功标志

✓ **PASS** 如果：
- KG 为加载的游戏显示
- Dashboard 显示正确的回合数
- 故事内容匹配已加载的游戏
- 没有错误

✗ **FAIL** 如果：
- KG 不显示
- Dashboard 显示错误的回合数
- 加载了错误的游戏
- 有错误消息

### 截图

**截图名称**: `qa-scenario3-kg-loaded.png`

---

## 场景 4: 新应用 → 显示提示消息

### 执行步骤

1. **打开浏览器新标签或私密窗口**
   - 使用 Ctrl+Shift+N 打开新的私密窗口（这样没有缓存）
   - 或者完全关闭浏览器并重新打开

2. **访问应用**
   - 在地址栏输入: `http://127.0.0.1:7860`
   - 等待页面加载

3. **观察初始状态**
   - 看到 Genre 输入框
   - 看到 "Start New Game" 按钮
   - **不要点击任何东西**

### 验证清单

- [ ] **左侧边栏中是否有提示消息？**
  - 应该看到文字: "The knowledge graph will appear after starting a game"
  - 这是一个信息性消息（蓝色背景）

- [ ] **Dashboard 是否可见或隐藏？**
  - Dashboard 标题可能显示或隐藏
  - 如果显示，指标应该都是 0

- [ ] **是否有错误？**
  - 没有红色错误消息
  - 没有破损的 UI

### 成功标志

✓ **PASS** 如果：
- 提示消息正确显示
- Dashboard 要么隐藏要么显示安全的默认值
- 没有错误

✗ **FAIL** 如果：
- 提示消息缺失
- 显示错误而不是提示
- UI 破损

### 截图

**截图名称**: `qa-scenario4-no-game.png`

---

## 证据收集

完成所有 4 个场景后，收集以下截图：

### 必需截图

1. `qa-scenario1-kg-display.png` — KG 在开始游戏后立即显示
2. `qa-scenario2-kg-update.png` — KG 在第一步操作后仍然存在
3. `qa-scenario3-kg-loaded.png` — KG 为加载的游戏显示（可选，如果有存档）
4. `qa-scenario4-no-game.png` — 新应用显示提示消息

### 存档位置

将所有截图保存到:
```
.sisyphus/evidence/
```

---

## 测试结果记录

创建一个文本文件 `.sisyphus/evidence/qa-results.txt`，记录：

```
QA TEST RESULTS - Task 1: KG Rendering Fix
Date: [Today's Date]
Tester: [Your Name]

Scenario 1 (Start Game → KG Display): [PASS / FAIL]
  Notes: [Any observations]

Scenario 2 (First Action → KG Updates): [PASS / FAIL]
  Notes: [Any observations]

Scenario 3 (Load Save → KG Renders): [PASS / FAIL]
  Notes: [Any observations]

Scenario 4 (No Game → Fallback): [PASS / FAIL]
  Notes: [Any observations]

Overall Result: [PASS / FAIL]

Issues Found (if any):
- [Issue 1]
- [Issue 2]

Console Errors Observed:
- [Error 1]
- [Error 2]
```

---

## 问题排查

### 如果 KG 不显示

1. **检查浏览器控制台**
   - 按 F12
   - 查看 "Console" 标签
   - 查找与 "render_kg_html" 或 "kg_html" 相关的错误

2. **检查 Streamlit 应用日志**
   - 查看启动应用的终端
   - 查找 ERROR 或 WARNING 消息

3. **刷新页面**
   - 按 F5 或 Ctrl+R
   - 再次尝试 "Start New Game"

### 如果 Dashboard 不显示

1. **检查是否有 API 错误**
   - 控制台可能显示 API 调用失败
   - 确认 `.env` 中的 API key 是否正确

2. **检查网络连接**
   - 确保互联网连接正常
   - 有时候 API 服务可能不可用

### 如果出现其他错误

1. **记录错误信息**
   - 截图错误消息
   - 记录确切的文本

2. **检查 Python 版本**
   - 运行: `python --version`
   - 应该是 3.10 或更高版本

3. **重启应用**
   - 按 Ctrl+C 停止 Streamlit
   - 再次运行 `streamlit run app.py`

---

## 完成后

1. **保存所有证据**
   - 将截图放在 `.sisyphus/evidence/`

2. **记录结果**
   - 创建 `qa-results.txt`
   - 记录 PASS 或 FAIL

3. **如果全部通过**
   - 代码已准备好部署
   - 可以合并到主分支

4. **如果有失败**
   - 记录具体的问题
   - 与开发团队联系进行调试

---

## 需要帮助？

如果您在运行这些测试时遇到问题，请：

1. 检查: `.sisyphus/verification/qa-verification-plan.md` (详细的技术指南)
2. 查看: `.sisyphus/TASK-1-COMPLETION-REPORT.md` (实现报告)
3. 查看: 最新的 Git 提交 `737fd66` 的具体更改

祝测试顺利！🚀

