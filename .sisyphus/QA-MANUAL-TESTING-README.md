# 🎯 QA 测试准备完成 - 手动执行指南

## 📋 当前状态

✅ **实现**: 完成
✅ **自动化测试**: 全部通过
✅ **代码提交**: 完成 (Hash: 737fd66)
⏳ **手动 QA**: 准备就绪

---

## 🚀 如何启动手动测试

### 步骤 1: 启动应用程序

打开终端并运行：

```bash
cd C:\Develop\python_projects\COMP5423_NLP\story_maker
streamlit run app.py --logger.level=debug
```

应该看到：
```
Local URL: http://localhost:7860
Network URL: http://<your-ip>:7860
```

### 步骤 2: 打开浏览器

访问: **http://127.0.0.1:7860**

### 步骤 3: 执行测试

按照以下指南执行 4 个测试场景：

**详细指南**: `.sisyphus/evidence/QA-MANUAL-TESTING-GUIDE.md`

---

## 🧪 4 个测试场景

### Scenario 1: 新游戏 → KG 立即显示

**步骤**:
1. 输入流派: "fantasy"
2. 点击 "🎮 Start New Game"
3. 等待加载完成

**检查**:
- ✓ 左侧边栏中是否显示知识图谱（网络图）？
- ✓ Dashboard 显示 Turns, Entities, Conflicts 指标？
- ✓ 浏览器控制台中没有错误？

**截图**: `qa-scenario1-kg-display.png`

---

### Scenario 2: 第一步操作 → KG 继续工作

**步骤**:
1. 从场景 1 继续
2. 点击第一个分支选项（按钮 "1."）
3. 等待处理完成

**检查**:
- ✓ KG 是否仍然可见？
- ✓ Turns 指标从 0 变为 1？
- ✓ 故事是否继续进展？

**截图**: `qa-scenario2-kg-update.png`

---

### Scenario 3: 加载存档 → KG 为加载状态呈现

**步骤**:
1. 打开新标签页访问应用
2. 点击侧边栏的 "💾 Save / Load"
3. 选择一个保存的游戏
4. 点击 "📂 Load Selected Save"

**检查**:
- ✓ KG 是否显示？
- ✓ Turns 指标是否大于 0？
- ✓ 故事内容是否正确？

**截图**: `qa-scenario3-kg-loaded.png`

---

### Scenario 4: 新应用 → 显示提示消息

**步骤**:
1. 打开新的私密窗口（Ctrl+Shift+N）
2. 访问 http://127.0.0.1:7860
3. 不要点击任何东西，只观察

**检查**:
- ✓ 是否显示提示: "The knowledge graph will appear after starting a game"？
- ✓ Dashboard 是否隐藏或显示默认值？

**截图**: `qa-scenario4-no-game.png`

---

## 📸 证据收集

所有截图保存到:
```
.sisyphus/evidence/
```

需要的文件:
- [ ] `qa-scenario1-kg-display.png`
- [ ] `qa-scenario2-kg-update.png`
- [ ] `qa-scenario3-kg-loaded.png` (可选)
- [ ] `qa-scenario4-no-game.png`
- [ ] `qa-results.txt` (测试结果记录)

---

## ✓ 成功标准

### PASS (全部通过)
- ✅ 所有 4 个场景都通过
- ✅ KG 在游戏启动后立即显示
- ✅ Dashboard 正确更新
- ✅ 没有控制台错误
- ✅ 所有截图已保存

**结果**: ✅ 代码可以部署到生产环境

### FAIL (有失败)
- ❌ 任何场景失败
- ❌ KG 不显示或显示错误
- ❌ Dashboard 指标错误
- ❌ 控制台有错误
- ❌ UI 破损或崩溃

**结果**: ❌ 需要调试和修复

---

## 📝 记录模板

创建文件 `.sisyphus/evidence/qa-results.txt`:

```
QA TEST RESULTS - Task 1: KG Rendering Fix
=========================================

Date: [Today's Date]
Tester: [Your Name]
Environment: Windows 10/11, Python 3.10+, Streamlit

Scenario 1 (Start Game → KG Display): PASS / FAIL
  Duration: [time]
  Notes: [observations]

Scenario 2 (First Action → KG Updates): PASS / FAIL
  Duration: [time]
  Notes: [observations]

Scenario 3 (Load Save → KG Renders): PASS / FAIL
  Duration: [time]
  Notes: [observations]

Scenario 4 (No Game → Fallback): PASS / FAIL
  Duration: [time]
  Notes: [observations]

Overall Result: PASS / FAIL

Performance Observations:
- Game startup time: [seconds]
- KG rendering time: [seconds]
- Page responsiveness: [Good/Fair/Poor]

Issues Found:
- [Issue 1]
- [Issue 2]

Console Errors:
- [Error 1]
- [Error 2]

Recommendations:
- [Recommendation 1]
- [Recommendation 2]
```

---

## 🔧 问题排查

### KG 不显示？

1. **打开浏览器开发工具** (F12)
2. **查看 Console 标签** 查找错误
3. **查看终端输出** 查找 ERROR 或 WARNING
4. **刷新页面** (F5) 并重试

### Dashboard 不显示？

1. **检查 API 密钥** (.env 文件)
2. **检查网络连接**
3. **查看浏览器控制台** 了解 API 错误

### 其他问题？

1. **重启 Streamlit** (Ctrl+C, 然后重新运行)
2. **清除浏览器缓存**
3. **查看完整的测试指南**: `.sisyphus/evidence/QA-MANUAL-TESTING-GUIDE.md`

---

## 📚 参考文档

- **实现计划**: `.sisyphus/plans/fix-kg-dashboard-display.md`
- **逻辑验证**: `.sisyphus/verification/task1-logic-check.md`
- **自动化测试**: `.sisyphus/verification/task1-automated-check.md`
- **完成报告**: `.sisyphus/TASK-1-COMPLETION-REPORT.md`
- **手动测试指南**: `.sisyphus/evidence/QA-MANUAL-TESTING-GUIDE.md` (此文件)

---

## ⏱️ 预计时间

| 部分 | 时间 |
|------|------|
| 应用启动 | 3-5 分钟 |
| Scenario 1 | 2-3 分钟 |
| Scenario 2 | 2-3 分钟 |
| Scenario 3 | 3-4 分钟 |
| Scenario 4 | 1-2 分钟 |
| 记录结果 | 2-3 分钟 |
| **总计** | **13-20 分钟** |

---

## ✨ 下一步

### 如果 QA 通过 ✅

1. 更新 `.sisyphus/evidence/qa-results.txt` → 标记为 PASS
2. 代码已准备好部署
3. 可以合并到主分支
4. 完成工作！🎉

### 如果 QA 失败 ❌

1. 记录具体问题和错误
2. 保存所有证据（截图、错误日志）
3. 与开发团队联系
4. 根据反馈调试和重新测试

---

## 🎯 现在就开始！

**准备好了吗？让我们启动测试吧！**

```bash
# 终端中运行
streamlit run app.py --logger.level=debug

# 然后在浏览器中打开
http://127.0.0.1:7860
```

详细步骤见: `.sisyphus/evidence/QA-MANUAL-TESTING-GUIDE.md`

祝你测试顺利! 🚀

---

## 问题？

如有任何问题，请查看：
- **技术详情**: `.sisyphus/TASK-1-COMPLETION-REPORT.md`
- **Git 提交**: `737fd66` (查看确切的代码更改)
- **完整文档**: `.sisyphus/verification/` 目录

