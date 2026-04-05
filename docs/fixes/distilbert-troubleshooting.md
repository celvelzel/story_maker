# 故障排查指南：DistilBERT 与分词器兼容性

本指南解决与 DistilBERT NLU 模块（意图分类和情感分析）相关的常见运行时问题和配置错误。

## 1. 主要错误：`unexpected keyword argument 'token_type_ids'`

### 症状

应用在 NLU 处理期间崩溃，出现以下回溯信息：
```text
TypeError: DistilBertForSequenceClassification.forward() got an unexpected keyword argument 'token_type_ids'
# 类型错误：DistilBertForSequenceClassification.forward() 收到了意外的关键字参数 'token_type_ids'
```

### 根本原因
*   **架构**：DistilBERT 是 BERT 的精简版本，明确移除了段嵌入（`token_type_ids`）以节省空间
*   **库冲突**：较新版本的 `transformers`（4.40+）即使对不使用这些字段的模型也默认返回 `token_type_ids`
*   **签名不匹配**：将这些额外的键传递给模型的 `forward()` 方法会触发 Python `TypeError`

---

## 2. 已实施的解决方案

代码库现在包含**四层防御**来应对此类及类似问题：

### 第 1 层：动态输入过滤（自动）

位于 `src/nlu/intent_classifier.py` 和 `src/nlu/sentiment_analyzer.py` 中。
系统现在使用 `inspect.signature` 在传递数据之前检查模型实际接受的参数。
*   **操作**：自动剥离不在模型 `forward` 签名中的任何键
*   **优势**：这是"面向未来"的——它将处理未来 `transformers` 更新添加的任何新字段，无需修改代码

### 第 2 层：分词器加固

*   **操作**：分词器调用现在显式设置 `return_token_type_ids=False`
*   **优势**：减少内存开销，并防止问题字段在源头上被创建

### 第 3 层：健壮的模型加载（重试 + 兜底）

如果模型因 GPU 内存不足、文件缺失或版本不匹配而加载失败：
*   **重试**：系统尝试加载 3 次，每次间隔 1 秒
*   **兜底**：如果 3 次尝试后仍加载失败，系统**自动且透明地**切换到基于关键词的规则匹配（`rule_fallback`）
*   **零停机**：即使深度学习模型不可用，游戏也会继续运行

### 第 4 层：依赖锁定

*   **操作**：`requirements.txt` 锁定 `transformers>=4.40.0,<4.50.0`
*   **优势**：防止主要库更新带来的破坏性变更，同时允许安全补丁

---

## 3. 诊断步骤

如果怀疑 NLU 存在问题，请按以下步骤操作：

### 步骤 1：运行健康检查

运行专用诊断脚本验证环境：
```bash
python scripts/health_check.py -v
```
**查找：** `[+] [PASS] token_type_ids compatibility`（token_type_ids 兼容性通过）

### 步骤 2：验证 NLU 状态

检查应用日志中的 NLU 初始化摘要：
```text
✓ All NLU modules successfully loaded (No fallbacks)
# ✓ 所有 NLU 模块成功加载（无兜底）
  - Intent: ✓ distilbert-base-uncased active
  # 意图：✓ distilbert-base-uncased 已激活
```
如果显示 `backend: rule_fallback`，说明模型加载失败。请检查日志中更早的 "Model loading failed" 信息。

### 步骤 3：检查 CUDA/GPU

如果使用 GPU，确保 `torch.cuda.is_available()` 为 true。如果 GPU 已满或不可用，系统将自动把模型切换到 CPU。

---

## 4. 常见故障排查场景

| 问题 | 解决方案 |
|:---|:---|
| **内存不足（OOM）** | 系统会捕获 OOM 错误并回退到规则匹配。要修复，请关闭其他应用或在 `.env` 中设置 `DEVICE=cpu` |
| **模型文件缺失** | 确保 `models/` 目录包含微调后的产物。如果缺失，系统使用规则匹配 |
| **Transformers 版本不正确** | 如果出现 "Transformers version warning"，运行 `pip install -r requirements.txt` 以对齐已测试的版本范围 |

---

## 5. 验证命令

| 目标 | 命令 |
|:---|:---|
| 测试兼容性逻辑 | `pytest tests/test_intent_classifier_compat.py -v` |
| 测试完整 NLU 流水线 | `pytest tests/test_nlu.py -v` |
| 完整系统集成测试 | `pytest tests/test_integration.py -v` |
