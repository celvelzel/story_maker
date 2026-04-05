# DistilBERT 兼容性修复

**报告 ID：** FIX-2026-0320-001  
**日期：** 2026 年 3 月 20 日  
**状态：** ✅ **已解决并验证**

## 1. 问题描述

### 症状

在使用微调后的 DistilBERT 意图分类器或 DistilRoBERTa 情感分析器进行推理时，系统崩溃并报错：
```text
TypeError: DistilBertForSequenceClassification.forward() got an unexpected keyword argument 'token_type_ids'
# 类型错误：DistilBertForSequenceClassification.forward() 收到了意外的关键字参数 'token_type_ids'
```

### 根本原因分析

1.  **架构差异**：与 BERT 不同，DistilBERT 架构移除了 `token_type_ids`（段嵌入）以减少参数量。其 `forward()` 方法不接受此参数
2.  **分词器默认行为**：许多版本的 `transformers` `AutoTokenizer` 默认返回 `token_type_ids`，以确保与标准 BERT 的兼容性，即使模型是 DistilBERT
3.  **直接传递**：代码之前将分词器返回的整个字典直接传递给模型（`self.model(**inputs)`），导致 `TypeError`

---

## 2. 综合五层防护方案

我们实施了多层防御策略，确保在不同硬件和库版本下的稳定性：

| 层级 | 措施 | 目的 |
|:---|:---|:---|
| **1. 输入过滤** | `_filter_model_inputs()` 工具函数 | 自动剥离 `model.forward` 签名中不存在的字段 |
| **2. 分词器配置** | `return_token_type_ids=False` | 从源头阻止不兼容字段的生成 |
| **3. 安全加载** | 重试 + 兜底机制 | 对瞬态 IO/GPU 问题实施 3 次重试，完全失败时回退到基于规则的 NLU |
| **4. 依赖锁定** | `transformers>=4.40.0,<4.50.0` | 确保代码库在已验证的库版本范围内运行 |
| **5. 健康检查** | `scripts/health_check.py` | 应用启动前的主动诊断工具，验证环境就绪性 |

---

## 3. 实现细节

### 自动输入过滤

我们使用 Python 的 `inspect` 模块动态确定模型接受的参数，而非硬编码允许的字段。

```python
def _filter_model_inputs(self, inputs):
    """
    动态过滤分词器输出，仅包含模型 forward 方法明确接受的参数
    """
    # 获取模型 forward 方法的签名
    signature = inspect.signature(self.model.forward)
    parameters = signature.parameters
    # 如果模型接受 **kwargs（可变关键字参数），则无需过滤
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return inputs
    # 获取允许的参数名集合
    allowed_keys = set(parameters.keys())
    # 仅保留模型接受的键值对
    return {k: v for k, v in inputs.items() if k in allowed_keys}
```

### 推理流水线加固

`_model_predict` 和 `_model_analyze` 方法现在同时使用分词器级和模型级的防御：

```python
inputs = self.tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=self.max_length,
    padding=True,
    return_token_type_ids=False,  # 源头级防御：不生成 token_type_ids
)
inputs = self._filter_model_inputs(inputs)  # 模型级防御：过滤不接受的参数
```

---

## 4. 验证与测试

### 测试套件结果
- **总测试数**：266 通过，0 失败
- **新增回归测试**：`tests/test_intent_classifier_compat.py`（7 个测试，覆盖过滤和兜底逻辑）

### 健康检查工具

运行 `python scripts/health_check.py` 确认：
- [PASS] `token_type_ids` 兼容性
- [PASS] IntentClassifier 预测稳定性
- [PASS] CUDA / 设备可用性
- [PASS] 模型目录完整性

---

## 5. 维护指南

1.  **环境迁移**：将项目迁移到新机器时，始终运行 `python scripts/health_check.py -v`
2.  **库更新**：如果 `transformers` 更新到 v4.49 以上，监控日志中的 "Transformers version warning" 并重新运行兼容性测试
3.  **模型替换**：如果从 DistilBERT 切换到*需要* `token_type_ids` 的模型（如 BERT-base），动态过滤将自动允许该字段通过，无需修改代码
