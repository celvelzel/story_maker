# 修复报告：DistilBERT Tokenizer 与 Model 输入兼容性加固

**报告编号**: FIX-2026-0320-001  
**修复日期**: 2026-03-20  
**影响范围**: `src/nlu/intent_classifier.py`、`src/nlu/sentiment_analyzer.py`  
**提交哈希**: `42e65d7`  
**严重程度**: 中（运行时报错，不影响数据安全）

---

## 1. 问题描述

### 1.1 错误现象

```
TypeError: DistilBertForSequenceClassification.forward() got an unexpected keyword argument 'token_type_ids'
```

在较新版本的 `transformers` 库中，使用 `DistilBertForSequenceClassification` 进行推理时，调用 `self.model(**inputs)` 会抛出 `TypeError`。

### 1.2 影响范围

- `IntentClassifier`（使用本地微调的 DistilBERT 模型）
- `SentimentAnalyzer`（使用 `distilroberta-base`，潜在风险）

---

## 2. 根因分析

| 因素 | 说明 |
|------|------|
| 模型架构差异 | DistilBERT 是 BERT 的蒸馏版本，移除了 `token_type_ids`（段落嵌入/segment embeddings），因此 `forward()` 签名中**不接受**此参数 |
| Tokenizer 行为 | 部分 `transformers` 版本中，`AutoTokenizer` 默认仍会返回 `token_type_ids` 字段 |
| 直接透传输入 | 代码中直接使用 `self.model(**inputs)`，将 tokenizer 返回的所有字段原样传给模型，导致额外字段被拒绝 |

---

## 3. 修复方案

### 3.1 修复策略：通用防护（Generic Input Filtering）

**设计原则**：
- 不硬编码特定字段名（如只移除 `token_type_ids`），而是**按模型 `forward` 签名自动过滤**。
- 对 tokenizer 侧做防御性参数设置（`return_token_type_ids=False`），减少无效字段生成。
- 保留 `attention_mask`、`input_ids` 等必要字段，兼容不同模型架构。

**优点**：
- 一次修复，覆盖当前及未来可能的额外字段问题。
- 不依赖具体 `transformers` 版本细节。
- 对 DistilBERT、RoBERTa、BERT 等模型均有效。

### 3.2 改动详情

#### 改动 1：`src/nlu/intent_classifier.py`

| 位置 | 改动内容 |
|------|----------|
| 导入 | 增加 `import inspect` |
| `_model_predict()` | tokenizer 调用增加 `return_token_type_ids=False`；新增 `_filter_model_inputs()` 过滤输入；`.to(self.device)` 延后到过滤后执行 |
| 新增方法 | `_filter_model_inputs(self, inputs)` — 按 `model.forward` 签名过滤输入字典 |

#### 改动 2：`src/nlu/sentiment_analyzer.py`

| 位置 | 改动内容 |
|------|----------|
| 导入 | 增加 `import inspect` |
| `_model_analyze()` | tokenizer 调用增加 `return_token_type_ids=False`；新增 `_filter_model_inputs()` 过滤输入；`.to(self.device)` 延后到过滤后执行 |
| 新增方法 | `_filter_model_inputs(self, inputs)` — 按 `model.forward` 签名过滤输入字典 |

### 3.3 核心过滤逻辑

```python
def _filter_model_inputs(self, inputs):
    """Filter tokenizer outputs to parameters accepted by model.forward."""
    signature = inspect.signature(self.model.forward)
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_kwargs:
        return inputs  # 模型接受 **kwargs，不过滤

    allowed_keys = set(parameters.keys())
    return {key: value for key, value in inputs.items() if key in allowed_keys}
```

**逻辑说明**：
1. 获取模型的 `forward` 参数签名。
2. 判断是否接受 `**kwargs`（`VAR_KEYWORD`）—— 若接受则不干预。
3. 否则，仅保留签名中显式声明的参数键，过滤掉任何额外字段。

---

## 4. 验证结果

### 4.1 功能验证

| 测试命令 | 结果 |
|----------|------|
| `pytest tests/test_integration.py` | **9 passed**，无失败 |

### 4.2 回归检查

- 意图分类输出结构：`{"intent": str, "confidence": float}` — 未变化
- 情感分析输出结构：`{"emotion": str, "confidence": float, "scores": dict}` — 未变化
- 规则回退逻辑（`rule_fallback`）— 未修改，不受影响

### 4.3 性能影响

- 每次推理增加一次字典过滤（`O(k)`，k 为 tokenizer 输出字段数，通常 ≤ 4）。
- 耗时增量远小于模型前向传播，可忽略不计。

### 4.4 Git 提交信息

```
fix: harden tokenizer-model input compatibility

- Add _filter_model_inputs() to filter tokenizer outputs against model.forward signature
- Set return_token_type_ids=False on tokenizer calls
- Apply fixes to both IntentClassifier and SentimentAnalyzer
```

---

## 5. 兼容性评估

| 场景 | 状态 |
|------|------|
| DistilBERT + 新版 transformers | ✅ 已修复 |
| DistilBERT + 旧版 transformers | ✅ 兼容（过滤逻辑兜底） |
| DistilRoBERTa | ✅ 兼容 |
| 未来切换到 BERT / RoBERTa | ✅ 兼容（签名过滤自动适配） |
| 规则回退模式（模型未加载） | ✅ 不受影响 |

---

## 6. 后续建议

1. **单元测试补充**：增加定向用例，验证输入含 `token_type_ids` 时不抛异常。
2. **模型路径检查**：定期验证本地微调模型与当前 `transformers` 版本的兼容性。
3. **文档更新**：将本修复记录同步到 `docs/troubleshooting_distilbert.md` 中。

---

*报告生成时间：2026-03-20*
