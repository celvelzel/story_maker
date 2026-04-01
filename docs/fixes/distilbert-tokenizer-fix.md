# 技术报告：DistilBERT 分词器-模型接口加固

**报告 ID：** FIX-2026-0320-001  
**日期：** 2026 年 3 月 20 日  
**状态：** ✅ **已解决**  
**受影响模块：** `src/nlu/intent_classifier.py`、`src/nlu/sentiment_analyzer.py`

## 1. 执行摘要

本报告详细说明了 HuggingFace `transformers` 分词器与 DistilBERT 序列分类模型之间接口不匹配的技术修复。该修复实现了一种动态输入过滤机制，防止在不同库版本和模型架构下进行推理时出现 `TypeError` 崩溃。

---

## 2. 问题陈述

### 2.1 错误症状

在运行某些版本的 `transformers` 库（特别是 4.40+）进行推理时，出现以下错误：
```text
TypeError: DistilBertForSequenceClassification.forward() got an unexpected keyword argument 'token_type_ids'
# 类型错误：DistilBertForSequenceClassification.forward() 收到了意外的关键字参数 'token_type_ids'
```

### 2.2 根本原因分析

| 因素 | 描述 |
|:---|:---|
| **架构差异** | DistilBERT 是 BERT 的精简版本。为节省参数，它移除了段嵌入（`token_type_ids`）。因此，其 `forward()` 方法未定义此参数 |
| **分词器默认值** | `AutoTokenizer`（即使加载 DistilBERT 配置）通常默认返回 `token_type_ids`，以保持与标准 BERT 流水线的向后兼容 |
| **直接传递** | 之前的实现使用 `self.model(**inputs)`，盲目地将分词器返回的所有字典键传递给模型 |

---

## 3. 实现：动态输入过滤

### 3.1 设计原则

我们没有硬编码需要移除的"坏"键列表，而是基于 Python 的**内省**能力实现了一个通用防护层。

### 3.2 核心逻辑

`_filter_model_inputs` 方法被添加到 NLU 基类中：

```python
import inspect

def _filter_model_inputs(self, inputs):
    """
    动态过滤分词器输出，仅包含模型 forward 方法明确接受的参数
    """
    # 获取模型 forward 方法的签名
    signature = inspect.signature(self.model.forward)
    parameters = signature.parameters
    
    # 检查模型是否使用 **kwargs（捕获所有参数）
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    if accepts_kwargs:
        return inputs  # 如果模型足够灵活，无需过滤
    
    # 基于显式参数名进行过滤（input_ids、attention_mask 等）
    allowed_keys = set(parameters.keys())
    return {k: v for k, v in inputs.items() if k in allowed_keys}
```

### 3.3 分词器防御性设置

除了过滤器之外，我们还更新了分词器调用以显式请求更少的字段：

```python
inputs = self.tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    return_token_type_ids=False,  # 主防御：不生成 token_type_ids
    ...
)
```

---

## 4. 验证结果

### 4.1 回归测试
- **集成测试**：`pytest tests/test_integration.py` 通过（9/9）
- **NLU 单元测试**：`pytest tests/test_nlu.py` 通过（17/17）

### 4.2 性能影响

`inspect.signature` 的开销由 Python 缓存，字典过滤的时间复杂度为 `O(k)`，其中 k 通常为 3-4。延迟增加不到 1 毫秒，与模型前向传播相比可忽略不计。

---

## 5. 兼容性矩阵

| 场景 | 状态 |
|:---|:---|
| DistilBERT + Transformers 4.40+ | ✅ 已修复 |
| DistilBERT + 旧版 Transformers | ✅ 兼容（过滤安全） |
| 标准 BERT / RoBERTa | ✅ 兼容（签名感知） |
| 基于规则的兜底模式 | ✅ 不受影响 |

---

## 6. 建议

1.  **监控未来版本**：虽然此修复是通用的，但 `transformers` API 的重大更新（如 v5.0）应使用 `scripts/health_check.py` 进行验证
2.  **代码一致性**：任何使用 HuggingFace 模型的新 NLU 模块必须继承或实现 `_filter_model_inputs` 模式
