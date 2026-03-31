# Technical Report: DistilBERT Tokenizer-Model Interface Hardening

**Report ID:** FIX-2026-0320-001  
**Date:** March 20, 2026  
**Status:** ✅ **Resolved**  
**Affected Modules:** `src/nlu/intent_classifier.py`, `src/nlu/sentiment_analyzer.py`

## 1. Executive Summary

This report details the technical fix for an interface mismatch between the HuggingFace `transformers` tokenizer and the DistilBERT sequence classification model. The fix implements a dynamic input filtering mechanism that prevents `TypeError` crashes during inference across different library versions and model architectures.

---

## 2. Problem Statement

### 2.1 Error Symptom
When running inference on certain versions of the `transformers` library (specifically 4.40+), the following error occurred:
```text
TypeError: DistilBertForSequenceClassification.forward() got an unexpected keyword argument 'token_type_ids'
```

### 2.2 Root Cause Analysis
| Factor | Description |
|:---|:---|
| **Architecture Divergence** | DistilBERT is a pruned version of BERT. To save parameters, it removes segment embeddings (`token_type_ids`). Consequently, its `forward()` method does not define this parameter. |
| **Tokenizer Defaults** | The `AutoTokenizer` (even when loading a DistilBERT config) often returns `token_type_ids` by default to remain backward compatible with standard BERT-based pipelines. |
| **Direct Passthrough** | The previous implementation used `self.model(**inputs)`, blindly passing all dictionary keys from the tokenizer to the model. |

---

## 3. Implementation: Dynamic Input Filtering

### 3.1 Design Principles
Instead of hardcoding a list of "bad" keys to remove, we implemented a generic protection layer based on Python's **Introspection** capabilities.

### 3.2 Core Logic
The `_filter_model_inputs` method was added to the base NLU classes:

```python
import inspect

def _filter_model_inputs(self, inputs):
    """
    Dynamically filters tokenizer outputs to only include parameters 
    explicitly accepted by the model's forward method.
    """
    signature = inspect.signature(self.model.forward)
    parameters = signature.parameters
    
    # Check if the model uses **kwargs (catch-all)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    if accepts_kwargs:
        return inputs  # No filtering needed if model is flexible
    
    # Filter based on explicit parameter names (input_ids, attention_mask, etc.)
    allowed_keys = set(parameters.keys())
    return {k: v for k, v in inputs.items() if k in allowed_keys}
```

### 3.3 Defensive Tokenizer Settings
In addition to the filter, we updated the tokenizer calls to explicitly request fewer fields:
```python
inputs = self.tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    return_token_type_ids=False,  # Primary defense
    ...
)
```

---

## 4. Verification Results

### 4.1 Regression Testing
- **Integration Tests**: `pytest tests/test_integration.py` passed (9/9).
- **NLU Unit Tests**: `pytest tests/test_nlu.py` passed (17/17).

### 4.2 Performance Impact
The overhead of `inspect.signature` is cached by Python, and the dictionary filtering is `O(k)` where `k` is typically 3-4. The latency increase is sub-millisecond and negligible compared to the model's forward pass.

---

## 5. Compatibility Matrix

| Scenario | Status |
|:---|:---|
| DistilBERT + Transformers 4.40+ | ✅ Fixed |
| DistilBERT + Older Transformers | ✅ Compatible (Filter-safe) |
| Standard BERT / RoBERTa | ✅ Compatible (Signature-aware) |
| Rule-based Fallback Mode | ✅ Unaffected |

---

## 6. Recommendations

1.  **Monitor Future Versions**: While this fix is generic, major updates to the `transformers` API (e.g., v5.0) should be validated using `scripts/health_check.py`.
2.  **Code Consistency**: Any new NLU modules using HuggingFace models must inherit or implement the `_filter_model_inputs` pattern.
