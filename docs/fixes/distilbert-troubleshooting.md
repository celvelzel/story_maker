# Troubleshooting: DistilBERT / Tokenizer Compatibility

## Issue

`TypeError: DistilBertForSequenceClassification.forward() got an unexpected keyword argument 'token_type_ids'`

## Root Cause

- **DistilBERT Architecture**: DistilBERT does not use `token_type_ids` (segment embeddings).
- **Tokenizer behavior**: Some `transformers` versions return `token_type_ids` by default.
- **Strict model signatures**: When `**inputs` is passed to the model, an extra `token_type_ids` key causes a `TypeError`.

## Applied Fixes

### 1. Generic Input Filtering (Primary Defense)
Location: `src/nlu/intent_classifier.py`, `src/nlu/sentiment_analyzer.py`

```python
def _filter_model_inputs(self, inputs):
    signature = inspect.signature(self.model.forward)
    parameters = signature.parameters
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in parameters.values()
    )
    if accepts_kwargs:
        return inputs  # Model accepts extra kwargs, no filtering needed
    allowed_keys = set(parameters.keys())
    return {k: v for k, v in inputs.items() if k in allowed_keys}
```

- **Why**: Automatically strips any key the model doesn't accept, not just `token_type_ids`. Future-proof against new tokenizer fields.
- **Benefits**: Works for DistilBERT, BERT, RoBERTa, and any model variant.

### 2. Tokenizer Configuration (Secondary Defense)
```python
inputs = self.tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=...,
    padding=True,
    return_token_type_ids=False,  # Prevents token_type_ids from being generated
)
```

- **Why**: Reduces unnecessary fields at the source, making filtering faster.

### 3. Model Loading Safety Wrapper
Location: `src/nlu/intent_classifier.py`, `src/nlu/sentiment_analyzer.py`

- **Retry logic**: Up to 3 attempts with 1-second delay for transient load failures.
- **Clean fallback**: On any failure, resets model state and falls back to keyword matching.
- **Version warning**: Warns if `transformers >= 4.50` is detected (outside tested range).

```python
MAX_RETRIES = 3
RETRY_DELAY = 1.0

def load(self) -> None:
    for attempt in range(1, self.MAX_RETRIES + 1):
        try:
            # ... load logic ...
            return
        except Exception as exc:
            if attempt < self.MAX_RETRIES:
                time.sleep(self.RETRY_DELAY)
                self._reset_model_state()
    # Falls back to rule-based keyword matching
    self._reset_model_state()
    self.backend = "rule_fallback"
```

### 4. Dependency Version Pins
Location: `requirements.txt`

```
transformers>=4.40.0,<4.50.0
```

- **Why**: Locks to the tested range. Newer major versions may introduce breaking changes.

## Deployment Checklist

Before deploying to a new environment, run:
```bash
# Health check (verifies all dependencies and model loading)
python scripts/health_check.py -v

# Full test suite
pytest tests/ -v
```

## Additional Safeguards

- **Cross-device support**: Automatic CPU/GPU detection; model moves to available device.
- **Rule fallback**: If model loading fails (missing weights, GPU unavailable, incompatible version), the system transparently falls back to keyword-based classification — no downtime.
- **Error isolation**: `_check_transformers_version()` wraps version checks in try/except to avoid crashing in unusual environments.

## Verification

```bash
# Unit tests for the compatibility fix
pytest tests/test_intent_classifier_compat.py -v

# Full integration tests
pytest tests/test_integration.py -v

# All tests
pytest tests/ -v
```

Expected result: **All tests pass** (266 tests).
