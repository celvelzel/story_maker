# Troubleshooting Guide: DistilBERT & Tokenizer Compatibility

This guide addresses common runtime issues and configuration errors related to the DistilBERT NLU modules (Intent Classification and Sentiment Analysis).

## 1. Primary Error: `unexpected keyword argument 'token_type_ids'`

### Symptom
The application crashes during NLU processing with the following traceback:
```text
TypeError: DistilBertForSequenceClassification.forward() got an unexpected keyword argument 'token_type_ids'
```

### Root Cause
*   **Architecture**: DistilBERT is a distilled version of BERT that explicitly removes segment embeddings (`token_type_ids`) to save space.
*   **Library Conflict**: Newer versions of `transformers` (4.40+) return `token_type_ids` by default even for models that don't use them.
*   **Signature Mismatch**: Passing these extra keys to the model's `forward()` method triggers a Python `TypeError`.

---

## 2. Implemented Solutions

The codebase now includes a **4-layer defense** against this and similar issues:

### Layer 1: Dynamic Input Filtering (Automatic)
Located in `src/nlu/intent_classifier.py` and `src/nlu/sentiment_analyzer.py`.
The system now uses `inspect.signature` to check what the model actually accepts before passing the data.
*   **Action**: Any key not in the model's `forward` signature is automatically stripped.
*   **Benefit**: This is "future-proof"—it will handle any new fields added by future `transformers` updates without code changes.

### Layer 2: Tokenizer Hardening
*   **Action**: Tokenizer calls now explicitly set `return_token_type_ids=False`.
*   **Benefit**: Reduces memory overhead and prevents the problematic field from being created in the first place.

### Layer 3: Robust Model Loading (Retry + Fallback)
If the model fails to load due to GPU memory issues, missing files, or version mismatches:
*   **Retry**: The system attempts to load 3 times with a 1-second delay.
*   **Fallback**: If loading fails after 3 attempts, the system **automatically and transparently** switches to keyword-based rule matching (`rule_fallback`).
*   **Zero Downtime**: The game will continue to run even if the deep learning models are unavailable.

### Layer 4: Dependency Pinning
*   **Action**: `requirements.txt` pins `transformers>=4.40.0,<4.50.0`.
*   **Benefit**: Prevents breaking changes from major library updates while allowing security patches.

---

## 3. Diagnostic Steps

If you suspect NLU issues, follow these steps:

### Step 1: Run the Health Check
Run the dedicated diagnostic script to verify your environment:
```bash
python scripts/health_check.py -v
```
**Look for:** `[+] [PASS] token_type_ids compatibility`.

### Step 2: Verify NLU Status
Check the application logs for the NLU initialization summary:
```text
✓ All NLU modules successfully loaded (No fallbacks)
  - Intent: ✓ distilbert-base-uncased active
```
If it shows `backend: rule_fallback`, the model failed to load. Check for "Model loading failed" earlier in the logs.

### Step 3: Check CUDA/GPU
If using a GPU, ensure `torch.cuda.is_available()` is true. The system will automatically move the model to CPU if the GPU is full or unavailable.

---

## 4. Common Troubleshooting Scenarios

| Issue | Solution |
|:---|:---|
| **Out of Memory (OOM)** | The system will catch the OOM error and fall back to rules. To fix, close other applications or set `DEVICE=cpu` in your `.env`. |
| **Missing Model Files** | Ensure the `models/` directory contains the fine-tuned artifacts. If missing, the system uses rules. |
| **Wrong Transformers Version** | If you see a "Transformers version warning", run `pip install -r requirements.txt` to align with the tested range. |

---

## 5. Verification Commands

| Goal | Command |
|:---|:---|
| Test Compatibility Logic | `pytest tests/test_intent_classifier_compat.py -v` |
| Test Full NLU Pipeline | `pytest tests/test_nlu.py -v` |
| Full System Integration | `pytest tests/test_integration.py -v` |
