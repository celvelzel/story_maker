# Local Model Tuning & Bug Fix Report

> **Last Updated**: 2026-03-31

**Date**: 2026-03-27  
**Scope**: Improving developer experience, handling inference timeouts, and fixing encoding issues for local llama.cpp integration.

---

## 1. Problem Identification

Following the initial integration of `llama.cpp` for local CPU inference, several issues were identified:
1. **Poor Visibility**: Lack of detailed logging made it difficult to track request progress and model performance.
2. **Read Timeouts**: Local CPU inference often exceeded the default 60-second timeout, leading to frequent `Connection error` failures in the Streamlit frontend.
3. **Encoding Crashes**: Emoji characters in logs caused `gbk codec can't encode character` errors on certain Windows terminals.

## 2. Implementation Details

### 2.1 Enhanced Logging & Terminal Compatibility
Modified `src/utils/api_client.py` to provide structured, terminal-safe logging:
- **New Log Nodes**:
  - `[LLM] RECEIVED request`: Displays message roles and a 100-character content preview.
  - `[LLM] PROCESSING`: Shows active parameters (`model`, `temperature`, `max_tokens`).
  - `[LLM] DONE`: Reports execution time and token usage (input/output).
- **Encoding Fix**: Removed decorative emojis (📥, 🔄, ✅, ⚠️) from logs to prevent crashes on non-UTF-8 Windows consoles.

### 2.2 Dynamic Timeout Configuration
To accommodate the slower nature of CPU-based inference, timeouts were moved to the global configuration:
- **`config.py` Updates**:
  - Added `OPENAI_TIMEOUT_CONNECT` (Default: 10s).
  - Added `OPENAI_TIMEOUT_READ` (Default: 60s).
- **`.env` Specifics for Local Inference**:
  - Increased `OPENAI_TIMEOUT_READ` to 180s for llama.cpp setups to prevent premature disconnection during long generations.
- **Client Implementation**:
  - `src/utils/api_client.py` now uses `httpx.Timeout` objects to apply these granular settings to the OpenAI-compatible client.

### 2.3 Documentation Support
- Created `docs/guides/local-model-startup.md`: A quick-start guide for local model parameters and log interpretation.
- Created `docs/guides/zero-to-hero-deployment.md`: A comprehensive, multi-platform deployment reference.

## 3. Verification & Results
- **Logging**: ✅ Verified that terminal logs correctly display request/response cycles.
- **Encoding**: ✅ No further `gbk` codec errors observed on Windows.
- **Stability**: ✅ Local models can now generate long story segments (up to 3 minutes) without being interrupted by timeouts.
- **Flexibility**: ✅ Confirmed that `api_client` dynamically respects `.env` timeout adjustments.

## 4. Next Steps
- Monitor CPU performance across different hardware to refine default timeout values.
- Enhance UI loading states to clearly inform users when a local model is in use and that longer response times are expected.
