# LLM JSON Truncation Fix

## Issue
The Knowledge Graph relation extractor (`dual_extract` and `extract`) was failing with JSON decoding errors:
`Expecting ',' delimiter` or `Unterminated string starting at...`

## Cause
The LLM response was being truncated because the `max_tokens` limit was set too low (512 for full extraction, 256 for single player input). Due to the rich schema requirements (including `description`, `status`, `state_changes`, and `context`), the generated JSON output frequently exceeded these limits, causing the JSON string to end prematurely before closing. 

Additionally, the original raw truncated string was not logged, making debugging difficult since the error was caught and swallowed or only the standard JSON exception was raised.

## Solution
1. **Increased Token Limits:** 
   - In `src/knowledge_graph/relation_extractor.py`, `extract` and `extract_dual` `max_tokens` increased from 512 to 1024.
   - `_extract_player_input` `max_tokens` increased from 256 to 512.
2. **Improved Error Logging:**
   - In `src/utils/api_client.py`, wrapped `json.loads(raw)` in a `try-except` block.
   - Added logging to output the raw, unparsed string (`raw`) if a `JSONDecodeError` occurs, enabling easier inspection of what the LLM attempted to generate when it fails.
