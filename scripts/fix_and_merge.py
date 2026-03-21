"""Fix broken JSONL files and merge all datasets (v3).

Fix strategies per file:
- chatgpt, gemini: Unescaped " in system prompt — replace with known-good text
- qwen: Literal nn/n instead of escaped newlines — replace with known-good text
- mimo: Double-escaped \\\" throughout — normalize to \"
- deepseek, doubao, grok: Truncated lines — skip unparseable lines
- yiyan: Empty file — skip
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

DATA_DIR = Path(__file__).parent.parent / "data" / "nlg_dataset"
OUTPUT_FILE = DATA_DIR / "combined_all.jsonl"

# Known-good system prompt content (the exact text between "content": " and ")
# In JSON this needs literal \n as escape sequences (2 chars each)
GOOD_SYSTEM_PROMPT = (
    "You are an expert interactive-fiction narrator for a text-adventure game."
    "\\n\\nRules:\\n"
    "1. Always narrate in **second person** ('You see\\u2026', 'You feel\\u2026').\\n"
    "2. Keep each response between 2-4 paragraphs.\\n"
    "3. Maintain absolute consistency with the world state provided.\\n"
    "4. Use vivid, sensory language \\u2014 sights, sounds, smells.\\n"
    "5. Never mention game mechanics, stats, or that you are an AI.\\n"
    "6. Seamlessly incorporate the player's action into the narrative.\\n"
    "7. End the passage at a moment that invites the player to act next."
)


def validate_sample(sample: Dict[str, Any]) -> Tuple[bool, str]:
    if "messages" not in sample:
        return False, "missing 'messages'"
    messages = sample["messages"]
    if not isinstance(messages, list) or len(messages) < 3:
        return False, "need >=3 messages"
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if roles[0] != "system" or roles[-1] != "assistant":
        return False, "bad role order"
    if len(messages[-1].get("content", "").strip()) < 10:
        return False, "assistant too short"
    return True, ""


def try_parse(line: str) -> Tuple[bool, Dict]:
    try:
        sample = json.loads(line)
        ok, _ = validate_sample(sample)
        return ok, sample if ok else {}
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return False, {}


def fix_chatgpt_gemini(line: str) -> str:
    """Replace the broken system prompt content with known-good version.

    chatgpt/gemini have unescaped " inside system prompt strings.
    We replace everything between the first "content": " and the next
    "}, {"role": "user" with the known-good system prompt.
    """
    marker = '"role": "system", "content": "'
    idx = line.find(marker)
    if idx == -1:
        return line

    start = idx + len(marker)

    # Find the end of the system message: "}, {"role": "user"
    end_marker = '"}, {"role": "user"'
    end_idx = line.find(end_marker, start)
    if end_idx == -1:
        # Try alternate ending patterns
        end_marker = '"},  {"role": "user"'
        end_idx = line.find(end_marker, start)
        if end_idx == -1:
            return line

    return line[:start] + GOOD_SYSTEM_PROMPT + line[end_idx:]


def fix_qwen(line: str) -> str:
    """Fix qwen's literal nn/n newlines and missing ** markers.

    Strategy: same as chatgpt — replace the broken system prompt entirely.
    Then also fix any nn/n patterns in the rest of the content.
    """
    # First try replacing the system prompt
    line = fix_chatgpt_gemini(line)

    # If still broken, try fixing nn/n patterns globally
    ok, _ = try_parse(line)
    if ok:
        return line

    # Replace literal nn (at word boundary) with \n\n
    # But be careful: only replace nn that looks like newline artifacts
    # Pattern: nn followed by uppercase (start of new line)
    import re
    fixed = re.sub(r'nn(?=[A-Z])', r'\\n\\n', line)
    fixed = re.sub(r'(?<=[.?!])n(?=\d+\. )', r'\\n', fixed)
    return fixed


def fix_mimo(line: str) -> str:
    """Fix mimo's issues: unescaped quotes in system prompt + double-escaped quotes.

    Step 1: Replace system prompt with known-good version (fixes unescaped ")
    Step 2: Normalize remaining \\\" to \" in the rest of the content
    """
    BS = chr(92)
    QT = chr(34)

    # Step 1: Replace system prompt
    line = fix_chatgpt_gemini(line)

    # Step 2: Normalize backslash-quote to just quote
    # (mimo has \\\" where it should have \")
    line = line.replace(BS + QT, QT)

    return line


def process_file(filepath: Path, fix_fn=None) -> Tuple[List[Dict], int, int]:
    valid = []
    fixed_count = 0
    failed_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue

            ok, sample = try_parse(line)
            if ok:
                valid.append(sample)
                continue

            if fix_fn:
                fixed_line = fix_fn(line)
                ok, sample = try_parse(fixed_line)
                if ok:
                    valid.append(sample)
                    fixed_count += 1
                    continue

            failed_count += 1

    return valid, fixed_count, failed_count


def main():
    jsonl_files = sorted(DATA_DIR.glob("*.jsonl"))
    jsonl_files = [f for f in jsonl_files if f.name != OUTPUT_FILE.name]

    print(f"Found {len(jsonl_files)} files in {DATA_DIR}\n")
    print("=" * 70)

    fix_fns = {
        "combined_dataset_chatgpt.jsonl": fix_chatgpt_gemini,
        "combined_dataset_gemini.jsonl": fix_chatgpt_gemini,
        "combined_dataset_qwen.jsonl": fix_qwen,
        "combined_dataset_mimo.jsonl": fix_mimo,
    }

    all_samples: List[Dict] = []

    for filepath in jsonl_files:
        name = filepath.name
        fix_fn = fix_fns.get(name)

        with open(filepath, "r", encoding="utf-8") as f:
            raw_lines = sum(1 for l in f if l.strip())

        direct_valid, _, _ = process_file(filepath)

        if len(direct_valid) == raw_lines and raw_lines > 0:
            print(f"\n{name}: {len(direct_valid)} valid (no fixes needed)")
            all_samples.extend(direct_valid)
        elif raw_lines == 0:
            print(f"\n{name}: empty file, skipped")
        else:
            repaired_valid, fixed, failed = process_file(filepath, fix_fn)
            gained = len(repaired_valid) - len(direct_valid)
            print(f"\n{name}: {raw_lines} lines")
            print(f"  Before fix: {len(direct_valid)} valid, {raw_lines - len(direct_valid)} broken")
            print(f"  After fix:  {len(repaired_valid)} valid ({fixed} fixed, {failed} still broken)")
            if gained > 0:
                print(f"  >>> GAINED {gained} samples!")
            all_samples.extend(repaired_valid)

    import random
    random.seed(42)
    random.shuffle(all_samples)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Stats
    story = 0
    option = 0
    for s in all_samples:
        user_content = ""
        for m in s["messages"]:
            if m["role"] == "user":
                user_content = m.get("content", "")
                break
        if "Generate exactly" in user_content or "generate exactly" in user_content:
            option += 1
        else:
            story += 1

    print("\n" + "=" * 70)
    print(f"\nFINAL RESULT")
    print(f"  Total:  {len(all_samples)} samples")
    print(f"  Story:  {story}")
    print(f"  Option: {option}")
    print(f"  Output: {OUTPUT_FILE}")
    import os
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"  Size:   {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
