"""Validate and merge NLG dataset JSONL files.

验证 data/nlg_dataset/ 下所有 JSONL 文件的格式，
输出验证报告并将所有有效数据合并为一个文件。
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

DATA_DIR = Path(__file__).parent.parent / "data" / "nlg_dataset"
OUTPUT_FILE = DATA_DIR / "combined_all.jsonl"


def validate_sample(sample: Dict[str, Any], line_num: int, filename: str) -> Tuple[bool, str]:
    """Validate a single JSONL sample.

    Returns (is_valid, error_message).
    """
    # Must have "messages" key
    if "messages" not in sample:
        return False, f"missing 'messages' key"

    messages = sample["messages"]
    if not isinstance(messages, list):
        return False, f"'messages' is not a list (got {type(messages).__name__})"

    if len(messages) < 3:
        return False, f"only {len(messages)} messages (need at least 3: system, user, assistant)"

    # Check each message structure
    roles_seen = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False, f"message[{i}] is not a dict"
        if "role" not in msg:
            return False, f"message[{i}] missing 'role'"
        if "content" not in msg:
            return False, f"message[{i}] missing 'content'"
        if msg["role"] not in ("system", "user", "assistant"):
            return False, f"message[{i}] has invalid role '{msg['role']}'"
        if not isinstance(msg["content"], str):
            return False, f"message[{i}] content is not a string"
        roles_seen.append(msg["role"])

    # Must have system + user + assistant in order
    if roles_seen[0] != "system":
        return False, f"first message role is '{roles_seen[0]}', expected 'system'"
    if roles_seen[-1] != "assistant":
        return False, f"last message role is '{roles_seen[-1]}', expected 'assistant'"

    # Assistant content should not be empty
    assistant_content = messages[-1]["content"].strip()
    if len(assistant_content) < 10:
        return False, f"assistant content too short ({len(assistant_content)} chars)"

    return True, ""


def validate_file(filepath: Path) -> Tuple[List[Dict], List[Tuple[int, str]]]:
    """Validate all lines in a JSONL file.

    Returns (valid_samples, errors).
    """
    valid_samples = []
    errors = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append((line_num, f"invalid JSON: {e}"))
                continue

            is_valid, err_msg = validate_sample(sample, line_num, filepath.name)
            if is_valid:
                valid_samples.append(sample)
            else:
                errors.append((line_num, err_msg))

    return valid_samples, errors


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} does not exist")
        sys.exit(1)

    jsonl_files = sorted(DATA_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"ERROR: no .jsonl files found in {DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL files in {DATA_DIR}\n")
    print("=" * 70)

    all_valid_samples = []
    total_errors = 0

    for filepath in jsonl_files:
        valid_samples, errors = validate_file(filepath)
        status = "PASS" if len(errors) == 0 else f"WARN ({len(errors)} errors)"

        print(f"\n{filepath.name}")
        print(f"  Status: {status}")
        print(f"  Valid samples: {len(valid_samples)}")
        print(f"  Errors: {len(errors)}")

        if errors:
            # Show first 5 errors
            for line_num, msg in errors[:5]:
                print(f"    Line {line_num}: {msg}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more")

        # Check sample structure from first valid sample
        if valid_samples:
            s = valid_samples[0]
            roles = [m["role"] for m in s["messages"]]
            assistant_len = len(s["messages"][-1]["content"])
            print(f"  Structure: {roles} | assistant ~{assistant_len} chars")

        all_valid_samples.extend(valid_samples)
        total_errors += len(errors)

    # Shuffle for training diversity
    import random
    random.seed(42)
    random.shuffle(all_valid_samples)

    # Write merged file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in all_valid_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print(f"\nMERGE SUMMARY")
    print(f"  Total valid samples: {len(all_valid_samples)}")
    print(f"  Total errors skipped: {total_errors}")
    print(f"  Output: {OUTPUT_FILE}")

    # Task breakdown
    story_count = 0
    option_count = 0
    for s in all_valid_samples:
        user_content = ""
        for m in s["messages"]:
            if m["role"] == "user":
                user_content = m["content"]
                break
        if "Generate exactly" in user_content or ("options" in user_content.lower() and "JSON" in user_content):
            option_count += 1
        else:
            story_count += 1

    print(f"\n  Story generation samples: {story_count}")
    print(f"  Option generation samples: {option_count}")


if __name__ == "__main__":
    main()
