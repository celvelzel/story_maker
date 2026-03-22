"""Fine-tune Llama-3.2-3B-Instruct for story generation using ms-swift + LoRA."""
import json
import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings

logger = logging.getLogger(__name__)

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert interactive-fiction narrator for a text-adventure game.\n"
    "Rules:\n"
    '1. Always narrate in **second person** ("You see…", "You feel…").\n'
    "2. Keep each response between 2-4 paragraphs.\n"
    "3. Maintain absolute consistency with the world state provided.\n"
    "4. Use vivid, sensory language — sights, sounds, smells.\n"
    "5. Never mention game mechanics, stats, or that you are an AI.\n"
    "6. Seamlessly incorporate the player's action into the narrative.\n"
    "7. End the passage at a moment that invites the player to act next."
)


# ── Data preparation ───────────────────────────────────────────

def convert_arrow_to_jsonl(arrow_path: str, output_path: str, max_samples: int = 0) -> str:
    """Convert WritingPrompts Arrow dataset to ms-swift JSONL format.

    Each Arrow sample has ``prompt`` and ``story`` fields.
    Output JSONL format (one per line)::

        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "<prompt>"},
            {"role": "assistant", "content": "<story>"}
        ]}

    Parameters
    ----------
    arrow_path : str
        Path to the HuggingFace Arrow dataset directory
        (e.g. ``data/raw/writingprompts``).
    output_path : str
        Destination ``.jsonl`` file path.
    max_samples : int
        If > 0, limit to this many samples (useful for quick experiments).

    Returns
    -------
    str
        The output file path.
    """
    from datasets import load_from_disk

    dataset = load_from_disk(arrow_path)
    if "train" in dataset:
        dataset = dataset["train"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            prompt = (sample.get("prompt") or "").strip()
            story = (sample.get("story") or "").strip()
            if not prompt or not story:
                continue

            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": story},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if 0 < max_samples <= count:
                break

    logger.info("Wrote %d samples to %s", count, output_path)
    return output_path


def ensure_jsonl(data_path: str, output_dir: str, max_samples: int = 0) -> str:
    """Return a JSONL path, converting from Arrow if necessary.

    Parameters
    ----------
    data_path : str
        Either a ``.jsonl`` file or an Arrow dataset directory.
    output_dir : str
        Directory to write the converted JSONL if needed.
    max_samples : int
        Passed through to :func:`convert_arrow_to_jsonl`.

    Returns
    -------
    str
        Path to a ``.jsonl`` file ready for ms-swift.
    """
    if data_path.endswith(".jsonl") and os.path.isfile(data_path):
        return data_path

    jsonl_path = os.path.join(output_dir, "train_dataset.jsonl")
    if os.path.exists(jsonl_path):
        logger.info("Reusing existing JSONL: %s", jsonl_path)
        return jsonl_path

    return convert_arrow_to_jsonl(data_path, jsonl_path, max_samples=max_samples)


# ── Training ───────────────────────────────────────────────────

def train(args):
    """Fine-tune Llama-3.2-3B-Instruct with LoRA via ms-swift."""
    from swift.llm import sft_main, TrainArguments

    jsonl_path = ensure_jsonl(args.data_path, args.output_dir, args.max_samples)

    print(f"Fine-tuning {MODEL_ID} with LoRA via ms-swift")
    print(f"  Dataset:        {jsonl_path}")
    print(f"  LoRA rank:      {args.lora_r}")
    print(f"  LoRA alpha:     {args.lora_alpha}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Max length:     {args.max_length}")
    print(f"  Output:         {args.output_dir}")

    result = sft_main(
        TrainArguments(
            model=MODEL_ID,
            train_type="lora",
            dataset=[jsonl_path],
            torch_dtype="bfloat16",
            # LoRA
            lora_rank=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            # Training
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            gradient_checkpointing=True,
            # Sequence
            max_length=args.max_length,
            # Logging / saving
            output_dir=args.output_dir,
            logging_steps=10,
            save_strategy="epoch",
            report_to=["tensorboard"],
        )
    )

    logger.info("Training finished. Best checkpoint: %s", result)

    # Persist training config alongside the adapter weights
    config = {
        "base_model": MODEL_ID,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_length": args.max_length,
        "data_path": args.data_path,
    }
    config_path = os.path.join(args.output_dir, "training_config.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\nTraining config saved to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama-3.2-3B-Instruct with LoRA (ms-swift)"
    )
    parser.add_argument(
        "--data_path",
        default=str(settings.DATA_DIR / "raw" / "writingprompts"),
        help="Path to Arrow dataset directory or .jsonl file",
    )
    parser.add_argument(
        "--output_dir",
        default=str(settings.PROJECT_ROOT / "models" / "story_generator_lora"),
        help="Output directory for checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Limit training samples (0 = use all)",
    )

    args = parser.parse_args()
    train(args)
