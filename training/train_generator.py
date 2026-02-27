"""Fine-tune GPT-2 for story generation using LoRA (PEFT)."""
import os
import sys
import argparse
from pathlib import Path

import torch
from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings

# This script is for optional GPT-2 LoRA fine-tuning (legacy).
# The main project uses API-based NLG (gpt-4o-mini) — this is NOT required.
GENERATOR_MODEL_NAME = "gpt2"


def prepare_dataset(data_path: str, tokenizer, max_length: int = 512):
    """
    Load and prepare dataset for causal language modeling.
    
    Expected format: each sample is a story segment with
    [SETTING], [CHARACTERS], [STORY] markers.
    """
    if os.path.exists(data_path):
        dataset = load_from_disk(data_path)
        if "train" in dataset:
            dataset = dataset["train"]
    else:
        # Fallback: create a small demo dataset
        print(f"WARNING: {data_path} not found. Using demo data.")
        dataset = Dataset.from_dict({
            "text": [
                "[SETTING] A dark forest with ancient trees.\n[CHARACTERS] A wandering knight, a mysterious fairy\n[STORY] The knight ventured deeper into the forest, guided by a faint glow between the trees. The fairy appeared, hovering just above the mossy ground. 'Turn back, mortal,' she whispered. 'The Forest King does not welcome strangers.' But the knight drew his sword, its blade gleaming with enchanted light.",
                "[SETTING] A bustling medieval marketplace.\n[CHARACTERS] A merchant, a thief, a guard captain\n[STORY] The marketplace hummed with activity as vendors called out their wares. Among the crowd, a hooded figure slipped between the stalls, fingers quick and light. The merchant noticed too late that his purse was gone. 'Thief!' he bellowed. The guard captain turned, eyes narrowing.",
            ] * 50  # Repeat for minimal training
        })
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized


def train(args):
    """Fine-tune GPT-2 with LoRA for story generation."""
    print(f"Fine-tuning {GENERATOR_MODEL_NAME} with LoRA")
    print(f"  LoRA rank: {args.lora_r}")
    print(f"  LoRA alpha: {args.lora_alpha}")
    print(f"  Epochs: {args.epochs}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["c_attn", "c_proj"],  # GPT-2 attention layers
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    train_dataset = prepare_dataset(args.data_path, tokenizer, args.max_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting LoRA fine-tuning...")
    trainer.train()
    
    # Save LoRA weights
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nLoRA model saved to {args.output_dir}")
    
    # Save training config for reference
    config = {
        "base_model": GENERATOR_MODEL_NAME,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
    }
    
    import json
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 with LoRA")
    parser.add_argument("--data_path", default="data/processed/stories", help="Training data path")
    parser.add_argument("--output_dir", default="models/story_generator_lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    args = parser.parse_args()
    train(args)
