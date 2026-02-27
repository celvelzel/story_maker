"""Train intent classifier: fine-tune RoBERTa on text adventure actions."""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings


class IntentDataset(Dataset):
    """Dataset for intent classification training."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    """Compute accuracy for evaluation."""
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def create_synthetic_training_data():
    """
    Create synthetic training data for intent classification.
    
    In production, this should be replaced with annotated data from LIGHT dataset.
    """
    data = {
        "action": [
            "attack the goblin with my sword",
            "fight the dragon",
            "defend against the incoming arrows",
            "strike the bandit leader",
            "shoot an arrow at the wolf",
            "engage the enemy soldiers",
            "push the boulder aside",
            "kick down the door",
        ],
        "dialogue": [
            "talk to the village elder",
            "speak with the merchant",
            "ask the guard about the prisoner",
            "greet the stranger",
            "tell the innkeeper about our quest",
            "chat with the blacksmith",
            "have a conversation with the wizard",
            "say hello to the fairy",
        ],
        "explore": [
            "explore the dark cave",
            "search the room for hidden items",
            "look around the village",
            "investigate the strange noise",
            "wander through the forest",
            "survey the battlefield",
            "scout ahead for danger",
            "explore the abandoned temple",
        ],
        "use_item": [
            "use the healing potion",
            "drink the magic elixir",
            "apply the antidote",
            "activate the ancient scroll",
            "equip the enchanted armor",
            "throw the smoke bomb",
            "eat the bread to restore health",
            "use the key on the locked door",
        ],
        "ask_info": [
            "what is this place",
            "where can I find the blacksmith",
            "who is the king of this land",
            "how do I cross the river",
            "why is the village abandoned",
            "describe the creature",
            "examine the ancient inscription",
            "read the letter",
        ],
        "rest": [
            "rest by the campfire",
            "sleep at the inn",
            "wait until morning",
            "set up camp for the night",
            "meditate to focus my mind",
            "take a short nap",
            "heal my wounds and rest",
            "sit down and catch my breath",
        ],
        "trade": [
            "trade with the merchant",
            "buy a sword from the blacksmith",
            "sell my old armor",
            "barter for supplies",
            "negotiate a better price",
            "offer gold for the map",
            "shop for potions",
            "exchange gems for food",
        ],
        "other": [
            "sing a song to pass the time",
            "write a letter home",
            "pray at the shrine",
            "craft a torch from branches",
            "build a shelter",
            "start a campfire",
            "dance around the bonfire",
            "do nothing",
        ],
    }
    
    texts = []
    labels = []
    label2id = {label: i for i, label in enumerate(settings.INTENT_LABELS)}
    
    for intent, examples in data.items():
        for text in examples:
            texts.append(text)
            labels.append(label2id[intent])
    
    return texts, labels


def train(args):
    """Main training function."""
    print(f"Training intent classifier with {settings.INTENT_MODEL_NAME}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(settings.INTENT_MODEL_NAME)
    
    # Prepare data
    texts, labels = create_synthetic_training_data()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.INTENT_MODEL_NAME,
        num_labels=len(settings.INTENT_LABELS),
        id2label={i: l for i, l in enumerate(settings.INTENT_LABELS)},
        label2id={l: i for i, l in enumerate(settings.INTENT_LABELS)},
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    results = trainer.evaluate()
    print(f"\nValidation Results: {results}")
    
    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train intent classifier")
    parser.add_argument("--output_dir", default="models/intent_classifier", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    train(args)
