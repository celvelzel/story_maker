"""Data download scripts for StoryWeaver training data."""
import os
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def download_light_dataset():
    """
    Download the LIGHT dataset from Facebook Research.
    
    LIGHT: ~110K dialogues in fantasy text adventure settings.
    Contains: scenes, characters, objects, dialogue, actions.
    
    URL: https://parl.ai/projects/light/
    
    Usage:
        pip install parlai
        parlai display_data -t light_dialog
    """
    print("=== LIGHT Dataset ===")
    print("The LIGHT dataset can be obtained via ParlAI:")
    print("  pip install parlai")
    print("  parlai display_data -t light_dialog")
    print(f"  Save to: {DATA_DIR / 'light'}")
    print()
    
    # Alternative: download directly
    try:
        from datasets import load_dataset
        dataset = load_dataset("light_dialog", trust_remote_code=True)
        save_path = DATA_DIR / "light"
        save_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(save_path))
        print(f"LIGHT dataset saved to {save_path}")
    except Exception as e:
        print(f"Auto-download failed: {e}")
        print("Please download manually from ParlAI.")


def download_writing_prompts():
    """
    Download the WritingPrompts dataset (Reddit).
    
    ~300K prompt-story pairs, good for story generation training.
    
    URL: https://huggingface.co/datasets/euclaise/writingprompts
    """
    print("=== WritingPrompts Dataset ===")
    try:
        from datasets import load_dataset
        dataset = load_dataset("euclaise/writingprompts")
        save_path = DATA_DIR / "writingprompts"
        save_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(save_path))
        print(f"WritingPrompts saved to {save_path}")
    except Exception as e:
        print(f"Download failed: {e}")


def download_roc_stories():
    """
    Download ROCStories / StoryCloze dataset.
    
    ~100K five-sentence stories, good for narrative coherence.
    
    Note: Requires registration at https://cs.rochester.edu/nlp/rocstories/
    """
    print("=== ROCStories Dataset ===")
    print("ROCStories requires manual registration:")
    print("  https://cs.rochester.edu/nlp/rocstories/")
    print(f"  Save to: {DATA_DIR / 'rocstories'}")
    print()
    
    # Try HuggingFace alternative
    try:
        from datasets import load_dataset
        dataset = load_dataset("story_cloze", "2016", data_dir=str(DATA_DIR / "rocstories"))
        save_path = DATA_DIR / "rocstories"
        save_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(save_path))
        print(f"StoryCloze saved to {save_path}")
    except Exception as e:
        print(f"Auto-download failed: {e}")
        print("Please register and download manually.")


def download_all():
    """Download all available datasets."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Downloading StoryWeaver training data...\n")
    download_writing_prompts()
    print()
    download_light_dataset()
    print()
    download_roc_stories()
    print("\nDone! Check the data/raw directory.")


if __name__ == "__main__":
    download_all()
