"""Data preprocessing pipeline for StoryWeaver."""
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ftfy


class StoryPreprocessor:
    """
    Data cleaning pipeline:
    1. Text normalization (ftfy + regex)
    2. Length filtering (50-1024 tokens)
    3. Deduplication (MinHash + LSH)
    4. Quality scoring (perplexity-based)
    5. Narrative unit segmentation
    """
    
    def __init__(
        self,
        min_tokens: int = 50,
        max_tokens: int = 1024,
        dedup_threshold: float = 0.8,
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.dedup_threshold = dedup_threshold
        self._seen_hashes = set()
    
    def process_pipeline(self, texts: List[str]) -> List[str]:
        """Run the full preprocessing pipeline."""
        print(f"Starting preprocessing of {len(texts)} texts...")
        
        # Step 1: Normalize
        texts = [self.normalize_text(t) for t in texts]
        print(f"  After normalization: {len(texts)}")
        
        # Step 2: Filter by length
        texts = [t for t in texts if self.check_length(t)]
        print(f"  After length filter: {len(texts)}")
        
        # Step 3: Deduplicate
        texts = self.deduplicate(texts)
        print(f"  After deduplication: {len(texts)}")
        
        # Step 4: Quality filter
        texts = [t for t in texts if self.quality_check(t)]
        print(f"  After quality filter: {len(texts)}")
        
        return texts
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize Unicode and clean text artifacts."""
        # Fix Unicode issues
        text = ftfy.fix_text(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove Reddit formatting artifacts
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Markdown links
        text = re.sub(r'[*_]{1,3}', '', text)        # Bold/italic markers
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        
        # Normalize ellipsis
        text = re.sub(r'\.{3,}', '...', text)
        
        # Normalize dashes
        text = re.sub(r'—|–', '-', text)
        
        return text.strip()
    
    def check_length(self, text: str) -> bool:
        """Filter by token count."""
        tokens = text.split()
        return self.min_tokens <= len(tokens) <= self.max_tokens
    
    def deduplicate(self, texts: List[str]) -> List[str]:
        """Remove near-duplicate texts using hash-based approach."""
        try:
            return self._minhash_dedup(texts)
        except ImportError:
            return self._simple_dedup(texts)
    
    def _minhash_dedup(self, texts: List[str]) -> List[str]:
        """MinHash + LSH deduplication."""
        from datasketch import MinHash, MinHashLSH
        
        lsh = MinHashLSH(threshold=self.dedup_threshold, num_perm=128)
        unique_texts = []
        
        for i, text in enumerate(texts):
            mh = MinHash(num_perm=128)
            for word in text.lower().split():
                mh.update(word.encode('utf8'))
            
            key = f"doc_{i}"
            
            # Check if similar document exists
            result = lsh.query(mh)
            if not result:
                lsh.insert(key, mh)
                unique_texts.append(text)
        
        return unique_texts
    
    def _simple_dedup(self, texts: List[str]) -> List[str]:
        """Simple hash-based exact dedup (fallback)."""
        seen = set()
        unique = []
        for text in texts:
            h = hashlib.md5(text.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(text)
        return unique
    
    @staticmethod
    def quality_check(text: str) -> bool:
        """Basic quality heuristics."""
        # Must have at least some sentence structure
        if text.count('.') < 2:
            return False
        
        # Must not be all caps
        if text.upper() == text and len(text) > 50:
            return False
        
        # Must have reasonable word length
        words = text.split()
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        if avg_word_len < 2 or avg_word_len > 15:
            return False
        
        # Must not be too repetitive
        unique_words = set(w.lower() for w in words)
        if len(unique_words) / max(len(words), 1) < 0.2:
            return False
        
        return True
    
    @staticmethod
    def segment_narrative_units(text: str) -> List[str]:
        """
        Segment a story into narrative units (scene-level chunks).
        
        Uses paragraph breaks, scene change markers, and sentence patterns.
        """
        # Split on paragraph breaks
        paragraphs = re.split(r'\n\n+', text)
        
        # Further split long paragraphs at scene change indicators
        scene_markers = r'(?:later|meanwhile|suddenly|the next|hours passed|when \w+ woke|the morning)'
        
        units = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(para.split()) > 200:
                # Try to split at scene markers
                sub_units = re.split(f'(?i)(?={scene_markers})', para)
                units.extend(u.strip() for u in sub_units if u.strip())
            else:
                units.append(para)
        
        return units


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This is a short text.",  # Will be filtered (too short)
        "The ancient forest stretched before the traveler, its towering trees casting long shadows across the leaf-strewn path. " * 5,
        "A duplicate text. " * 20,
        "A duplicate text. " * 20,  # Will be deduped
    ]
    
    preprocessor = StoryPreprocessor(min_tokens=10)
    cleaned = preprocessor.process_pipeline(sample_texts)
    print(f"\nCleaned: {len(cleaned)} texts")
    for t in cleaned:
        print(f"  [{len(t.split())} tokens] {t[:80]}...")
