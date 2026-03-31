# Sentiment Analysis Design

> **Last Updated:** 2026-04-01  
> **Module:** `src/nlu/sentiment_analyzer.py`

## 1. Overview

The Sentiment Analyzer detects emotional tone in player input, enabling the narrative to adapt its mood, pacing, and word choice to match the player's emotional state.

## 2. Architecture

```
Player Input ──► SentimentAnalyzer.analyze()
                      │
              ┌───────┴───────┐
              │ Model Loaded? │
              └───────┬───────┘
                      │
         ┌────────────┴────────────┐
         │                         │
    ┌────▼────┐             ┌──────▼──────┐
    │ Neural  │             │ Rule-Based  │
    │ Model   │             │ Fallback    │
    │         │             │             │
    │ distil- │             │ Keyword     │
    │ roberta │             │ Matching    │
    └────┬────┘             └──────┬──────┘
         │                         │
         └────────────┬────────────┘
                      ▼
         {emotion, confidence, scores}
```

## 3. Emotion Model

### 3.1 Neural Backend

- **Model:** `j-hartmann/emotion-english-distilroberta-base`
- **Labels:** 7 Ekman emotions + neutral
- **Load Strategy:** Up to 3 retry attempts with 1s delay between attempts

### 3.2 Emotion Labels

| Label | Example Triggers |
|-------|-----------------|
| `anger` | "attack", "kill", "furious", "destroy", "revenge" |
| `disgust` | "disgusting", "vile", "repulsive", "filthy" |
| `fear` | "afraid", "scared", "danger", "flee", "panic" |
| `joy` | "happy", "wonderful", "love", "celebrate", "victory" |
| `sadness` | "sad", "lost", "cry", "mourn", "lonely", "despair" |
| `surprise` | "wow", "unexpected", "shocked", "incredible" |
| `neutral` | Default when no strong emotion detected |

## 4. Rule-Based Fallback

When the neural model is unavailable, keyword matching is used:

```python
# Each emotion has a keyword list
matches = sum(1 for kw in keywords if kw in text_lower)
score = min(matches / 3.0, 1.0)  # Normalize
```

If no emotion scores above 0.1, returns `neutral` with 0.5 confidence.

## 5. Output Format

```python
{
    "emotion": "determined",     # Dominant emotion label
    "confidence": 0.7834,        # Confidence score (0.0-1.0)
    "scores": {                  # All emotion scores
        "anger": 0.05,
        "disgust": 0.02,
        "fear": 0.15,
        "joy": 0.10,
        "sadness": 0.03,
        "surprise": 0.08,
        "neutral": 0.57
    }
}
```

## 6. Integration with Game Engine

The emotion result is passed through the entire pipeline:

1. **Story Generation:** The `emotion` parameter is included in the `STORY_CONTINUE_PROMPT`, allowing the LLM to adjust narrative mood.
2. **KG Tracking:** Entity nodes can store `last_emotion` for emotional context.
3. **Debug Output:** Available in `TurnResult.nlu_debug["emotion"]`.

```python
# In GameEngine.process_turn():
emotion_result = self.sentiment.analyze(resolved)
emotion = emotion_result.get("emotion", "neutral")

story_text = self.story_gen.continue_story(
    player_input=resolved,
    intent=intent,
    kg_summary=kg_summary,
    history=history,
    emotion=emotion,  # Passed to LLM for mood adjustment
)
```

## 7. Configuration

The SentimentAnalyzer has no dedicated config parameters. It uses the shared transformers library with version compatibility checking (warns if transformers >= 4.50).

## 8. Error Handling

- **Model Load Failure:** Falls back to keyword-based emotion detection.
- **Transformers Version:** Warns if version >= 4.50 (tested range: 4.40–4.49).
- **Runtime Errors:** Returns `neutral` with 0.5 confidence on any analysis failure.

---
*Related: [entity-importance.md](entity-importance.md) for KG emotion tracking.*
