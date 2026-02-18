# Semantic VAD Improvements for Moshi Wrapper

## Summary

This document describes the improvements made to the LinTO-Moshi wrapper to properly utilize the Moshi server's semantic Voice Activity Detection (VAD) capabilities for intelligent utterance boundary detection.

## Problem Statement

The original wrapper implementation:
- ❌ Ignored `Step` messages containing semantic VAD probabilities
- ❌ Used only primitive timer-based heuristics (punctuation + 1.5s delay)
- ❌ Misunderstood `Marker` events as server-generated end-of-utterance signals
- ❌ Had no way to properly detect when Moshi identified a complete utterance

## Solution

### 1. Semantic VAD Processing (NEW)

**File:** `kyutai/stt/processing/streaming.py`

Added `SemanticVADTracker` class that:
- Processes `Step` messages from Moshi server containing probability distributions
- Tracks VAD signal history for smoothing and noise reduction
- Detects end-of-utterance when VAD probabilities exceed configurable threshold
- Provides debugging capabilities with detailed logging

### 2. Intelligent Utterance Detection

The wrapper now supports three detection modes:

**Mode 1: VAD + Punctuation (Default - Recommended)**
- Waits for BOTH semantic VAD signal AND sentence-ending punctuation
- Most reliable for production use
- Set: `VAD_REQUIRE_PUNCTUATION=true`

**Mode 2: VAD Only (Aggressive)**
- Triggers on semantic VAD signal alone
- Faster response but may trigger prematurely
- Set: `VAD_REQUIRE_PUNCTUATION=false`

**Mode 3: Timer Only (Legacy Fallback)**
- Original behavior: punctuation + timer delay
- Disable VAD: `USE_SEMANTIC_VAD=false`

### 3. Configuration Options

All configuration via environment variables:

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `USE_SEMANTIC_VAD` | `true` | `true`/`false` | Enable semantic VAD |
| `VAD_THRESHOLD` | `0.5` | `0.0-1.0` | End-of-utterance probability threshold |
| `VAD_HISTORY_SIZE` | `3` | `1-10` | Consecutive high signals needed |
| `VAD_DELAY` | `0.3` | `0.1-2.0` | Post-VAD delay (seconds) |
| `VAD_REQUIRE_PUNCTUATION` | `true` | `true`/`false` | Require punctuation confirmation |
| `LOG_VAD` | `false` | `true`/`false` | Debug VAD probabilities |
| `LOG_TRANSCRIPTS` | `false` | `true`/`false` | Log finals with reason |
| `FINAL_TRANSCRIPT_DELAY` | `1.5` | `0.5-5.0` | Fallback timer (seconds) |

### 4. Documentation Updates

**Updated Files:**
- `kyutai/README.md` - Added VAD configuration section with examples
- `kyutai/PROTOCOL.md` - Clarified message types and Marker event behavior
- `kyutai/stt/processing/streaming.py` - Comprehensive inline documentation

## Technical Details

### Step Message Format

```python
{
    "type": "Step",
    "step_idx": 123,
    "prs": [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # Head 1 (6 dimensions)
        [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],  # Head 2
        [0.12, 0.22, 0.32, 0.42, 0.52, 0.62],  # Head 3
        [0.11, 0.21, 0.31, 0.41, 0.51, 0.61]   # Head 4
    ],
    "buffered_pcm": 1920
}
```

The `prs` field contains probability distributions from the model's 4 extra heads (configured in `config-stt-en_fr-hf.toml`). These represent semantic states including end-of-utterance predictions.

### VAD Detection Algorithm

```python
1. Receive Step message with prs probabilities
2. Extract maximum probability from first head: max_vad_prob = max(prs[0])
3. Add to history buffer: vad_history.append(max_vad_prob)
4. If buffer size >= VAD_HISTORY_SIZE:
   a. Calculate average: avg_vad = mean(vad_history)
   b. If avg_vad > VAD_THRESHOLD:
      - VAD triggered!
      - Check finalization conditions (punctuation, transcript, etc.)
      - Schedule final transcript with VAD_DELAY
```

### Finalization Decision Tree

```
New Word arrives
    ├─ Has punctuation?
    │   ├─ YES → Set has_punctuation = True
    │   │   ├─ USE_SEMANTIC_VAD = True?
    │   │   │   ├─ VAD already triggered?
    │   │   │   │   ├─ YES → Finalize after VAD_DELAY (fast)
    │   │   │   │   └─ NO → Wait for VAD or FINAL_TRANSCRIPT_DELAY (fallback)
    │   │   │   └─ VAD disabled → Finalize after FINAL_TRANSCRIPT_DELAY
    │   └─ NO → Continue accumulating
    │
Step message arrives (VAD update)
    ├─ VAD triggered?
        ├─ YES → Set vad_triggered = True
            ├─ VAD_REQUIRE_PUNCTUATION = True?
            │   ├─ YES → Need punctuation
            │   │   ├─ has_punctuation = True? → Finalize after VAD_DELAY
            │   │   └─ NO → Wait for punctuation
            │   └─ NO → Finalize after VAD_DELAY (aggressive)
            └─ NO → Continue monitoring
```

## Testing Recommendations

### 1. Basic Functionality Test
```bash
# Start with default settings
LOG_TRANSCRIPTS=true LOG_LEVEL=INFO PYTHONPATH=kyutai uv run python websocket/websocketserver.py
```

### 2. Debug VAD Behavior
```bash
# See VAD probabilities in real-time
LOG_VAD=true LOG_TRANSCRIPTS=true LOG_LEVEL=DEBUG PYTHONPATH=kyutai uv run python websocket/websocketserver.py
```

### 3. Adjust Sensitivity
```bash
# More sensitive (triggers earlier)
VAD_THRESHOLD=0.3 PYTHONPATH=kyutai uv run python websocket/websocketserver.py

# Less sensitive (more conservative)
VAD_THRESHOLD=0.7 PYTHONPATH=kyutai uv run python websocket/websocketserver.py
```

### 4. Aggressive Mode (No Punctuation Required)
```bash
# Finalize on VAD signal alone
VAD_REQUIRE_PUNCTUATION=false VAD_THRESHOLD=0.6 PYTHONPATH=kyutai uv run python websocket/websocketserver.py
```

## Benefits

✅ **Intelligent Detection**: Uses ML-based semantic understanding, not just timers
✅ **Faster Response**: VAD_DELAY (0.3s) vs FINAL_TRANSCRIPT_DELAY (1.5s)
✅ **Configurable**: Tune for your use case (speed vs accuracy)
✅ **Backwards Compatible**: Can disable VAD to restore original behavior
✅ **Robust Fallback**: Timer-based backup if VAD is uncertain
✅ **Better Debugging**: Detailed logging shows why finals were triggered

## Future Improvements

Potential enhancements for future iterations:

1. **Per-Head Analysis**: Analyze each of the 4 extra heads separately to understand different semantic signals
2. **Dynamic Thresholds**: Adapt VAD_THRESHOLD based on audio characteristics
3. **Multi-Language Support**: Tune VAD parameters per language
4. **Confidence Scores**: Include VAD confidence in transcript metadata
5. **Custom Head Selection**: Allow configuration of which extra head(s) to use for VAD
6. **Statistical Analysis**: Collect VAD statistics for model improvement

## Credits

Improvements based on analysis of:
- Kyutai's Moshi server implementation (`batched_asr.rs`, `asr.rs`)
- Moshi 1B model architecture with 4 extra heads (6 dimensions each)
- Delayed Streams Modeling framework documentation
