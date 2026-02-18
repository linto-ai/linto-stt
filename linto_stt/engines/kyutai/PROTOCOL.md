# Kyutai and LinTO Streaming Protocols

This document describes how the LinTO wrapper communicates with the Kyutai ASR server.

The wrapper allows any LinTO compatible application to use Kyutai by
simply pointing the WebSocket client to the usual `/streaming` route while the
backend itself forwards the requests to the Kyutai server.

## LinTO WebSocket API

A LinTO client opens a WebSocket connection to `/streaming` and sends:

1. A JSON configuration message containing at least the `sample_rate` field, e.g.
   ```json
   {"config": {"sample_rate": 16000}}
   ```
2. Raw PCM audio chunks encoded as 16‑bit little endian integers. When the
   stream ends the client sends a JSON message `{"eof": 1}`.
3. The server replies with JSON messages:
   - `{"partial": "..."}` for intermediate results.
   - `{"text": "..."}` when the transcription (utterance) is complete... well, based on silence.

## Kyutai Server Protocol

The Kyutai server exposes a WebSocket endpoint `/api/asr-streaming`. Clients must
provide a header `kyutai-api-key` containing their API key.

Data frames are encoded with MessagePack and are 24&nbsp;kHz mono floating point
values. The main message types are:

**Client → Server Messages:**
- `{"type": "Audio", "pcm": [float32, ...]}` – raw PCM samples at 24kHz.
- `{"type": "OggOpus", "data": bytes}` – Ogg/Opus encoded audio frames.
- `{"type": "Marker", "id": int}` – client-sent marker to track processing (echoed back by server after ASR delay).
- `{"type": "Init"}` – initialize/reset the connection (batched mode only).

**Server → Client Messages:**
- `{"type": "Word", "text": str, "start": float}` – decoded word with start timestamp.
- `{"type": "EndWord", "stop": float}` – marks the end of a word with stop timestamp.
- `{"type": "Step", "step_idx": int, "prs": [[float, ...], ...], "buffered_pcm": int}` – **semantic VAD probabilities**.
  - `prs`: List of probability distributions from the model's extra heads (4 heads × 6 dimensions for the 1B model)
  - The probability values represent semantic states including end-of-utterance detection
  - Sent on every inference step (~12.5 Hz)
- `{"type": "Marker", "id": int}` – server echo of client's marker after processing delay.
- `{"type": "Ready"}` – server ready to process (batched mode only).
- `{"type": "Error", "message": str}` – error message.

### Understanding Marker Events

**Important:** `Marker` messages are **client-controlled**, not automatic end-of-utterance signals from the server.

- The client sends a `Marker` to mark a point in the audio stream
- The server echoes it back after the ASR delay (accounting for model latency)
- This is useful for synchronization but NOT for automatic utterance detection
- For utterance detection, use the `Step` messages with semantic VAD probabilities

## Mapping LinTO to Kyutai

The wrapper acts as a proxy between a LinTO client and the Kyutai server:

1. **Audio Processing:**
   - The LinTO configuration is parsed to obtain the audio sample rate.
   - Each audio chunk from the client is converted from 16‑bit PCM to float32 and
     resampled to 24&nbsp;kHz when necessary.
   - Chunks are sent to the Kyutai server inside `Audio` messages. A leading
     warmup period discards initial audio to prime the model.

2. **End-of-Stream Handling:**
   - When the client sends `{"eof":1}` a `Marker` message followed by silence
     is forwarded to Kyutai to finalize transcription.

3. **Utterance Detection (NEW - Semantic VAD):**
   - `Word` messages from Kyutai are aggregated into partial transcripts and
     forwarded to the client using LinTO's `{"partial": "..."}` format.
   - `Step` messages containing semantic VAD probabilities (`prs` field) are
     processed to detect end-of-utterance events.
   - A final `{"text": ...}` message is sent when **both** of the following occur:
     - The semantic VAD signal exceeds the configured threshold (default: 0.5)
     - The last word ends with sentence-ending punctuation (`.`, `?`, `!`, `...`)
   - If VAD is disabled or `VAD_REQUIRE_PUNCTUATION=false`, the system falls back
     to timer-based or VAD-only detection respectively.

4. **Fallback Mechanism:**
   - If semantic VAD doesn't trigger within `FINAL_TRANSCRIPT_DELAY` seconds
     (default: 1.5s) after punctuation, the system sends a final transcript anyway.
   - This ensures the system remains responsive even if VAD signals are unclear.