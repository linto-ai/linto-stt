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

- `{"type": "Audio", "pcm": [float32, ...]}` – raw PCM samples.
- `{"type": "OggOpus", "data": bytes}` – Ogg/Opus encoded audio frames.
- `{"type": "Marker", "id": int}` – marks the end of an utterance.
- `{"type": "Word", "text": str, "start": float, "end": float}` – decoded words.

The server starts streaming `Word` events as it processes incoming audio and
finally sends a `Marker` event. Seems that this event is not sent at the end of an utterance. Still needs investigations.

## Mapping LinTO to Kyutai

The wrapper acts as a proxy between a LinTO client and the Kyutai server:

1. The LinTO configuration is parsed to obtain the audio sample rate.
2. Each audio chunk from the client is converted from 16‑bit PCM to float32 and
   resampled to 24&nbsp;kHz when necessary.
3. Chunks are sent to the Kyutai server inside `Audio` messages. A leading
   second of silence is sent to prime the model.
4. When the client sends `{"eof":1}` a `Marker` message followed by a short
   silence is forwarded to Kyutai so that it finalises transcription.
5. `Word` messages received from Kyutai are aggregated into partial transcripts
   that are forwarded to the client using LinTO's JSON format. A final
   `{"text": ...}` message is sent when a pause of 2 seconds is detected and the
   last word of the sentence ends with a punctuation mark (e.g. `.`, `?`, `!`).
   This is because the Kyutai ASR doesn't always send a `Marker` event to
   signal the end of an utterance.