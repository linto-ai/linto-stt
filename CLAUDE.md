# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LinTO-STT is a modular Speech-to-Text system with pluggable backends (NeMo, Kyutai) and three serving modes (HTTP, WebSocket streaming, Celery async tasks). Uses UV for dependency management.

## Commands

### Installation
```bash
# Install with NeMo backend (primary)
uv sync --extra nemo

# System dependencies for audio
apt install python3-pyaudio portaudio19-dev
```

### Running the Server
```bash
# HTTP mode (REST API)
uv run main.py -m http -b nemo -p 8080 -i 127.0.0.1

# WebSocket mode (real-time streaming)
uv run main.py -m websocket -b nemo -p 8001

# Celery worker mode (async task queue)
uv run main.py -m task -b nemo
```

### Testing
```bash
# Test HTTP transcription
curl -X POST "http://localhost:8080/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test/bonjour.wav;type=audio/wav"

# Test Celery (requires Redis running)
python test-celery.py
```

## Architecture

```
main.py                         # Entry point
linto_stt/
├── __init__.py                 # CLI parser, mode dispatching, backend loading
├── http_server/                # FastAPI REST API (/transcribe, /healthcheck)
├── websocket/websocketserver.py # WebSocket streaming server
├── celery/                     # Celery task broker config
├── punctuation/recasepunc.py   # Post-processing: recasing & punctuation
└── backends/
    ├── nemo/                   # NVIDIA NeMo backend (primary)
    │   ├── .envdefault         # Default configuration
    │   └── stt/processing/
    │       ├── __init__.py     # Model loading, warmup, decode export
    │       ├── decoding.py     # Main transcription logic
    │       ├── streaming.py    # WebSocket streaming handler
    │       ├── vad.py          # Voice Activity Detection (auditok/silero)
    │       ├── load_model.py   # HuggingFace/local model loading
    │       ├── text_normalize.py # Text post-processing
    │       └── utils.py        # Audio loading utilities
    └── kyutai/                 # Kyutai Moshi wrapper (external server)
        └── stt/processing/
            ├── streaming.py    # Semantic VAD wrapper
            └── moshi_client.py # File transcription client
```

### Backend Loading
Backends are dynamically imported via `import_stt_module(backend, submodule)`. Each backend must expose:
- `MODEL` - loaded model instance
- `USE_GPU` - boolean flag
- `decode(audio, model, with_metadata, language)` - transcription function
- `load_wave_buffer(buffer)` - audio loading
- `warmup()` - model initialization

### Processing Pipeline (NeMo)
1. Audio loading → 16kHz mono normalization
2. VAD (optional) → Remove silence segments
3. Long file handling → Split files > 540s into 360s chunks with 5s overlap
4. Model inference → Transcription with word-level timestamps
5. Text normalization → Punctuation, emoji removal, formatting
6. Recasing (optional) → FlauBERT/BERT-based punctuation restoration

## Configuration

Environment variables configured via `.env` file or `backends/nemo/.envdefault`:

**Model Selection:**
```bash
MODEL=nvidia/parakeet-tdt-0.6b-v2  # English (default)
ARCHITECTURE=rnnt_bpe

# French alternative:
MODEL=linagora/linto_stt_fr_fastconformer
ARCHITECTURE=hybrid_bpe
```

**Key Parameters:**
- `VAD=auditok|silero|false` - Voice Activity Detection method
- `DEVICE=cuda|cpu|cuda:0` - Compute device
- `LONG_FILE_THRESHOLD=540` - Split files longer than this (seconds)
- `STREAMING_PAUSE_FOR_FINAL=1.2` - Silence duration before final transcript

## API Formats

**HTTP POST /transcribe:**
- Request: `multipart/form-data` with `file` field, optional `?language=` param
- Response: JSON with `text`, `confidence-score`, `words[]` (timestamps) or plain text based on `Accept` header

**WebSocket Streaming:**
- Config message: `{"config": {"sample_rate": 16000}}`
- Audio: Binary PCM frames
- Responses: `{"partial": "..."}` or `{"text": "..."}` (final)

## Key Differences Between Backends

| Feature | NeMo | Kyutai |
|---------|------|--------|
| Sample rate | 16kHz | 24kHz |
| VAD | auditok/silero | Semantic VAD |
| Deployment | Self-contained | External server wrapper |
| Streaming | Native | WebSocket proxy |
