# LinTO-STT

LinTO-STT is an API for Automatic Speech Recognition (ASR).

LinTO-STT can either be used as a standalone transcription service or deployed within a micro-services infrastructure using a message broker connector.

It supports both offline and real-time (streaming) transcriptions.

## Backends

The following STT backends are supported (see each README for backend-specific details):

| Backend | Description | Modes |
|---------|-------------|-------|
| [NeMo](linto_stt/backends/nemo/README.md) | NVIDIA NeMo toolkit | http, websocket, task |
| [Whisper](linto_stt/backends/whisper/README.md) | OpenAI Whisper models | http, websocket, task |
| [Kaldi](linto_stt/backends/kaldi/README.md) | Kaldi/Vosk toolkit | http, websocket, task |
| [Kyutai](linto_stt/backends/kyutai/README.md) | Kyutai Moshi STT wrapper | websocket only |

## Install

### Local (development)

```sh
apt install python3-pyaudio portaudio19-dev
```

```sh
uv sync --extra [kaldi|whisper|whisper-ctranslate|nemo|kyutai]
```

### Docker

#### Build

A single Dockerfile is used for all backends. Specify the backend with `--build-arg SERVICE_NAME`:

```sh
docker build -t linto-stt-nemo:latest --build-arg SERVICE_NAME=nemo .
docker build -t linto-stt-whisper:latest --build-arg SERVICE_NAME=whisper .
docker build -t linto-stt-kaldi:latest --build-arg SERVICE_NAME=kaldi .
docker build -t linto-stt-kyutai:latest --build-arg SERVICE_NAME=kyutai .
```

Or pull pre-built images:

```sh
docker pull lintoai/linto-stt-nemo
docker pull lintoai/linto-stt-whisper
docker pull lintoai/linto-stt-kaldi
```

#### Run

```sh
# HTTP mode (file transcription)
docker run -p 8080:80 -e SERVICE_MODE=http -e SERVICE_NAME=nemo \
  --env-file .env linto-stt-nemo:latest

# WebSocket mode (streaming)
docker run -p 8080:80 -e SERVICE_MODE=websocket -e SERVICE_NAME=nemo \
  --env-file .env linto-stt-nemo:latest

# Celery task mode (async via message broker)
docker run -e SERVICE_MODE=task -e SERVICE_NAME=nemo \
  -v ~/data/audio:/opt/audio \
  --env-file .env linto-stt-nemo:latest
```

## Run (local)

```sh
# HTTP / Websocket
uv run main.py -m [http|websocket] -b [kaldi|whisper|nemo|kyutai] -p [listening_port] -i [listening_ip]

# Celery
uv run main.py -m task -b [kaldi|whisper|nemo]
```

## Serving Modes

![Serving Modes](https://i.ibb.co/qrtv3Z6/platform-stt.png)

STT can be used in three ways:

- **HTTP** (`SERVICE_MODE=http`): Deploys a HTTP server with a Swagger UI. Send audio files via POST requests.
- **WebSocket** (`SERVICE_MODE=websocket`): Deploys a WebSocket server for real-time streaming transcription.
- **Celery Task** (`SERVICE_MODE=task`): Connects a Celery worker to a message broker for async processing. Requires `SERVICES_BROKER` to be set.

## Docker Options

- **GPU**: Add `--gpus all` and set `DEVICE=cuda`. On multi-GPU machines, use `CUDA_VISIBLE_DEVICES` to select a specific GPU.
- **Cache mount**: Mount a local cache folder to avoid re-downloading models each time:
  ```sh
  -v ~/.cache:/var/www/.cache
  ```
  If `USER_ID`/`GROUP_ID` are set, use `/home/appuser/.cache` instead.
- **Model volume**: Mount a local model file or folder:
  ```sh
  -v /path/to/model.nemo:/opt/model.nemo
  ```
- **User/Group**: Set `USER_ID` and `GROUP_ID` to avoid file permission issues with mounted volumes (default: `33`, www-data).

Full example:
```sh
docker run -p 8080:80 -it --name linto-stt-nemo \
  -e SERVICE_MODE=websocket \
  -e MODEL=linagora/linto_stt_fr_fastconformer \
  -e ARCHITECTURE=hybrid_bpe \
  -e DEVICE=cuda \
  -e USER_ID=$(id -u) \
  -e GROUP_ID=$(id -g) \
  --gpus all \
  -v ~/.cache:/home/appuser/.cache \
  lintoai/linto-stt-nemo
```

## API Reference

### HTTP API

#### GET /healthcheck

Returns `"1"` if the service is running.

#### POST /transcribe

Transcription endpoint.

- **Content-Type**: `multipart/form-data`
- **File**: Audio file (WAV 16bit 16kHz recommended)
- **Language** (optional query param): Override the `LANGUAGE` environment variable

Response (`Accept: application/json`):
```json
{
    "text": "This is the transcription as text",
    "words": [
        {"word": "This", "start": 0.0, "end": 0.124, "conf": 0.82341},
        ...
    ],
    "language": "en",
    "confidence-score": 0.879
}
```

With `Accept: text/plain`, returns only the raw text.

#### GET /docs

Swagger/OpenAPI interface.

### WebSocket Protocol

The streaming protocol follows these steps:

1. Client sends a JSON config: `{"config": {"sample_rate": 16000}}`
2. Client sends audio chunks (binary) → go to 3, or `{"eof": 1}` → go to 5
3. Server sends a partial `{"partial": "this is a "}` or final `{"text": "this is a transcription"}` result
4. Back to 2
5. Server sends a final result and closes the connection

Final results are triggered by punctuation marks detected by the model, silence (`STREAMING_PAUSE_FOR_FINAL`), or as a fallback by `STREAMING_FINAL_MAX_DURATION`.

### Celery Task Format

In task mode, operations are triggered via tasks sent through the message broker. A shared storage folder must be mounted to `/opt/audio` (e.g. `-v ~/data/audio:/opt/audio`).

Worker arguments: `file_path: str, with_metadata: bool`

- **file_path**: Location of the file within the shared folder
- **with_metadata**: If `True`, word timestamps and confidence are computed

Response format is the same as the HTTP JSON response.

The celery tasks can be managed using [LinTO Transcription service](https://github.com/linto-ai/linto-transcription-service).

## Punctuation Model (recasepunc)

If your model outputs lower-case text without punctuation, you can use a recasepunc model (version 0.4+) to add punctuation marks to final results.

Available models trained on [Common Crawl](http://data.statmt.org/cc-100/):
- French: [fr.24000](https://github.com/benob/recasepunc/releases/download/0.4/fr.24000)
- English: [en.22000](https://github.com/benob/recasepunc/releases/download/0.4/en.22000)
- Italian: [it.23000](https://github.com/benob/recasepunc/releases/download/0.4/it.23000)
- Chinese: [zh-Hant.17000](https://github.com/benob/recasepunc/releases/download/0.4/zh-Hant.17000)

Mount the model and set the `PUNCTUATION_MODEL` variable:
```sh
-v /path/to/fr.24000:/opt/models/fr.24000 -e PUNCTUATION_MODEL=/opt/models/fr.24000
```

## Configuration

See [ENV.md](ENV.md) for a complete reference of all environment variables.

### Backend Quick Configs

**Nemo** (French):
```
ARCHITECTURE=hybrid_bpe_rnnt
MODEL=linagora/linto_stt_fr_fastconformer
```

**Kaldi** (Vosk model):
```
MODEL_PATH=/path/to/vosk_model
MODEL_TYPE=vosk
```

**Kyutai** (requires a running [moshi server](https://github.com/kyutai-labs/delayed-streams-modeling)):
```
KYUTAI_URL=ws://localhost:9002
```

## Testing

### Manual test

```sh
curl -X POST "http://localhost:8080/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test/bonjour.wav;type=audio/wav"
```

### Automated test suite (pytest)

Install the backend you want to test along with the `test` extra:

```sh
uv sync --extra nemo --extra test
```

Example commands:

```sh
# All NeMo tests (UV only, no Docker)
uv run pytest -m nemo --uv-only

# Docker tests only
uv run pytest -m docker

# NeMo CPU, no Docker
uv run pytest test/test_nemo.py -m "not docker and not gpu"

# Whisper on GPU
uv run pytest -m whisper --device cuda

# Kaldi (requires model paths)
uv run pytest -m kaldi --kaldi-am-path /path/to/AM --kaldi-lm-path /path/to/LM
```

**CLI options:**

| Option | Description |
|--------|-------------|
| `--backend` | Only run tests for this backend (`nemo`, `whisper`, `kaldi`) |
| `--device` | Target device: `cpu` (default) or `cuda` |
| `--uv-only` | Only run UV-based tests (skip Docker) |
| `--docker-only` | Only run Docker-based tests |
| `--server-timeout` | Timeout in seconds for server startup (default: 600) |
| `--kaldi-am-path` | Path to Kaldi acoustic model |
| `--kaldi-lm-path` | Path to Kaldi language model |

**Markers:**

| Marker | Description |
|--------|-------------|
| `nemo` | Backend NeMo |
| `whisper` | Backend Whisper |
| `kaldi` | Backend Kaldi |
| `docker` | Tests that build and run a Docker container |
| `uv` | Tests via UV subprocess |
| `gpu` | Requires CUDA |
| `slow` | Tests taking > 2 minutes |

## License

This project is licensed under AGPLv3 (see LICENSE).
