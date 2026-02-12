# Environment Variables

Reference for all environment variables used by LinTO-STT, grouped by category.

## Common (all backends)

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_NAME` | `nemo` | Backend STT (`nemo`, `whisper`, `kaldi`, `kyutai`). Also used as Docker build arg |
| `SERVICE_MODE` | `http` | Serving mode: `http`, `websocket`, `task` |
| `PORT` | `80` (Docker) / `8080` (CLI) | Listening port |
| `IP` | `0.0.0.0` (Docker) / `127.0.0.1` (CLI) | Bind address |
| `LANGUAGE` | `*` (auto-detect) | Language code (`fr`, `en`, `*` for auto) |
| `DEVICE` | `cuda` if available, else `cpu` | Compute device (`cuda`, `cpu`, `cuda:0`...) |
| `NUM_THREADS` | system default | CPU thread count (falls back to `OMP_NUM_THREADS`) |
| `OMP_NUM_THREADS` | system default | OpenMP thread count, used as fallback for `NUM_THREADS` |
| `PUNCTUATION_MODEL` | _(none)_ | Path to a recasing/punctuation model (FlauBERT/BERT) |

## Docker / Entrypoint

| Variable | Default | Description |
|----------|---------|-------------|
| `USER_ID` | `33` (entrypoint) / `1000` (.envdefault) | UID of the process inside the container |
| `GROUP_ID` | `33` (entrypoint) / `1000` (.envdefault) | GID of the process inside the container |

## Celery / Task queue

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICES_BROKER` | `redis://172.17.0.1:6379` | Redis broker URL |
| `BROKER_PASS` | _(empty)_ | Redis password |
| `CONCURRENCY` | `1` (nemo) / `2` (whisper) | Number of Celery workers |

## VAD (NeMo, Whisper)

| Variable | Default | Description |
|----------|---------|-------------|
| `VAD` | `auditok` | VAD method: `auditok`, `silero`, `false` |
| `VAD_DILATATION` | `0.5` | Padding around speech segments (seconds) |
| `VAD_MIN_SPEECH_DURATION` | `0.1` | Minimum speech duration (seconds) |
| `VAD_MAX_SILENCE_DURATION` | `0.1` | Maximum silence duration within speech (seconds). **Note:** .envdefault files define `VAD_MIN_SILENCE_DURATION` but the code reads `VAD_MAX_SILENCE_DURATION` |

## Streaming / WebSocket (NeMo, Whisper)

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMING_MIN_CHUNK_SIZE` | `0.5` | Minimum buffer size before transcription (seconds) |
| `STREAMING_BUFFER_TRIMMING_SEC` | `10.0` (nemo) / `8.0` (whisper) | Maximum buffer size (seconds) |
| `STREAMING_PAUSE_FOR_FINAL` | `1.2` (nemo .envdefault) / `1.0` (code default) | Silence before final result (seconds) |
| `STREAMING_TIMEOUT_FOR_SILENCE` | _(none)_ | Multiplier for silence/timeout detection |
| `STREAMING_FINAL_MIN_DURATION` | `2.0` | Minimum duration for a final result (seconds) |
| `STREAMING_FINAL_MAX_DURATION` | `20.0` | Maximum duration for a final result (seconds) |
| `STREAMING_MAX_WORDS_IN_BUFFER` | `5` | Max words in the streaming buffer |
| `STREAMING_MAX_PARTIAL_ACTUALIZATION_PER_SECOND` | `4` | Max frequency of partial results per second |

## NeMo-specific

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `nvidia/parakeet-tdt-0.6b-v2` | ASR model (HuggingFace ID or local path) |
| `ARCHITECTURE` | `rnnt_bpe` | Model architecture (`ctc_bpe`, `rnnt_bpe`, `hybrid_bpe`) |
| `PROMPT` | _(none)_ | Context prompt for the model |
| `LONG_FILE_THRESHOLD` | `540` | Long file split threshold (seconds) |
| `LONG_FILE_CHUNK_LEN` | `360` | Chunk size for long files (seconds) |
| `LONG_FILE_CHUNK_CONTEXT_LEN` | `5` | Overlap between chunks (seconds) |
| `DEBUG` | `0` | Streaming debug mode (`1` or `true` to enable) |

## Whisper-specific

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `large-v3` | Whisper model (size name, HuggingFace ID, or local path) |
| `PROMPT` | _(none)_ | Context prompt for the model |
| `alignment_model` | _(none)_ | Alignment model for word-level timestamps |
| `USE_ACCURATE` | `true` | Accurate decoding parameters (beam_size=5) |
| `ENABLE_STREAMING` | `false` | Enable HTTP streaming |

## Kaldi-specific

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/opt/model` | Path to the Vosk model |

## Kyutai-specific

| Variable | Default | Description |
|----------|---------|-------------|
| `KYUTAI_URL` | `ws://localhost:8080` | Moshi server WebSocket URL |
| `KYUTAI_API_KEY` | `public_token` | Kyutai API key |
| `USE_SEMANTIC_VAD` | `true` | Enable Moshi semantic VAD |
| `VAD_THRESHOLD` | `0.5` | VAD probability threshold |
| `VAD_HISTORY_SIZE` | `3` | Consecutive signals for VAD confirmation |
| `VAD_DELAY` | `0.3` | Delay after VAD trigger (seconds) |
| `VAD_REQUIRE_PUNCTUATION` | `true` | Combine VAD with punctuation detection |
| `FINAL_TRANSCRIPT_DELAY` | `1.5` | Delay before sending final transcript (seconds) |
| `LOG_TRANSCRIPTS` | `false` | Debug logging for transcripts |
| `LOG_VAD` | `false` | Debug logging for VAD |

## Hardcoded (not configurable)

These are set programmatically and cannot be overridden:

| Variable | Value | Set by |
|----------|-------|--------|
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | NeMo & Whisper `__init__.py` |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | `recasepunc.py` |
