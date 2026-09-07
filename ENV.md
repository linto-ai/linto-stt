# Environment Variables

Reference for all environment variables used by LinTO-STT, grouped by category.

## Common (all engines)

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_ENGINE` | `nemo` | STT engine to use (`nemo`, `whisper`, `kaldi`, `kyutai`). Used as Docker build arg and CLI default |
| `SERVICE_NAME` | `stt` | Celery queue/worker name in task mode. Defaults to the engine name when not set |
| `SERVICE_MODE` | `http` | STT serving mode: `http`, `task`, `websocket` |
| `PORT` | `80` (Docker) / `8080` (CLI) | Listening port |
| `IP` | `0.0.0.0` (Docker) / `127.0.0.1` (CLI) | Bind address |
| `LANGUAGE` | `*` (auto-detect) | Language to recognize. `*` for automatic detection, or a language code (`fr`, `en`), BCP-47 code (`fr-FR`), or language name (`French`) |
| `DEVICE` | `cuda` if available, else `cpu` | Device to use for the model (`cpu`, `cuda`). By default, GPU/CUDA is used if available, CPU otherwise |
| `CUDA_VISIBLE_DEVICES` | _(all)_ | GPU device index to use when running on GPU/CUDA. Recommended to also set `CUDA_DEVICE_ORDER=PCI_BUS_ID` on multi-GPU machines |
| `NUM_THREADS` | `torch.get_num_threads()` | Number of threads (maximum) to speed up transcription when running on CPU. Falls back to `OMP_NUM_THREADS` |
| `OMP_NUM_THREADS` | system default | OpenMP thread count, used as fallback for `NUM_THREADS` |
| `PUNCTUATION_MODEL` | _(none)_ | Path to a recasepunc model, for recovering punctuation and upper case letters in streaming |

## Docker / Entrypoint

| Variable | Default | Description |
|----------|---------|-------------|
| `USER_ID` | `33` (entrypoint) / `1000` (.envdefault) | UID of the process inside the container |
| `GROUP_ID` | `33` (entrypoint) / `1000` (.envdefault) | GID of the process inside the container |

## Celery / Task queue

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICES_BROKER` | `redis://172.17.0.1:6379` | URL of the message broker (Redis, RabbitMQ, Amazon SQS) |
| `BROKER_PASS` | _(empty)_ | Broker password |
| `CONCURRENCY` | `1` (nemo) / `2` (whisper) | Maximum number of parallel requests plus one. `CONCURRENCY=0` means 1 worker, `CONCURRENCY=1` means 2 workers, etc. |

## VAD (NeMo, Whisper)

| Variable | Default | Description |
|----------|---------|-------------|
| `VAD` | `auditok` | Voice Activity Detection method. VAD detects human speech in an audio stream. Use `false` to disable. Values: `auditok`, `silero`, `false` |
| `VAD_DILATATION` | `0.5` | How much (in seconds) to enlarge each speech segment detected by the VAD |
| `VAD_MIN_SPEECH_DURATION` | `0.1` | Minimum duration (in seconds) of a speech segment |
| `VAD_MIN_SILENCE_DURATION` | `0.1` | Minimum duration (in seconds) of a silence segment |

## Streaming / WebSocket (NeMo, Whisper)

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMING_MIN_CHUNK_SIZE` | `0.5` | Minimal size of the buffer (in seconds) before transcribing. Used to lower hardware usage (low value = high usage, high value = low usage) |
| `STREAMING_BUFFER_TRIMMING_SEC` | `10.0` (nemo) / `8.0` (whisper) | Maximum targeted length of the buffer (in seconds). Tries to cut after a transcription has been made (bigger value = higher hardware usage) |
| `STREAMING_PAUSE_FOR_FINAL` | `1.2` (nemo .envdefault) / `1.0` (code default) | Minimum duration of silence (in seconds) needed between words to output a final. Used if no punctuation marks are found in text |
| `STREAMING_TIMEOUT_FOR_SILENCE` | _(none)_ | If VAD is applied externally, allows the server to detect silence. Packet duration is determined from the first packet; if a packet is not received during `packet_duration * STREAMING_TIMEOUT_FOR_SILENCE` it considers silence is present. Value should be between 1 and 2 |
| `STREAMING_FINAL_MIN_DURATION` | `2.0` | Minimum duration of a final result (seconds) |
| `STREAMING_FINAL_MAX_DURATION` | `20.0` | Maximum duration of a final result (seconds). Fallback when no punctuation or silence triggers a final |
| `STREAMING_MAX_WORDS_IN_BUFFER` | `5` | How many words can stay in the buffer (i.e. how many words can be changed). Default is 4 in NeMo README |
| `STREAMING_MAX_PARTIAL_ACTUALIZATION_PER_SECOND` | `4` | Maximum number of messages the server can send to the client per second. Set to 0 to deactivate |
| `STREAMING_NATIVE_PARTIAL_INTERVAL` | `2.0` | (NeMo, native cache-aware streaming only) Cadence, in seconds of received audio, between partial results. Larger = fewer partials, less compute. Does not apply to the buffered streaming path used for offline models (the `STREAMING_*` settings above) |

## NeMo-specific

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `nvidia/parakeet-tdt-0.6b-v2` | Path to a NeMo model or HuggingFace identifier |
| `ARCHITECTURE` | `rnnt_bpe` | Architecture of the model. Supported: `ctc_bpe`, `rnnt_bpe`, `hybrid_bpe`. Hybrid models can use `hybrid_bpe_ctc` or `hybrid_bpe_rnnt` variants |
| `ATT_CONTEXT_SIZE` | _(model-dependent)_ | Left attention context, in encoder frames (~80 ms each). Resolved at load time: offline models default to `128` (local attention via `rel_pos_local_attn`); cache-aware streaming models keep their trained context unless this is set |
| `ATT_CONTEXT_SIZE_RIGHT` | _(model-dependent)_ | Right (look-ahead) attention context, in encoder frames. Larger = better accuracy but higher latency. Defaults: offline models use the same value as `ATT_CONTEXT_SIZE`; cache-aware streaming models use `0`. For cache-aware models only specific values are valid (model-specific, e.g. `nvidia/nemotron-3.5-asr-streaming-0.6b` supports `0, 3, 6, 13`) |
| `PROMPT` | _(none)_ | Context prompt for the model |
| `LONG_FILE_THRESHOLD` | `540` | A file longer than this (in seconds) will be split into smaller chunks to avoid Out of Memory issues. Depends on VRAM/RAM |
| `LONG_FILE_CHUNK_LEN` | `360` | For long file transcription, size of the chunks (in seconds) into which the audio is split. Depends on VRAM/RAM |
| `LONG_FILE_CHUNK_CONTEXT_LEN` | `5` | For long file transcription, context added at the beginning and end of each chunk (in seconds) to avoid losing words at boundaries |
| `DEBUG` | `0` | Streaming debug mode (`1` or `true` to enable) |

## Whisper-specific

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `large-v3` | Path to a Whisper model, type of Whisper model used, or HuggingFace identifier |
| `PROMPT` | _(none)_ | Initial prompt for the Whisper model (faster-whisper's `initial_prompt`): free text prepended as previous context to encourage a certain transcription style |
| `HOTWORDS` | _(none)_ | Free-text hint phrases/words to bias the transcription toward a given spelling (e.g. proper nouns, jargon). Only supported with the CTranslate2/faster-whisper backend; ignored by the `whisper_timestamped` backend |
| `ALIGNMENT_MODEL` | _(none)_ | (Deprecated) Path to a wav2vec model for word alignment, or HuggingFace repository name or torchaudio pipeline |
| `USE_ACCURATE` | `true` | Use more expensive parameters for better transcriptions (but slower). Uses beam_size=5 |
| `ENABLE_STREAMING` | `false` | (Legacy) For the HTTP mode, redirects to websocket mode if enabled |

## Kaldi-specific

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TYPE` | _(required)_ | Type of STT model used: `lin` (LinTO acoustic + language models) or `vosk` (Vosk all-in-one model) |
| `MODEL_PATH` | `/opt/model` | Path to the Vosk model (when `MODEL_TYPE=vosk`) |

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
