# Environment Variables

Reference for all environment variables used by LinTO-STT, grouped by category.

## Common (all engines)

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_ENGINE` | `nemo` | STT engine to use (`nemo`, `whisper`, `kaldi`, `kyutai`, `hf`). Used as Docker build arg and CLI default |
| `SERVICE_NAME` | `stt` | Celery queue/worker name in task mode. Defaults to the engine name when not set |
| `SERVICE_MODE` | `http` | STT serving mode: `http`, `task`, `websocket` |
| `PORT` | `80` (Docker) / `8080` (CLI) | Listening port |
| `IP` | `0.0.0.0` (Docker) / `127.0.0.1` (CLI) | Bind address |
| `LANGUAGE` | `*` (auto-detect) | Default language to recognize. `*` for automatic detection, or a language code (`fr`, `en`), BCP-47 code (`fr-FR`), or language name (`French`). Can be overridden per request with the `language` query parameter (HTTP) or task argument |
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
| `HOTWORDS` | _(none)_ | Default free-text hint phrases/words to bias the transcription toward a given spelling (e.g. proper nouns, jargon). Can be overridden per request with the `hotwords` query parameter (HTTP) or task argument. Only supported with the CTranslate2/faster-whisper backend; ignored by the `whisper_timestamped` backend |
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

## HF-specific

The `hf` engine serves Hugging Face ASR models in `transformers` format (Qwen3-ASR,
Cohere Transcribe, Whisper, and on the `vllm` backend Voxtral, Granite Speech, ...).
See the engine README.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-ASR-1.7B-hf` | Transformers-native ASR checkpoint, auto-detected by `model_type`: Qwen3-ASR (`qwen3_asr`, e.g. `Qwen/Qwen3-ASR-0.6B-hf`) or a seq2seq ASR (e.g. `cohere_asr`: `CohereLabs/cohere-transcribe-03-2026`). Seq2seq models require an explicit `LANGUAGE` (no auto-detect) and have no hotwords. Needs transformers ≥ 5.13 (the `hf`/`hf-vllm` extra) |
| `ALIGNER_MODEL` | `Qwen/Qwen3-ForcedAligner-0.6B-hf` | Forced aligner for word-level timestamps (Qwen3-ASR has none natively): a Qwen3-ForcedAligner checkpoint, or `mms` for Meta's language-agnostic MMS aligner (torchaudio `MMS_FA`, any language through romanisation; weights fetched into the torch hub cache, `TORCH_HOME`). Run only on metadata requests (JSON / Celery task), never on the plain-text path. Preloaded by the startup warmup on GPU (so the first timestamped request is not slow), loaded lazily on first use otherwise. Tuned for 11 languages (`zh, en, yue, fr, de, it, ja, ko, pt, ru, es`); other languages are aligned without a language hint (timestamps are still produced, accuracy not validated). Set empty or `none` to disable (then `words` is empty for metadata requests) |
| `ALIGNER_FALLBACK_MODEL` | `mms` | Second aligner (same values as `ALIGNER_MODEL`) used for the requests the primary aligner does not cover: a language outside its set, or audio longer than its limit (5 minutes for the Qwen aligner). Loaded only on the first request that needs it (1.2 GB download for `mms` if not cached). Set to `none` to always use the primary aligner |
| `DTYPE` | `bfloat16` | Model compute dtype (`bfloat16`, `float16`, `float32`, `auto`) |
| `ATTN_IMPLEMENTATION` | _(transformers default: `sdpa`)_ | Attention kernel. The default `sdpa` already dispatches to Flash-Attention kernels on supported GPUs (bf16/fp16) via PyTorch (efficient out of the box), and it also works on CPU. Set `flash_attention_2` only if you have separately installed the `flash-attn` package (GPU-only, heavy build); gains over `sdpa` for short-audio ASR inference are usually small |
| `MAX_NEW_TOKENS` | `1280` | Cap on generated tokens per pass (per chunk for long audio). Only a safety cap: generation stops at the end-of-speech token, so a generous value is free for short audio; `1280` covers a dense 4-min chunk. Raise it (with the chunk length) if long/dense speech gets truncated |
| `LONG_FILE_THRESHOLD` | `240` | Audio longer than this (seconds) is transcribed in consecutive chunks instead of one pass, to bound VRAM and context (they grow with the audio length) and avoid truncation. Only for models that do not split long audio themselves: Qwen3-ASR on the in-process `transformers` backend, and Voxtral on the `vllm` backend (vLLM feeds it whole). No effect otherwise: the `vllm` backend chunks the other models server-side (~30 s) and seq2seq models (Cohere, Whisper) self-chunk. Independent from NeMo's same-named variable |
| `LONG_FILE_CHUNK_LEN` | `240` | Chunk length (seconds) when an audio exceeds `LONG_FILE_THRESHOLD` (same scope). Non-overlapping; word timestamps are offset per chunk. ~4 min keeps the Qwen 1.7B footprint ~8 GB in-process, stays under the aligner's ~5-min limit and within the default 4096-token context of the `vllm` backend for Voxtral; raise it (with `MAX_NEW_TOKENS`) for fewer passes when VRAM allows |
| `BACKEND` | `vllm` | Inference backend for ASR. `vllm` (default; a managed `vllm serve` subprocess talked to over HTTP; faster; serves any vLLM-servable ASR model; needs the `hf-vllm` extra and a CUDA GPU). `transformers` (in-process HF; CPU-capable; the portable fallback; does not cover `vllm`-only models such as Voxtral). **On CPU set `BACKEND=transformers`** (the default `vllm` needs a GPU). Word timestamps always use the transformers forced aligner regardless of backend |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.25` (pinned cache) / `0.6` | (`BACKEND=vllm`) With the KV cache pinned (the default), only a startup check: vLLM refuses to start unless this fraction of the card is free (0.25 by default, low enough for workers to share a card). With `VLLM_KV_CACHE_MEMORY_BYTES=0`, the fraction of the card vLLM reserves for weights and KV cache (0.6 by default) |
| `VLLM_MAX_MODEL_LEN` | `4096` | (`BACKEND=vllm`) Context-window cap of one request, kept within each model's own limit (Whisper 448, Cohere 1024). vLLM decodes audio in windows of about 30 s, far below 4096 tokens. The KV cache must hold at least one request of this length. Raise it, with the KV cache, for a model that vLLM does not window (Voxtral) |
| `VLLM_KV_CACHE_MEMORY_BYTES` | `536870912` (512 MiB) | (`BACKEND=vllm`) Absolute KV-cache budget in bytes (vLLM `kv_cache_memory_bytes`): vLLM takes the model weights plus exactly this cache, whatever the size of the card. The default fits Qwen3-ASR-1.7B and Cohere Transcribe; larger caches did not make transcription faster. Set to `0` to size the cache with `VLLM_GPU_MEMORY_UTILIZATION` instead (see the engine README, "GPU memory of the vllm backend") |
| `VLLM_ENFORCE_EAGER` | `false` | (`BACKEND=vllm`, optional) Skip torch.compile + CUDA-graph capture. Uses less VRAM and starts almost instantly (no ~minute compile) at some throughput cost, useful on memory-tight GPUs |
| `VLLM_SERVE_HOST` | `127.0.0.1` | (`BACKEND=vllm`) Host the managed `vllm serve` subprocess binds to (loopback by default; the engine talks to it in-process) |
| `VLLM_SERVE_PORT` | _(auto)_ | (`BACKEND=vllm`) Port for the managed `vllm serve` subprocess. Unset picks a free port per server (so a leftover/orphaned server never blocks a new one); set it to pin a fixed port |
| `VLLM_SERVE_STARTUP_TIMEOUT` | `900` | (`BACKEND=vllm`) Seconds to wait for the server to become healthy (model load) before failing |
| `VLLM_SERVE_REQUEST_TIMEOUT` | `600` | (`BACKEND=vllm`) Per-request HTTP timeout in seconds (long audio is chunked server-side) |
| `VLLM_SERVE_EXTRA_ARGS` | _(unset)_ | (`BACKEND=vllm`, optional) Extra arguments appended verbatim to the `vllm serve` command line (e.g. `--max-num-seqs 8`) |
| `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` | `512` | (`BACKEND=vllm`) vLLM's limit on the size of one uploaded audio (vLLM's own default, 25, rejects about 13 minutes of 16 kHz 16-bit WAV) |
| `VLLM_MAX_AUDIO_DECODE_DURATION_S` | `14400` | (`BACKEND=vllm`) vLLM's limit on the duration of one audio in seconds (vLLM's own default, 600, rejects files over 10 minutes) |
| `VLLM_MAX_AUDIO_DECODE_BYTES` | `1073741824` (1 GiB) | (`BACKEND=vllm`) vLLM's limit on the size of one audio once decoded (float32 samples) |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | (`BACKEND=vllm`) Disables vLLM's FlashInfer sampler, whose compilation at startup needs the CUDA toolkit (`nvcc`), absent from the image. No effect on the output (greedy decoding) |

`LANGUAGE` and `HOTWORDS` behave as in the Common section (per-request overridable); `HOTWORDS` is passed to the model as a biasing prompt. WebSocket/streaming is not supported for this engine.

## Hardcoded (not configurable)

These are set programmatically and cannot be overridden:

| Variable | Value | Set by |
|----------|-------|--------|
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | NeMo & Whisper `__init__.py` |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | `recasepunc.py` |
