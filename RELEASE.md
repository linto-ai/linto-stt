# 2.2.0
- Hotwords are now a **per-request** parameter (the `hotwords` HTTP query parameter / Celery task argument) in addition to the `HOTWORDS` environment default, mirroring the existing per-request `language` handling (previously Whisper-only and settable only via the environment)
- New **HF** engine (`STT_ENGINE=hf`): speech-to-text with Hugging Face `transformers`-native ASR models (HTTP and Celery task modes). Per-model capabilities (language auto-detection, forced language, hotwords) are auto-detected from the model's processor (not its name), so an unsupported request is rejected with a clear error. Word-level timestamps come from a companion forced aligner, run only when timestamps are requested, for every model and language: the [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf) (`ALIGNER_MODEL`, default) or Meta's language-agnostic MMS aligner (`ALIGNER_MODEL=mms`), which by default also serves as a fallback (`ALIGNER_FALLBACK_MODEL`, loaded on first need) for the languages and the long files (over 5 minutes) the Qwen aligner does not cover. Supported models:
    - **Decoder-only LLM ASR** (e.g. Qwen3-ASR: `Qwen/Qwen3-ASR-1.7B-hf` default, `-0.6B-hf`): multilingual auto-detection or forced language, hotwords (via the model's context prompt), engine-side long-audio chunking (`LONG_FILE_THRESHOLD`/`LONG_FILE_CHUNK_LEN`)
    - **Encoder-decoder (seq2seq) ASR** (e.g. Cohere Transcribe `CohereLabs/cohere-transcribe-03-2026`): forced language, no hotwords, requires an explicit language (no auto-detection), self-chunks long audio
    - **Whisper** (`openai/whisper-large-v3`, `-turbo`) on both backends: forced language or auto-detection, long audio decoded server-side (vLLM) or with transformers' sequential long-form decoding
    - **More models via the `vllm` backend** (serve-only for now, from the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) / [LINAGORA benchmark](https://linagora-labs.github.io/asr_benchmark/)): Voxtral (`mistralai/Voxtral-Mini-3B-2507`, `-Small-24B-2507`), Granite-Speech (`ibm-granite/granite-speech-4.1-2b`), MOSS-Transcribe-Diarize. They run on the `vllm` backend with no code change (capabilities auto-detected); their in-process `transformers` path is not wired
    - Swappable ASR backend via `BACKEND`, sharing language/hotwords/chunking and the forced-aligner timestamps: `vllm` (default; a managed `vllm serve` subprocess over HTTP, GPU, higher throughput, serves any vLLM-servable ASR model) and `transformers` (in-process HF, CPU-capable)
    - Efficient by default: `sdpa` attention (FlashAttention-2 kernels on supported GPUs) and bfloat16
- Docker images (all engines):
    - Security: bump vulnerable Python dependencies (notably python-multipart, which fixes an arbitrary file write through the HTTP upload endpoint, plus Pillow, urllib3, anyio, certifi, msgpack, fsspec)
    - Smaller attack surface: no `uv`, no `pip` (kept for Whisper, which may install a model converter at runtime) and no setuid/setgid binaries in the runtime images; the HF GPU image keeps only the C compiler and C headers that vLLM's Triton kernels need
    - No `ffmpeg`, system `libsndfile` or `netcat` in the runtime images. Celery tasks read audio files with libsndfile (WAV, FLAC, OGG, MP3); other formats (M4A/AAC, WebM...) must be converted to WAV upstream. This changes NeMo, which previously accepted any format through ffmpeg
    - Fix Celery tasks of the `whisper-torch` variant, whose image lacked the `ffmpeg` command its file loading required
    - Smaller HF GPU image (from 10.8 GB to about 7.6 GB): no CUDA 12 libraries outside Whisper, no unused vLLM kernel, video or toolchain packages, no `ffmpeg`

# 2.1.1
- NeMo engine updates:
    - Add asymmetric local attention support. Configurable attention context via `ATT_CONTEXT_SIZE` (left) and the new `ATT_CONTEXT_SIZE_RIGHT` (right/look-ahead), with model-dependent defaults
    - Update NeMo toolkit to 3.x (pinned to a frozen `main` commit), required by newer models
    - Native cache-aware streaming for models that support it (e.g. `nvidia/nemotron-3.5-asr-streaming-0.6b`), automatically discriminated from offline models (which keep the buffered/simulated streaming path)
    - Support prompt-conditioned models (automatic language prompt, e.g. Nemotron ASR)
    - More robust model loading (OS-level `filelock` instead of the deprecated `lockfile`)
- Whisper engine updates:
    - Fix VAD option (silero was run when VAD was enabled, even if auditok -the default- was specified)
    - When VAD is enabled and finds no speech at all in an audio (silence, noise, music), decode nothing instead of the whole audio (which made Whisper hallucinate subtitle credits on silent chunks)
    - Use of LinTO faster-whisper fork, to support more whisper models (French finetuned & distilled model)

# 2.1.0
- NeMo engine updates:
    - Fix memory leak
    - Fix timestamps when VAD is enabled
    - Add support for specified (source and target) language
    - Add local attention support
- Whisper engine updates:
    - Fix issue #128 (speech dropped with empty or very short initial prompt)
    - Implement hotwords feature (environment variable HOTWORDS)

# 2.0.0
- Project restructuring: all engines (whisper, nemo, kaldi, kyutai) unified under a single `linto_stt` Python package
- Migration to [UV](https://docs.astral.sh/uv/) for dependency management, replacing `requirements.txt` with `pyproject.toml` and `uv.lock`
- Single Dockerfile for all engines (build arg `STT_ENGINE`)
- Unified entry point: `uv run main.py -m [http|websocket|task] -e [engine]`
- Python 3.12+ required
- Comprehensive test suite with `pytest` instead of `unittest`

## Previous releases (1.x)

Prior to 2.0.0, each DockerFile (now engine) had its own independent versioning:

- [Whisper releases](linto_stt/engines/whisper/RELEASE.md) (up to 1.0.7)
- [NeMo releases](linto_stt/engines/nemo/RELEASE.md) (up to 1.0.2)
- [Kaldi releases](linto_stt/engines/kaldi/RELEASE.md) (up to 1.1.0)
- [Kyutai releases](linto_stt/engines/kyutai/RELEASE.md) (up to 0.1.0)
