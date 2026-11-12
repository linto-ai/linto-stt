# 2.2.0
- Hotwords are now a **per-request** parameter (the `hotwords` HTTP query parameter / Celery task argument) in addition to the `HOTWORDS` environment default, mirroring the existing per-request `language` handling (previously Whisper-only and settable only via the environment)
- New **HF** engine (`STT_ENGINE=hf`, HTTP and Celery task modes): Hugging Face ASR models, served with vLLM (default) or in-process `transformers`, with word timestamps. Supported models: Qwen3-ASR, Cohere Transcribe, Whisper, and on the vLLM backend Voxtral and Granite Speech. See the [engine README](linto_stt/engines/hf/README.md)
- NeMo and Whisper engines: the model-load lock file is created in the temp directory instead of the working directory
- Docker images (all engines):
    - Security: bump vulnerable Python dependencies (notably python-multipart, which fixes an arbitrary file write through the HTTP upload endpoint, plus Pillow, urllib3, anyio, certifi, msgpack, fsspec)
    - Smaller attack surface: no `uv`, no `pip` (kept for Whisper, which may install a model converter at runtime) and no setuid/setgid binaries in the runtime images; the HF GPU image keeps only the C compiler and C headers that vLLM's Triton kernels need
    - No `ffmpeg`, system `libsndfile` or `netcat` in the runtime images. Celery tasks read audio files with libsndfile (WAV, FLAC, OGG, MP3); other formats (M4A/AAC, WebM...) must be converted to WAV upstream. This changes NeMo, which previously accepted any format through ffmpeg
    - Fix Celery tasks of the `whisper-torch` variant, whose image lacked the `ffmpeg` command its file loading required

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
