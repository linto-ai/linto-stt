# 2.1.2
- Whisper engine updates:
    - When VAD is enabled and finds no speech at all in an audio (silence, noise, music), decode nothing instead of the whole audio (which made Whisper hallucinate subtitle credits on silent chunks)

# 2.1.1
- NeMo engine updates:
    - Add asymmetric local attention support. Configurable attention context via `ATT_CONTEXT_SIZE` (left) and the new `ATT_CONTEXT_SIZE_RIGHT` (right/look-ahead), with model-dependent defaults
    - Update NeMo toolkit to 3.x (pinned to a frozen `main` commit), required by newer models
    - Native cache-aware streaming for models that support it (e.g. `nvidia/nemotron-3.5-asr-streaming-0.6b`), automatically discriminated from offline models (which keep the buffered/simulated streaming path)
    - Support prompt-conditioned models (automatic language prompt, e.g. Nemotron ASR)
    - More robust model loading (OS-level `filelock` instead of the deprecated `lockfile`)
- Whisper engine updates:
    - Fix VAD option (silero was run when VAD was enabled, even if auditok -the default- was specified)
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
