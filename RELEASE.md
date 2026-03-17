# 2.1.0
- Fix memory leak in NeMo engine
- Add local attention support for NeMo engine

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
