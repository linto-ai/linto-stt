import pytest

from helpers.env_setup import get_expected_regex
from helpers.transcription_client import transcribe_http


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------

# The `transformers` engine serves HF transformers-native ASR checkpoints. Tests
# use the smallest Qwen3-ASR family member (fastest to download and run); the
# 1.7B default (Qwen/Qwen3-ASR-1.7B-hf) shares the same interface.
TEST_MODEL = "Qwen/Qwen3-ASR-0.6B-hf"


# A falsy `device` (None/"") leaves DEVICE unset, so the server fixture fills it
# from the --device CLI option (default cpu) — i.e. the test "follows --device".
# Pass an explicit "cpu"/"cuda" only to pin a test to one device regardless of
# the CLI. GPU is strongly recommended for these LLM-based ASR models; CPU works
# but is slow.
def _config(device, env=None, id="default", mode="http"):
    env = {"MODEL": TEST_MODEL, **(env or {})}
    if device:
        env["DEVICE"] = device
    return pytest.param(
        {"engine": "transformers", "mode": mode, "port": 0, "env_overrides": env},
        id=id,
    )


# ---------------------------------------------------------------------------
# UV tests - integration
# ---------------------------------------------------------------------------


@pytest.mark.transformers
@pytest.mark.uv
class TestTransformersIntegration:
    """transformers engine (Qwen3-ASR) over HTTP via UV subprocess. Follows --device."""

    @pytest.mark.parametrize(
        "uv_server",
        [_config(None, {"LANGUAGE": "fr"}, id="fr")],
        indirect=True,
    )
    def test_integration(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


@pytest.mark.transformers
@pytest.mark.uv
@pytest.mark.gpu
class TestTransformersGPU:
    """transformers engine (Qwen3-ASR) pinned to CUDA via UV subprocess."""

    @pytest.mark.parametrize(
        "uv_server",
        [_config("cuda", {"LANGUAGE": "fr"}, id="fr-cuda")],
        indirect=True,
    )
    def test_integration_cuda(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


# ---------------------------------------------------------------------------
# UV tests - language selection
# ---------------------------------------------------------------------------


@pytest.mark.transformers
@pytest.mark.uv
class TestTransformersLanguages:
    """Language selection: automatic detection, and per-request override of the
    configured default (mirrors the LANGUAGE/language handling)."""

    @pytest.mark.parametrize(
        "uv_server",
        [_config(None, {"LANGUAGE": "*"}, id="nolanguage")],
        indirect=True,
    )
    def test_nolanguage(self, uv_server, test_audio_bonjour):
        """LANGUAGE=* → automatic detection; French audio still transcribes."""
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )

    @pytest.mark.parametrize(
        "uv_server",
        [_config(None, {"LANGUAGE": "en"}, id="language-over-config")],
        indirect=True,
    )
    def test_language_over_config(self, uv_server, test_audio_bonjour):
        """Config says en, but the request overrides to fr."""
        result = transcribe_http(
            uv_server["url"], str(test_audio_bonjour), language="fr"
        )
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


# ---------------------------------------------------------------------------
# UV tests - hotwords
# ---------------------------------------------------------------------------


# Real biasing demonstration on Hotel20sec.wav: unprompted, the 0.6B model
# mis-transcribes the spoken "chambre double" ("double room") as "chambre d'eau".
# Supplying "chambre double" as a hotword makes it emit "double" — a word absent
# from the un-biased output — which proves the context prompt steered the result.
HOTWORDS = "chambre double"
BIASED_WORD = "double"  # checked case-insensitively


@pytest.mark.transformers
@pytest.mark.uv
class TestTransformersHotwords:
    """Hotword biasing via the model's context prompt, exercised from the HOTWORDS
    env default and from the per-request `hotwords` query parameter.

    Qwen3-ASR hotwords are a soft context prompt that helps the model produce
    rare/ambiguous terms it would otherwise mis-hear. On Hotel20sec.wav the model
    renders the spoken "chambre double" as "chambre d'eau" unprompted; the
    `chambre double` hotword makes "double" appear. The baseline test confirms
    "double" is absent without the hotword, so its presence in the other two is a
    genuine biasing signal (not a word the model emits on its own)."""

    @pytest.mark.parametrize(
        "uv_server",
        [_config(None, {"LANGUAGE": "fr"}, id="baseline")],
        indirect=True,
    )
    def test_baseline_has_no_biased_word(self, uv_server, test_audio_hotel):
        """Control: without the hotword, the biased word is NOT produced."""
        result = transcribe_http(uv_server["url"], str(test_audio_hotel))
        assert get_expected_regex(str(test_audio_hotel), "fr").search(result), (
            f"Unexpected transcription: {result!r}"
        )
        assert BIASED_WORD not in result.lower(), (
            f"Baseline unexpectedly already contains {BIASED_WORD!r}: {result!r}"
        )

    @pytest.mark.parametrize(
        "uv_server",
        [_config(None, {"HOTWORDS": HOTWORDS}, id="hotwords-env")],
        indirect=True,
    )
    def test_hotwords_env(self, uv_server, test_audio_hotel):
        """The HOTWORDS env default biases the transcription toward 'double'."""
        result = transcribe_http(uv_server["url"], str(test_audio_hotel))
        assert BIASED_WORD in result.lower(), (
            f"Expected hotword-biased {BIASED_WORD!r} in transcription: {result!r}"
        )

    @pytest.mark.parametrize(
        "uv_server",
        [
            # HOTWORDS deliberately NOT set: biasing must come from the per-request
            # `hotwords` query parameter instead.
            _config(None, {"LANGUAGE": "fr"}, id="hotwords-per-request")
        ],
        indirect=True,
    )
    def test_hotwords_per_request(self, uv_server, test_audio_hotel):
        """The per-request `hotwords` param biases toward 'double' with no env set,
        proving it overrides the (empty) default."""
        result = transcribe_http(
            uv_server["url"], str(test_audio_hotel), hotwords=HOTWORDS
        )
        assert BIASED_WORD in result.lower(), (
            f"Expected hotword-biased {BIASED_WORD!r} in transcription: {result!r}"
        )


# ---------------------------------------------------------------------------
# Streaming (WebSocket) - unsupported
# ---------------------------------------------------------------------------


@pytest.mark.transformers
@pytest.mark.uv
class TestTransformersStreamingUnsupported:
    """This engine does NOT support WebSocket streaming (Qwen3-ASR streaming needs
    the vLLM backend, which is not wired here). There is therefore no positive
    streaming test; instead we assert the entry point fails loudly — wssDecode
    raises NotImplementedError — so the unsupported mode can never silently
    appear to work."""

    def test_wssdecode_raises_not_implemented(self, monkeypatch):
        # Keep the engine import cheap and offline: CPU + http means the
        # module-level warmup (which would load/download models) is skipped.
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("SERVICE_MODE", "http")
        pytest.importorskip("transformers")

        import asyncio

        from linto_stt.engines.transformers.stt.processing.streaming import wssDecode

        with pytest.raises(NotImplementedError):
            asyncio.run(wssDecode(None, None))


# ---------------------------------------------------------------------------
# Docker tests
# ---------------------------------------------------------------------------


@pytest.mark.transformers
@pytest.mark.docker
@pytest.mark.gpu
class TestTransformersDockerGPU:
    """transformers engine (Qwen3-ASR) via Docker on CUDA (validates the image build)."""

    @pytest.mark.parametrize(
        "docker_server",
        [
            pytest.param(
                {
                    "engine": "transformers",
                    "mode": "http",
                    "port": 0,
                    "dockerfile": "Dockerfile",
                    "use_gpu": True,
                    "env_overrides": {"MODEL": TEST_MODEL, "LANGUAGE": "fr"},
                },
                id="fr-cuda-docker",
            ),
        ],
        indirect=True,
    )
    def test_integration_cuda(self, docker_server, test_audio_bonjour):
        result = transcribe_http(docker_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


@pytest.mark.transformers
@pytest.mark.docker
@pytest.mark.gpu
class TestTransformersDockerCelery:
    """transformers engine via Celery task mode (Docker container + Redis broker).

    Exercises the `-m task` serving path end to end: a task is sent to the worker,
    which loads the audio via the engine's utils and runs decode (with metadata,
    so the aligner runs too)."""

    @pytest.mark.parametrize(
        "docker_server",
        [
            pytest.param(
                {
                    "engine": "transformers",
                    "mode": "task",
                    "port": 0,
                    "env_overrides": {
                        "MODEL": TEST_MODEL,
                        "LANGUAGE": "fr",
                        "SERVICES_BROKER": "redis://172.17.0.1:6379",
                    },
                },
                id="fr-task",
            ),
        ],
        indirect=True,
    )
    def test_transcribe_celery(self, docker_server, test_audio_bonjour, redis_server):
        from helpers.transcription_client import transcribe_celery

        result = transcribe_celery(
            "bonjour.wav", language="fr", broker_url=redis_server
        )
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )
