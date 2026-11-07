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

# A second model family (Cohere, a Whisper-like seq2seq ASR): a different
# processor API, no hotwords, a required language, its own long-audio chunking.
COHERE_MODEL = "CohereLabs/cohere-transcribe-03-2026"

# Hotwords biasing demo on Hotel20sec.wav: unprompted, the 0.6B model mis-hears
# the spoken "chambre double" ("double room") as "chambre d'eau". Supplying
# "chambre double" as a hotword makes it emit "double" -- a word absent from the
# un-biased output -- which proves the context prompt steered the result.
HOTWORDS = "chambre double"
BIASED_WORD = "double"  # checked case-insensitively


# A falsy `device` (None/"") leaves DEVICE unset, so the server fixture fills it
# from the --device CLI option (default cpu) -- i.e. the test "follows --device".
# Pass an explicit "cpu"/"cuda" to pin a test to one device. BACKEND is NOT set
# here, so tests follow the engine default (BACKEND=vllm); tests that want the
# in-process transformers backend pass BACKEND=transformers in `env`.
def _config(device, env=None, id="default", mode="http"):
    env = {"MODEL": TEST_MODEL, **(env or {})}
    if device:
        env["DEVICE"] = device
    return pytest.param(
        {"engine": "transformers", "mode": mode, "port": 0, "env_overrides": env},
        id=id,
    )


def _cohere_config(language, id, backend="vllm"):
    env = {"MODEL": COHERE_MODEL, "LANGUAGE": language, "DEVICE": "cuda", "BACKEND": backend}
    return pytest.param(
        {"engine": "transformers", "mode": "http", "port": 0, "env_overrides": env},
        id=id,
    )


def _vllm_installed():
    import importlib.util

    return importlib.util.find_spec("vllm") is not None


# ---------------------------------------------------------------------------
# Capability detection (CPU unit test, no model weights)
# ---------------------------------------------------------------------------


@pytest.mark.transformers
@pytest.mark.uv
class TestTransformersCapabilities:
    """Model-capability detection from the processor signature (NOT the model
    name). Cheap: loads only the processors (no model weights, no GPU)."""

    def test_capabilities_detected_from_processor(self, monkeypatch):
        # Importing the engine package runs its module-level warmup on GPU, which
        # with the default BACKEND=vllm would start a vLLM server. Force CPU + the
        # transformers backend so the import stays cheap (no warmup, no server).
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("SERVICE_MODE", "http")
        monkeypatch.setenv("BACKEND", "transformers")
        pytest.importorskip("transformers")
        pytest.importorskip("librosa")  # Cohere's processor needs it

        from transformers import AutoProcessor

        from linto_stt.engines.transformers.stt.processing.backends import (
            detect_capabilities,
        )

        qwen = detect_capabilities(AutoProcessor.from_pretrained(TEST_MODEL))
        assert qwen == {"hotwords": True, "language_spec": True, "language_auto": True}

        cohere = detect_capabilities(AutoProcessor.from_pretrained(COHERE_MODEL))
        assert cohere == {"hotwords": False, "language_spec": True, "language_auto": False}


# ---------------------------------------------------------------------------
# Feature coverage on the DEFAULT backend (BACKEND=vllm)
# ---------------------------------------------------------------------------


@pytest.mark.transformers
@pytest.mark.vllm
@pytest.mark.uv
@pytest.mark.gpu
@pytest.mark.skipif(not _vllm_installed(), reason="vllm extra not installed")
class TestTransformersVllm:
    """All features on the default backend (`vllm`, a managed `vllm serve`
    subprocess): automatic language detection, forced language (config + per
    request), and hotwords (env + per request). Qwen3-ASR affords every feature.

    The configs deliberately omit `BACKEND`, so these also assert the engine
    default is vLLM. Needs the `vllm` extra (`uv sync --extra vllm --extra test`)
    and a CUDA GPU; vLLM startup is slow, so use a generous `--server-timeout` on
    the first (uncached) run."""

    # --- language ---

    @pytest.mark.parametrize(
        "uv_server", [_config("cuda", {"LANGUAGE": "fr"}, id="fr")], indirect=True
    )
    def test_integration_forced_language(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )

    @pytest.mark.parametrize(
        "uv_server", [_config("cuda", {"LANGUAGE": "*"}, id="auto")], indirect=True
    )
    def test_language_auto_detect(self, uv_server, test_audio_bonjour):
        """LANGUAGE=* -> automatic detection; French audio still transcribes."""
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )

    @pytest.mark.parametrize(
        "uv_server", [_config("cuda", {"LANGUAGE": "en"}, id="lang-override")], indirect=True
    )
    def test_language_per_request_override(self, uv_server, test_audio_bonjour):
        """Config says en; the per-request language=fr overrides it."""
        result = transcribe_http(
            uv_server["url"], str(test_audio_bonjour), language="fr"
        )
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )

    # --- hotwords ---

    @pytest.mark.parametrize(
        "uv_server", [_config("cuda", {"LANGUAGE": "fr"}, id="no-hotwords")], indirect=True
    )
    def test_hotwords_baseline(self, uv_server, test_audio_hotel):
        """Control: without a hotword, the biased word is NOT produced."""
        result = transcribe_http(uv_server["url"], str(test_audio_hotel))
        assert get_expected_regex(str(test_audio_hotel), "fr").search(result), (
            f"Unexpected transcription: {result!r}"
        )
        assert BIASED_WORD not in result.lower(), (
            f"Baseline unexpectedly already contains {BIASED_WORD!r}: {result!r}"
        )

    @pytest.mark.parametrize(
        "uv_server",
        [_config("cuda", {"LANGUAGE": "fr", "HOTWORDS": HOTWORDS}, id="hotwords-env")],
        indirect=True,
    )
    def test_hotwords_env(self, uv_server, test_audio_hotel):
        """The HOTWORDS env default biases the transcription toward 'double'."""
        result = transcribe_http(uv_server["url"], str(test_audio_hotel))
        assert BIASED_WORD in result.lower(), (
            f"Expected hotword-biased {BIASED_WORD!r} in transcription: {result!r}"
        )

    @pytest.mark.parametrize(
        "uv_server", [_config("cuda", {"LANGUAGE": "fr"}, id="hotwords-request")], indirect=True
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
# Second model family (Cohere, seq2seq): generalisation + capability guards
# ---------------------------------------------------------------------------


@pytest.mark.transformers
@pytest.mark.vllm
@pytest.mark.uv
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not _vllm_installed(), reason="vllm extra not installed")
class TestTransformersCohere:
    """Cohere (`cohere_asr`, a seq2seq ASR): proves the engine generalises beyond
    Qwen3-ASR. Integration is checked on BOTH backends (`vllm` and `transformers`).
    The capability guards are backend-independent (enforced in `decode` from the
    detected capabilities): Cohere requires a language and has no hotwords. CUDA
    GPU; the ~2B model downloads on first use (and pulls `librosa`)."""

    @pytest.mark.parametrize(
        "uv_server",
        [
            _cohere_config("fr", "cohere-vllm", backend="vllm"),
            _cohere_config("fr", "cohere-transformers", backend="transformers"),
        ],
        indirect=True,
    )
    def test_integration(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )

    @pytest.mark.parametrize(
        "uv_server", [_cohere_config("fr", "cohere-hotwords")], indirect=True
    )
    def test_hotwords_rejected(self, uv_server, test_audio_bonjour):
        """Cohere has no hotwords -> a request with hotwords is rejected."""
        with pytest.raises(RuntimeError):
            transcribe_http(uv_server["url"], str(test_audio_bonjour), hotwords="test")

    @pytest.mark.parametrize(
        "uv_server", [_cohere_config("fr", "cohere-nolang")], indirect=True
    )
    def test_no_language_rejected(self, uv_server, test_audio_bonjour):
        """Cohere needs a language -> forcing auto-detect (language='*') is rejected."""
        with pytest.raises(RuntimeError):
            transcribe_http(uv_server["url"], str(test_audio_bonjour), language="*")


# ---------------------------------------------------------------------------
# Dedicated `transformers` backend suite (one model, feature matrix + Docker)
# ---------------------------------------------------------------------------


@pytest.mark.transformers
class TestTransformersBackend:
    """The in-process `transformers` backend (CPU-capable, the portable fallback),
    on one model (Qwen3-ASR). Feature matrix: with/without a forced language, and
    with/without hotwords. Plus one Docker container test validating the built
    image on this backend. The UV tests follow --device (default cpu); the Docker
    test runs on CUDA. All pin `BACKEND=transformers` (the engine default is
    `vllm`)."""

    @pytest.mark.uv
    @pytest.mark.parametrize(
        "uv_server",
        [_config(None, {"BACKEND": "transformers", "LANGUAGE": "fr"}, id="tf-fr")],
        indirect=True,
    )
    def test_language_forced(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )

    @pytest.mark.uv
    @pytest.mark.parametrize(
        "uv_server",
        [_config(None, {"BACKEND": "transformers", "LANGUAGE": "*"}, id="tf-auto")],
        indirect=True,
    )
    def test_language_auto_detect(self, uv_server, test_audio_bonjour):
        """LANGUAGE=* -> automatic detection on the transformers backend."""
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )

    @pytest.mark.uv
    @pytest.mark.parametrize(
        "uv_server",
        [_config(None, {"BACKEND": "transformers", "LANGUAGE": "fr"}, id="tf-no-hotwords")],
        indirect=True,
    )
    def test_hotwords_baseline(self, uv_server, test_audio_hotel):
        """Control: without a hotword, the biased word is NOT produced."""
        result = transcribe_http(uv_server["url"], str(test_audio_hotel))
        assert get_expected_regex(str(test_audio_hotel), "fr").search(result), (
            f"Unexpected transcription: {result!r}"
        )
        assert BIASED_WORD not in result.lower(), (
            f"Baseline unexpectedly already contains {BIASED_WORD!r}: {result!r}"
        )

    @pytest.mark.uv
    @pytest.mark.parametrize(
        "uv_server",
        [_config(None, {"BACKEND": "transformers", "LANGUAGE": "fr"}, id="tf-hotwords")],
        indirect=True,
    )
    def test_hotwords(self, uv_server, test_audio_hotel):
        """The per-request hotword biases the transcription toward 'double'."""
        result = transcribe_http(
            uv_server["url"], str(test_audio_hotel), hotwords=HOTWORDS
        )
        assert BIASED_WORD in result.lower(), (
            f"Expected hotword-biased {BIASED_WORD!r} in transcription: {result!r}"
        )

    @pytest.mark.docker
    @pytest.mark.gpu
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
                    "env_overrides": {
                        "MODEL": TEST_MODEL,
                        "LANGUAGE": "fr",
                        "BACKEND": "transformers",
                    },
                },
                id="tf-docker",
            ),
        ],
        indirect=True,
    )
    def test_docker_http(self, docker_server, test_audio_bonjour):
        """The transformers backend inside the built Docker image (CUDA)."""
        result = transcribe_http(docker_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


# ---------------------------------------------------------------------------
# Streaming (WebSocket) - unsupported
# ---------------------------------------------------------------------------


@pytest.mark.transformers
@pytest.mark.uv
class TestTransformersStreamingUnsupported:
    """This engine does NOT support WebSocket streaming. There is therefore no
    positive streaming test; instead we assert the entry point fails loudly --
    wssDecode raises NotImplementedError -- so the unsupported mode can never
    silently appear to work."""

    def test_wssdecode_raises_not_implemented(self, monkeypatch):
        # Keep the engine import cheap and offline: CPU + http means the
        # module-level warmup (which would load/download models) is skipped.
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("SERVICE_MODE", "http")
        monkeypatch.setenv("BACKEND", "transformers")
        pytest.importorskip("transformers")

        import asyncio

        from linto_stt.engines.transformers.stt.processing.streaming import wssDecode

        with pytest.raises(NotImplementedError):
            asyncio.run(wssDecode(None, None))


# ---------------------------------------------------------------------------
# Long-audio chunking (unit test, no model/GPU)
# ---------------------------------------------------------------------------


@pytest.mark.transformers
@pytest.mark.uv
class TestTransformersLongFileChunking:
    """Unit-test the long-file chunk/merge logic without loading a model: the
    per-segment transcription is mocked, and we check chunk count, text
    concatenation, per-chunk timestamp offsets, and that the merged words still
    reconstruct the merged text."""

    def test_chunks_merge_with_offsets(self, monkeypatch):
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("SERVICE_MODE", "http")
        monkeypatch.setenv("BACKEND", "transformers")
        pytest.importorskip("transformers")

        import numpy as np

        from linto_stt.engines.transformers.stt import LONG_FILE_CHUNK_LEN
        from linto_stt.engines.transformers.stt.processing import decoding
        from linto_stt.engines.transformers.stt.processing.utils import SAMPLE_RATE

        calls = []

        def fake_segment(model, audio, with_word_timestamps, language, prompt):
            idx = len(calls)
            calls.append(len(audio))
            text = f"word{idx}."
            if with_word_timestamps:
                words = [{"word": text, "start": 0.1, "end": 0.5}]
            else:
                words = text.split()
            return text, words, "fr"

        monkeypatch.setattr(decoding, "_transcribe_segment", fake_segment)

        # 2.5 chunks worth of audio -> ceil = 3 chunks.
        n = int(2.5 * LONG_FILE_CHUNK_LEN * SAMPLE_RATE)
        audio = np.zeros(n, dtype=np.float32)

        text, words, language = decoding._transcribe_long(
            model=None, audio=audio, with_word_timestamps=True, language="fr", prompt=None
        )

        assert len(calls) == 3, f"expected 3 chunks, got {len(calls)}"
        assert text == "word0. word1. word2."
        assert language == "fr"
        # Each chunk's timestamps are offset by that chunk's start time.
        assert words[0]["start"] == pytest.approx(0.1)
        assert words[1]["start"] == pytest.approx(0.1 + LONG_FILE_CHUNK_LEN)
        assert words[2]["start"] == pytest.approx(0.1 + 2 * LONG_FILE_CHUNK_LEN)
        # Merged words still reconstruct merged text (the harness invariant).
        text_from_words = " ".join(w["word"] for w in words)
        assert text.replace(" ", "") == text_from_words.replace(" ", "")


@pytest.mark.transformers
@pytest.mark.uv
@pytest.mark.slow
class TestTransformersAlignerLanguages:
    """The forced aligner is tuned for 11 languages but must still give word
    timestamps for the others (the language is only a word-splitting hint).
    Runs the real aligner on CPU on a 2s clip."""

    def test_unsupported_language_still_aligned(self, monkeypatch, caplog):
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("SERVICE_MODE", "http")
        monkeypatch.setenv("BACKEND", "transformers")
        pytest.importorskip("transformers")

        import logging
        from types import SimpleNamespace

        from linto_stt.engines.transformers.stt import ALIGNER_MODEL
        from linto_stt.engines.transformers.stt.processing import decoding
        from linto_stt.engines.transformers.stt.processing.load_model import load_aligner_model
        from linto_stt.engines.transformers.stt.processing.utils import load_audiofile

        processor, aligner = load_aligner_model(ALIGNER_MODEL, device="cpu", dtype="float32")
        model = SimpleNamespace(aligner_processor=processor, aligner_model=aligner)
        audio = load_audiofile("tests/bonjour.wav")
        text = "Bonjour."

        with caplog.at_level(logging.WARNING):
            words_nl = decoding._align_words(model, audio, text, "nl")  # not in the 11
        words_fr = decoding._align_words(model, audio, text, "fr")

        assert "without a language hint" in caplog.text
        assert [w["word"] for w in words_nl] == ["Bonjour."]
        assert 0 <= words_nl[0]["start"] < words_nl[0]["end"] <= len(audio) / 16000 + 0.1
        # Same model, same word splitting for a whitespace language: identical result.
        assert words_nl == words_fr


# ---------------------------------------------------------------------------
# Docker - Celery task mode
# ---------------------------------------------------------------------------


@pytest.mark.transformers
@pytest.mark.docker
@pytest.mark.gpu
class TestTransformersDockerCelery:
    """transformers engine via Celery task mode (Docker container + Redis broker),
    on the default `vllm` backend. Exercises the `-m task` serving path end to
    end: a task is sent to the worker, which loads the audio via the engine's
    utils and runs decode (with metadata, so the aligner runs too). The image
    ships the `vllm` extra, so the default backend runs in-container (GPU)."""

    @pytest.mark.parametrize(
        "docker_server",
        [
            pytest.param(
                {
                    "engine": "transformers",
                    "mode": "task",
                    "port": 0,
                    "use_gpu": True,
                    "env_overrides": {
                        "MODEL": TEST_MODEL,
                        "LANGUAGE": "fr",
                        "DEVICE": "cuda",
                        # Skip torch.compile + CUDA-graph capture so the in-container
                        # vLLM server starts well within the default --server-timeout
                        # (a cold container has no warm compile cache). ASR output is
                        # unaffected; only startup is faster.
                        "VLLM_ENFORCE_EAGER": "1",
                        "SERVICES_BROKER": "redis://172.17.0.1:6379",
                    },
                },
                id="fr-task-vllm",
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
