import pytest

from helpers.env_setup import get_expected_regex
from helpers.transcription_client import transcribe_http


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------

# The `hf` engine serves HF transformers-native ASR checkpoints. Tests
# use the smallest Qwen3-ASR family member (fastest to download and run); the
# 1.7B default (Qwen/Qwen3-ASR-1.7B-hf) shares the same interface.
TEST_MODEL = "Qwen/Qwen3-ASR-0.6B-hf"

# A second model family (Cohere, a Whisper-like seq2seq ASR): a different
# processor API, no hotwords, a required language, its own long-audio chunking.
COHERE_MODEL = "CohereLabs/cohere-transcribe-03-2026"
# Whisper (served by the hf engine on both backends); only its processor is used here.
WHISPER_MODEL = "openai/whisper-large-v3-turbo"

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
        {"engine": "hf", "mode": mode, "port": 0, "env_overrides": env},
        id=id,
    )


def _cohere_config(language, id, backend="vllm"):
    env = {"MODEL": COHERE_MODEL, "LANGUAGE": language, "DEVICE": "cuda", "BACKEND": backend}
    return pytest.param(
        {"engine": "hf", "mode": "http", "port": 0, "env_overrides": env},
        id=id,
    )


def _vllm_installed():
    import importlib.util

    return importlib.util.find_spec("vllm") is not None


# ---------------------------------------------------------------------------
# Capability detection (CPU unit test, no model weights)
# ---------------------------------------------------------------------------


@pytest.mark.hf
@pytest.mark.uv
class TestHfCapabilities:
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

        from linto_stt.engines.hf.stt.processing.backends import (
            detect_capabilities,
        )

        qwen = detect_capabilities(AutoProcessor.from_pretrained(TEST_MODEL))
        assert qwen == {"hotwords": True, "language_spec": True, "language_auto": True}

        cohere = detect_capabilities(AutoProcessor.from_pretrained(COHERE_MODEL))
        assert cohere == {"hotwords": False, "language_spec": True, "language_auto": False}
        # Whisper takes the language as a decoder prompt (not a processor
        # argument): it can be forced, and auto-detected otherwise.
        whisper = detect_capabilities(AutoProcessor.from_pretrained(WHISPER_MODEL))
        assert whisper == {"hotwords": False, "language_spec": True, "language_auto": True}


# ---------------------------------------------------------------------------
# Feature coverage on the DEFAULT backend (BACKEND=vllm)
# ---------------------------------------------------------------------------


@pytest.mark.hf
@pytest.mark.vllm
@pytest.mark.uv
@pytest.mark.gpu
@pytest.mark.skipif(not _vllm_installed(), reason="vllm extra not installed")
class TestHfVllm:
    """All features on the default backend (`vllm`, a managed `vllm serve`
    subprocess): automatic language detection, forced language (config + per
    request), and hotwords (env + per request). Qwen3-ASR affords every feature.

    The configs deliberately omit `BACKEND`, so these also assert the engine
    default is vLLM. Needs the `hf-vllm` extra (`uv sync --extra hf-vllm --extra test`)
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


@pytest.mark.hf
@pytest.mark.vllm
@pytest.mark.uv
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not _vllm_installed(), reason="vllm extra not installed")
class TestHfCohere:
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


def _whisper_config(language, id, backend):
    env = {"MODEL": WHISPER_MODEL, "LANGUAGE": language, "DEVICE": "cuda", "BACKEND": backend}
    return pytest.param(
        {"engine": "hf", "mode": "http", "port": 0, "env_overrides": env},
        id=id,
    )


@pytest.mark.hf
@pytest.mark.vllm
@pytest.mark.uv
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not _vllm_installed(), reason="vllm extra not installed")
class TestHfWhisper:
    """Whisper on the hf engine, on BOTH backends: the language is a decoder
    prompt (not a processor argument), so it must be accepted when forced and
    detected otherwise. CUDA GPU; the ~0.8B turbo model downloads on first use."""

    @pytest.mark.parametrize(
        "uv_server",
        [
            _whisper_config("fr", "whisper-vllm-fr", backend="vllm"),
            _whisper_config("fr", "whisper-transformers-fr", backend="transformers"),
            _whisper_config("*", "whisper-transformers-auto", backend="transformers"),
        ],
        indirect=True,
    )
    def test_integration(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


# ---------------------------------------------------------------------------
# Dedicated `transformers` backend suite (one model, feature matrix + Docker)
# ---------------------------------------------------------------------------


@pytest.mark.hf
class TestHfBackend:
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
                    "engine": "hf",
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


@pytest.mark.hf
@pytest.mark.uv
class TestHfStreamingUnsupported:
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

        from linto_stt.engines.hf.stt.processing.streaming import wssDecode

        with pytest.raises(NotImplementedError):
            asyncio.run(wssDecode(None, None))


# ---------------------------------------------------------------------------
# Long-audio chunking (unit test, no model/GPU)
# ---------------------------------------------------------------------------


@pytest.mark.hf
@pytest.mark.uv
class TestHfLongFileChunking:
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

        from linto_stt.engines.hf.stt import LONG_FILE_CHUNK_LEN
        from linto_stt.engines.hf.stt.processing import decoding
        from linto_stt.engines.hf.stt.processing.utils import SAMPLE_RATE

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


@pytest.mark.hf
@pytest.mark.uv
@pytest.mark.slow
class TestHfAligners:
    """Run the real forced aligners on CPU on the 1.2s `bonjour.wav` clip."""

    @pytest.fixture
    def engine(self, monkeypatch):
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("SERVICE_MODE", "http")
        monkeypatch.setenv("BACKEND", "transformers")
        pytest.importorskip("transformers")
        from linto_stt.engines.hf.stt.processing import aligners, utils

        return aligners, utils.load_audiofile("tests/bonjour.wav")

    @staticmethod
    def _check_bonjour(words, audio):
        assert [w["word"] for w in words] == ["Bonjour."]
        assert 0 <= words[0]["start"] < words[0]["end"] <= len(audio) / 16000 + 0.1

    def test_qwen_unsupported_language_still_aligned(self, engine, caplog):
        """The Qwen aligner is tuned for 11 languages but must still give word
        timestamps for the others (the language is only a word-splitting hint)."""
        import logging

        from linto_stt.engines.hf.stt import ALIGNER_MODEL

        aligners, audio = engine
        aligner = aligners.load_aligner(ALIGNER_MODEL, device="cpu", dtype="float32")
        assert aligner.covers("fr") and not aligner.covers("nl")

        with caplog.at_level(logging.WARNING):
            words_nl = aligner.align(audio, "Bonjour.", "nl")  # not in the 11
        words_fr = aligner.align(audio, "Bonjour.", "fr")

        assert "without a language hint" in caplog.text
        self._check_bonjour(words_nl, audio)
        # Same model, same word splitting for a whitespace language: identical result.
        assert words_nl == words_fr

    def test_mms_aligner(self, engine):
        """The MMS aligner (ALIGNER_MODEL=mms) covers any language through
        romanisation, and keeps the one-entry-per-token contract."""
        pytest.importorskip("uroman")
        aligners, audio = engine
        aligner = aligners.load_aligner("mms", device="cpu")
        assert aligner.covers("nl") and aligner.covers("ar")
        assert aligner._normalize("Привет,") == "privet"
        assert aligner._normalize("ça") == "sa"
        assert aligner._normalize("...") == ""
        # "-" is the CTC blank in the MMS dictionary: hyphens must be dropped.
        assert aligner._normalize("pouvez-vous") == "pouvezvous"
        aligner.align(audio, "Bonjour, pouvez-vous ?", "fr")

        self._check_bonjour(aligner.align(audio, "Bonjour.", "fr"), audio)
        # Punctuation-only tokens get a zero-width stamp at the previous end.
        words = aligner.align(audio, "Bonjour !", "fr")
        assert [w["word"] for w in words] == ["Bonjour", "!"]
        assert words[1]["start"] == words[1]["end"] == words[0]["end"]

    def test_mms_long_audio_consistency(self, engine):
        """Quality check of the MMS aligner beyond the Qwen aligner's 5-minute
        limit: a 5+ minute file built from the two test clips (with silences) is
        aligned in one pass, and every word must land where the same clip
        aligned alone puts it (no drift, no window-boundary damage)."""
        pytest.importorskip("uroman")
        import logging

        import numpy as np
        import torch

        aligners, _ = engine
        from linto_stt.engines.hf.stt.processing.utils import load_audiofile

        device = "cuda" if torch.cuda.is_available() else "cpu"
        aligner = aligners.load_aligner("mms", device=device)
        clips = [
            (load_audiofile("tests/bonjour.wav"), "Bonjour."),
            (
                load_audiofile("tests/Hotel20sec.wav"),
                "Bonjour mona. Bonjour monsieur. En quoi pouvez-vous aider aujourd'hui ? "
                "Euh je voudrais avoir une chambre pour deux personnes. Deux personnes "
                "chambre d'eau d'accord. Euh deux lits. Deux lits d'accord. Et puis ça "
                "serait pour deux nuits. Deux nuits.",
            ),
        ]
        silence = np.zeros(16000 // 2, dtype=np.float32)

        # Reference: each clip aligned alone, offset by its position in the file.
        pieces, texts, reference, offset = [], [], [], 0.0
        while offset < 310:
            for audio, text in clips:
                for w in aligner.align(audio, text, "fr"):
                    reference.append({**w, "start": w["start"] + offset, "end": w["end"] + offset})
                pieces += [audio, silence]
                texts.append(text)
                offset += (len(audio) + len(silence)) / 16000
        long_audio = np.concatenate(pieces)
        duration = len(long_audio) / 16000
        assert duration > 300 and not aligners.QwenForcedAligner.max_duration >= duration

        words = aligner.align(long_audio, " ".join(texts), "fr")
        assert [w["word"] for w in words] == [w["word"] for w in reference]
        content = [(w, r) for w, r in zip(words, reference) if aligner._normalize(w["word"])]
        assert len(content) > 500
        deltas = sorted(
            max(abs(w["start"] - r["start"]), abs(w["end"] - r["end"])) for w, r in content
        )
        logging.getLogger(__name__).info(
            f"MMS on {duration:.0f}s ({len(content)} words, {device}): deviation from the "
            f"per-clip alignment median={deltas[len(deltas) // 2] * 1000:.0f}ms "
            f"p95={deltas[int(0.95 * len(deltas))] * 1000:.0f}ms max={deltas[-1] * 1000:.0f}ms"
        )
        assert deltas[len(deltas) // 2] <= 0.05, "median deviation above 50 ms"
        assert deltas[int(0.95 * len(deltas))] <= 0.1, "more than 5% of words off by over 100 ms"
        # Word starts are stable; only the end of a word followed by a long pause
        # is ambiguous for a CTC aligner (it may stretch into the silence), so
        # the strict bound is on starts.
        start_deltas = [abs(w["start"] - r["start"]) for w, r in content]
        assert max(start_deltas) <= 0.5, f"a word start drifted by {max(start_deltas):.2f}s"
        starts = [w["start"] for w, _ in content]
        assert starts == sorted(starts) and content[-1][0]["end"] <= duration

    def test_fallback_routing(self, engine, monkeypatch):
        """ALIGNER_FALLBACK_MODEL takes over when the primary aligner does not
        cover the request: language outside its set, or audio over its limit.
        Uses fake aligners, so no model is loaded."""
        from types import SimpleNamespace

        from linto_stt.engines.hf.stt.processing import LazyLoadedModel

        def fake(name, languages, max_duration):
            def covers(language, duration=None):
                if duration is not None and max_duration and duration > max_duration:
                    return False
                return language is None or language in languages

            return SimpleNamespace(name=name, covers=covers, max_duration=max_duration)

        fakes = {
            "primary": fake("primary", {"fr", "en"}, 300.0),
            "fallback": fake("fallback", None, None),
        }
        fakes["fallback"].covers = lambda language, duration=None: True
        loaded = []

        def load(self, name):
            loaded.append(name)
            return fakes[name]

        monkeypatch.setattr(LazyLoadedModel, "_load_aligner", load)

        def model(fallback):
            return LazyLoadedModel(
                "unused", device="cpu", dtype="float32", attn_implementation=None,
                num_threads=None, aligner_model_name="primary",
                fallback_aligner_model_name=fallback,
            )

        m = model("fallback")
        assert m.aligner_for("fr", 10).name == "primary"
        assert m.aligner_for(None, 10).name == "primary"  # auto-detect, unknown language
        assert loaded == ["primary"]  # the fallback is not loaded until needed
        assert m.aligner_for("nl", 10).name == "fallback"  # language not covered
        assert m.aligner_for("fr", 400).name == "fallback"  # over the 5-minute limit
        assert loaded == ["primary", "fallback"]
        # Without a fallback the primary is always used.
        m = model(None)
        assert m.aligner_for("nl", 400).name == "primary"


# ---------------------------------------------------------------------------
# Docker - Celery task mode
# ---------------------------------------------------------------------------


@pytest.mark.hf
@pytest.mark.docker
@pytest.mark.gpu
class TestHfDockerCelery:
    """hf engine via Celery task mode (Docker container + Redis broker),
    on the default `vllm` backend. Exercises the `-m task` serving path end to
    end: a task is sent to the worker, which loads the audio via the engine's
    utils and runs decode (with metadata, so the aligner runs too). The image
    ships the `hf-vllm` extra, so the default backend runs in-container (GPU)."""

    @pytest.mark.parametrize(
        "docker_server",
        [
            pytest.param(
                {
                    "engine": "hf",
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
