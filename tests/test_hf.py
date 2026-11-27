import re
from pathlib import Path

import pytest

from conftest import LONG_AUDIO_REPEATS
from helpers.env_setup import build_env_dict, get_expected_regex
from helpers.server_runner import UVServerRunner
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
# Granite Speech 5.0: served through vLLM's chat endpoint, with the prompts of
# its model card (punctuation, keywords, a given language).
GRANITE_MODEL = "ibm-granite/granite-speech-5.0-2b"

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
    env = {"MODEL": COHERE_MODEL, "LANGUAGE": language, "DEVICE": "cuda", "BACKEND": backend,
           **(EAGER if backend == "vllm" else {})}
    return pytest.param(
        {"engine": "hf", "mode": "http", "port": 0, "env_overrides": env},
        id=id,
    )


# vLLM startup is dominated by torch.compile and CUDA-graph capture: most vLLM
# tests skip them (VLLM_ENFORCE_EAGER, same transcription), one server per
# configuration keeps the production setting (compiled).
EAGER = {"VLLM_ENFORCE_EAGER": "1"}


def _class_server(project_root, server_timeout, env):
    """A UV server shared by the tests of a class (a vLLM server takes minutes to
    start, a request seconds). Generator body for a class-scoped fixture."""
    runner = UVServerRunner(
        project_root=project_root, engine="hf", mode="http", port=0,
        env_dict=build_env_dict(project_root, "hf", env), timeout=server_timeout,
    )
    try:
        url = runner.start()
        yield {"url": url, "runner": runner, "engine": "hf", "mode": "http", "timeout": server_timeout}
    finally:
        runner.stop()


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

    def test_long_audio_handling_per_model(self, monkeypatch):
        """On the vllm backend the engine sends the whole file, except for the
        models vLLM does not cut into windows (Voxtral), which it chunks itself."""
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("SERVICE_MODE", "http")
        monkeypatch.setenv("BACKEND", "transformers")
        pytest.importorskip("transformers")

        from linto_stt.engines.hf.stt.processing.backends import (
            VllmServeBackend,
            _model_type,
        )

        not_windowed = VllmServeBackend.MODEL_TYPES_NOT_WINDOWED_BY_VLLM
        assert _model_type(VOXTRAL_MODEL) in not_windowed
        for model in (TEST_MODEL, COHERE_MODEL, WHISPER_MODEL):
            assert _model_type(model) not in not_windowed


# ---------------------------------------------------------------------------
# Feature coverage on the DEFAULT backend (BACKEND=vllm)
# ---------------------------------------------------------------------------


# One server per configuration, shared by the tests of its class (a vLLM server
# takes minutes to start, a request seconds). A class-scoped server stops at the
# end of its class; classes that start servers have one server configuration
# each, so two servers never run at once (one takes about 4 GB of GPU memory:
# vLLM, two aligners, ECAPA).
@pytest.fixture(scope="class")
def vllm_fr_server(project_root, server_timeout):
    """LANGUAGE=fr, compiled (the production setting)."""
    yield from _class_server(project_root, server_timeout,
                             {"MODEL": TEST_MODEL, "LANGUAGE": "fr", "DEVICE": "cuda"})


# All features on the default backend (`vllm`, a managed `vllm serve`
# subprocess): forced language (config), language identification (per-request
# `*`, which also proves the per-request override), an unsupported language, and
# hotwords (per request; the HOTWORDS env default is tested on the transformers
# backend, the code path being the same). Qwen3-ASR affords every feature. The
# config deliberately omits `BACKEND`, so these also assert the engine default
# is vLLM. Needs the `hf-vllm` extra (`uv sync --extra hf-vllm --extra test`) and
# a CUDA GPU; vLLM startup is slow, so use a generous `--server-timeout` on the
# first (uncached) run.
VLLM_MARKS = [
    pytest.mark.hf,
    pytest.mark.vllm,
    pytest.mark.uv,
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not _vllm_installed(), reason="vllm extra not installed"),
]


class TestHfVllm:
    """Qwen3-ASR on the vllm backend, one server (LANGUAGE=fr)."""

    pytestmark = VLLM_MARKS

    def test_integration_forced_language(self, vllm_fr_server, test_audio_bonjour):
        result = transcribe_http(vllm_fr_server["url"], str(test_audio_bonjour), full=True)
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result["text"]), (
            f"Unexpected transcription: {result}"
        )
        assert result["language"] == "fr"

    def test_language_identified_per_request(self, vllm_fr_server, test_audio_bonjour):
        """Per-request `*` overrides LANGUAGE=fr: the language is identified
        (LANGUAGE_IDENTIFICATION, on by default), passed to the model and
        returned."""
        result = transcribe_http(vllm_fr_server["url"], str(test_audio_bonjour),
                                 language="*", full=True)
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result["text"]), (
            f"Unexpected transcription: {result}"
        )
        assert result["language"] == "fr"

    def test_unsupported_language_rejected(self, vllm_fr_server, test_audio_bonjour):
        """A language Qwen3-ASR does not support (Swahili) -> HTTP 400 (vLLM
        itself would transcribe it, with a warning in its own log)."""
        with pytest.raises(RuntimeError, match=r"\(400\).*not supported by this model"):
            transcribe_http(vllm_fr_server["url"], str(test_audio_bonjour), language="sw")

    def test_hotwords_baseline(self, vllm_fr_server, test_audio_hotel):
        """Control: without a hotword, the biased word is NOT produced."""
        result = transcribe_http(vllm_fr_server["url"], str(test_audio_hotel))
        assert get_expected_regex(str(test_audio_hotel), "fr").search(result), (
            f"Unexpected transcription: {result!r}"
        )
        assert BIASED_WORD not in result.lower(), (
            f"Baseline unexpectedly already contains {BIASED_WORD!r}: {result!r}"
        )

    def test_hotwords_per_request(self, vllm_fr_server, test_audio_hotel):
        """The per-request `hotwords` param biases toward 'double' with no env set,
        proving it overrides the (empty) default."""
        result = transcribe_http(
            vllm_fr_server["url"], str(test_audio_hotel), hotwords=HOTWORDS
        )
        assert BIASED_WORD in result.lower(), (
            f"Expected hotword-biased {BIASED_WORD!r} in transcription: {result!r}"
        )


# ---------------------------------------------------------------------------
# Second model family (Cohere, seq2seq): generalisation + capability guards
# ---------------------------------------------------------------------------


# Cohere (`cohere_asr`, a seq2seq ASR): proves the engine generalises beyond
# Qwen3-ASR, on BOTH backends. Its capability guards (a required language, no
# hotwords) are enforced in `decode`, whatever the backend: see
# TestHfCapabilityGuards. CUDA GPU; the ~2B model downloads on first use (and
# pulls `librosa`).
@pytest.fixture(scope="class")
def cohere_vllm_server(project_root, server_timeout):
    """LANGUAGE=* (identified among Cohere's 14), eager."""
    yield from _class_server(project_root, server_timeout,
                             {"MODEL": COHERE_MODEL, "LANGUAGE": "*", "DEVICE": "cuda",
                              "BACKEND": "vllm", **EAGER})


class TestHfCohereVllm:
    """Cohere on the vllm backend, one server (LANGUAGE=*)."""

    pytestmark = VLLM_MARKS

    def test_language_identified(self, cohere_vllm_server, test_audio_bonjour):
        """No language: identified (LANGUAGE_IDENTIFICATION, on by default) among
        Cohere's 14, so this model works without a language."""
        result = transcribe_http(cohere_vllm_server["url"], str(test_audio_bonjour), full=True)
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result["text"]), (
            f"Unexpected transcription: {result}"
        )
        assert result["language"] == "fr"

    def test_language_forced_per_request(self, cohere_vllm_server, test_audio_bonjour):
        result = transcribe_http(cohere_vllm_server["url"], str(test_audio_bonjour),
                                 language="fr", full=True)
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result["text"]), (
            f"Unexpected transcription: {result}"
        )
        assert result["language"] == "fr"


class TestHfCohereTransformers:
    """Cohere on the transformers backend."""

    pytestmark = VLLM_MARKS

    @pytest.mark.parametrize(
        "uv_server", [_cohere_config("fr", "cohere-transformers", backend="transformers")],
        indirect=True,
    )
    def test_integration(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour), full=True)
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result["text"]), (
            f"Unexpected transcription: {result}"
        )
        assert result["language"] == "fr"


@pytest.mark.hf
@pytest.mark.uv
class TestHfCapabilityGuards:
    """`decode` rejects, before any decoding, a request the model cannot serve
    (capabilities detected from the processor, e.g. Cohere: a required language
    and no hotwords). Checked with a stub backend: no model, no server (the HTTP
    server turns the ValueError into an HTTP 400 for every engine)."""

    @pytest.fixture
    def decode(self, monkeypatch):
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("SERVICE_MODE", "http")
        monkeypatch.setenv("BACKEND", "transformers")
        pytest.importorskip("transformers")
        from linto_stt.engines.hf.stt.processing.decoding import decode

        return decode

    @staticmethod
    def _model(**caps):
        from types import SimpleNamespace

        asr = SimpleNamespace(supports_hotwords=caps.get("hotwords", False),
                              supports_language_spec=caps.get("language_spec", True),
                              supports_language_auto=caps.get("language_auto", False))
        return SimpleNamespace(asr=asr, language_identifier=None)

    def test_hotwords_rejected(self, decode):
        """No hotword mechanism (Cohere) -> a request with hotwords is rejected."""
        import numpy as np

        with pytest.raises(ValueError, match="hotwords"):
            decode(np.zeros(16000, dtype=np.float32), self._model(), False, language="fr", hotwords="test")

    def test_no_language_rejected(self, decode):
        """No language detection (Cohere) -> a request without a language is rejected."""
        import numpy as np

        with pytest.raises(ValueError, match="language detection"):
            decode(np.zeros(16000, dtype=np.float32), self._model(), False, language="*")

    def test_forced_language_rejected(self, decode):
        """A model that takes no forced language -> forcing one is rejected."""
        import numpy as np

        model = self._model(language_spec=False, language_auto=True)
        with pytest.raises(ValueError, match="forced language"):
            decode(np.zeros(16000, dtype=np.float32), model, False, language="fr")


class TestHfLanguages:
    """Language handling in `decode`: a given language must be one the model
    accepts (else HTTP 400), with a warning outside its documented languages;
    without a language, the language identifier (LANGUAGE_IDENTIFICATION) picks
    one among the documented (else accepted) languages, else the model detects
    it, else "*".
    Stub backend and identifier, except the last test (the real ECAPA on CPU)."""

    @pytest.fixture
    def decode(self, monkeypatch):
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("SERVICE_MODE", "http")
        monkeypatch.setenv("BACKEND", "transformers")
        pytest.importorskip("transformers")
        from linto_stt.engines.hf.stt.processing.decoding import decode

        return decode

    @staticmethod
    def _model(detected=None, language_auto=True, accepted=("fr", "en", "de"),
               recommended=None, identified=None):
        from types import SimpleNamespace

        calls = []

        def transcribe(audio, language=None, prompt=None):
            calls.append(language)
            return "bonjour", language or detected

        asr = SimpleNamespace(supports_hotwords=False, supports_language_spec=True,
                              supports_language_auto=language_auto, transcribe=transcribe,
                              accepted_languages=frozenset(accepted),
                              recommended_languages=frozenset(recommended) if recommended else None)
        identifier = None
        if identified:
            identifier = SimpleNamespace(
                identify=lambda audio, languages=None: (identified, identified),
                _code_in=lambda code, languages: code if code in languages else None)
        return SimpleNamespace(asr=asr, has_aligner=False, language_identifier=identifier), calls

    @staticmethod
    def _audio():
        import numpy as np

        return np.zeros(16000, dtype=np.float32)

    def test_unsupported_language_rejected(self, decode):
        model, calls = self._model()
        with pytest.raises(ValueError, match="not supported by this model"):
            decode(self._audio(), model, True, language="ru")
        assert calls == []  # rejected before any decoding

    def test_undocumented_language_warns(self, decode, caplog):
        import logging

        model, calls = self._model(recommended=("fr", "en"))
        with caplog.at_level(logging.WARNING):
            result = decode(self._audio(), model, True, language="de")
        assert "not among the model's documented languages" in caplog.text
        assert calls == ["de"] and result["language"] == "de"

    def test_given_language_is_not_identified(self, decode):
        model, calls = self._model(identified="en")
        result = decode(self._audio(), model, True, language="fr")
        assert calls == ["fr"] and result["language"] == "fr"

    def test_identified_language_is_forced_and_returned(self, decode):
        model, calls = self._model(identified="de")
        result = decode(self._audio(), model, True, language="*")
        assert calls == ["de"] and result["language"] == "de"

    def test_identification_among_documented_languages(self, decode, monkeypatch):
        """LANGUAGE_IDENTIFICATION_LANGUAGES: by default ("documented") the
        identifier chooses among the model card's languages when known, else
        among the supported ones; "supported": always among the supported ones."""
        from types import SimpleNamespace

        from linto_stt.engines.hf.stt.processing import languages

        def run(recommended):
            seen = []
            model, calls = self._model(accepted=("fr", "en", "de", "br"), recommended=recommended)
            model.language_identifier = SimpleNamespace(
                identify=lambda audio, languages=None: seen.append(languages) or ("fr", "fr"),
                _code_in=lambda code, languages: code if code in languages else None)
            assert decode(self._audio(), model, True, language="*")["language"] == "fr"
            return seen[0]

        assert run(("fr", "en")) == {"fr", "en"}
        assert run(None) == {"fr", "en", "de", "br"}
        monkeypatch.setattr(languages, "LANGUAGE_IDENTIFICATION_LANGUAGES", "supported")
        assert run(("fr", "en")) == {"fr", "en", "de", "br"}

    def test_identification_lets_a_model_without_detection_run(self, decode):
        """Cohere-like model (a language is required): works without one when
        the language is identified."""
        model, calls = self._model(language_auto=False, identified="fr")
        assert decode(self._audio(), model, True, language="*")["language"] == "fr"
        model, calls = self._model(language_auto=False)
        with pytest.raises(ValueError, match="no automatic language detection"):
            decode(self._audio(), model, True, language="*")

    def test_without_identification(self, decode):
        """The model detects the language: returned when the backend reports it
        (transformers), "*" when it does not (vLLM)."""
        model, calls = self._model(detected="French")
        result = decode(self._audio(), model, True, language="*")
        assert calls == [None] and result["language"] == "fr"
        model, calls = self._model(detected=None)
        assert decode(self._audio(), model, True, language="*")["language"] == "*"

    def test_card_languages(self, monkeypatch):
        """Model card `language:` metadata: Whisper's card lists Norwegian as a
        bare `no`, which YAML reads as False."""
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("SERVICE_MODE", "http")
        monkeypatch.setenv("BACKEND", "transformers")
        pytest.importorskip("transformers")
        from linto_stt.engines.hf.stt.processing.languages import parse_card_languages

        assert parse_card_languages(["en", False, "FR"]) == {"en", "no", "fr"}
        assert parse_card_languages("en") == {"en"}
        assert parse_card_languages(None) is None
        assert parse_card_languages([True, ""]) is None

    def test_ecapa_restricted_to_the_model_languages(self, monkeypatch):
        """The real ECAPA on CPU (86 MB download on first use): on a 1.2 s
        "Bonjour" its free choice is Breton; restricted to Qwen3-ASR's or
        Cohere's languages it is French. Mapped codes: Hebrew (iw -> he) and
        Tagalog (tl -> fil, Qwen3-ASR's code)."""
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("SERVICE_MODE", "http")
        monkeypatch.setenv("BACKEND", "transformers")
        pytest.importorskip("speechbrain")
        from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME

        from linto_stt.engines.hf.stt.processing.languages import LanguageIdentifier
        from linto_stt.engines.hf.stt.processing.utils import load_audiofile

        lid = LanguageIdentifier("cpu")
        assert "he" in lid.codes and "iw" not in lid.codes
        assert lid._code_in("tl", frozenset(LANGUAGE_CODE_TO_NAME)) == "fil"
        audio = load_audiofile(str(Path(__file__).resolve().parent / "bonjour.wav"))
        assert lid.identify(audio) == ("br", "br")
        assert lid.identify(audio, frozenset(LANGUAGE_CODE_TO_NAME)) == ("fr", "br")
        cohere = frozenset("ar de el en es fr it ja ko nl pl pt vi zh".split())
        assert lid.identify(audio, cohere) == ("fr", "br")


class TestHfConfidence:
    """The models served by the hf engine expose no usable confidence: the
    transcript and every word get a constant 1.0. Stub backend and aligner."""

    def test_constant_confidence(self, monkeypatch):
        from types import SimpleNamespace

        import numpy as np

        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("SERVICE_MODE", "http")
        monkeypatch.setenv("BACKEND", "transformers")
        pytest.importorskip("transformers")
        from linto_stt.engines.hf.stt.processing.decoding import decode

        asr = SimpleNamespace(supports_hotwords=False, supports_language_spec=True,
                              supports_language_auto=True,
                              transcribe=lambda audio, language=None, prompt=None: ("bonjour monsieur", "French"))
        aligner = SimpleNamespace(name="stub", align=lambda audio, text, language: [
            {"word": w, "start": float(i), "end": i + 0.5} for i, w in enumerate(text.split())])
        model = SimpleNamespace(asr=asr, has_aligner=True, language_identifier=None,
                                aligner_for=lambda language, duration=None: aligner)
        result = decode(np.zeros(16000, dtype=np.float32), model, True, language="fr")
        assert result["confidence-score"] == 1.0
        assert [w["conf"] for w in result["words"]] == [1.0, 1.0]


def _whisper_config(language, id, backend, marks=()):
    env = {"MODEL": WHISPER_MODEL, "LANGUAGE": language, "DEVICE": "cuda", "BACKEND": backend}
    return pytest.param(
        {"engine": "hf", "mode": "http", "port": 0, "env_overrides": env},
        id=id, marks=marks,
    )


# Whisper on the hf engine, on BOTH backends: the language is a decoder prompt
# (not a processor argument), so it must be accepted when forced and detected
# otherwise. CUDA GPU; the ~0.8B turbo model downloads on first use.
@pytest.fixture(scope="class")
def whisper_tf_server(project_root, server_timeout):
    """LANGUAGE=fr, transformers backend. Language identification off, so that
    a request without a language exercises Whisper's own detection."""
    yield from _class_server(project_root, server_timeout,
                             {"MODEL": WHISPER_MODEL, "LANGUAGE": "fr", "DEVICE": "cuda",
                              "BACKEND": "transformers", "LANGUAGE_IDENTIFICATION": "false"})


class TestHfWhisperTransformers:
    """Whisper on the transformers backend, one server."""

    pytestmark = VLLM_MARKS

    def test_language_forced(self, whisper_tf_server, test_audio_bonjour):
        result = transcribe_http(whisper_tf_server["url"], str(test_audio_bonjour), full=True)
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result["text"]), (
            f"Unexpected transcription: {result}"
        )
        assert result["language"] == "fr"

    def test_language_detected_by_whisper(self, whisper_tf_server, test_audio_bonjour):
        """Per-request `*`, identification off: Whisper detects the language,
        and the transformers backend reports it."""
        result = transcribe_http(whisper_tf_server["url"], str(test_audio_bonjour),
                                 language="*", full=True)
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result["text"]), (
            f"Unexpected transcription: {result}"
        )
        assert result["language"] == "fr"


class TestHfWhisperVllm:
    """Whisper on the vllm backend. Whisper is served by the whisper engine in
    production: this case (a server start) only runs with --long."""

    pytestmark = VLLM_MARKS

    @pytest.mark.parametrize(
        "uv_server",
        [_whisper_config("fr", "whisper-vllm-fr", backend="vllm", marks=pytest.mark.long)],
        indirect=True,
    )
    def test_integration(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


# ---------------------------------------------------------------------------
# Long audio on the vllm backend (run with --long)
# ---------------------------------------------------------------------------

VOXTRAL_MODEL = "mistralai/Voxtral-Mini-3B-2507"


@pytest.mark.hf
@pytest.mark.uv
class TestHfGranitePrompts:
    """Granite Speech prompts and the helpers around them (CPU, no model: the
    card text is given)."""

    @pytest.fixture
    def backends(self, monkeypatch):
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("SERVICE_MODE", "http")
        monkeypatch.setenv("BACKEND", "transformers")
        pytest.importorskip("transformers")
        from linto_stt.engines.hf.stt.processing import backends

        return backends

    def _prompts(self, backends, monkeypatch, card):
        monkeypatch.setattr(backends, "_card_text", lambda model_name: card)
        return backends.GranitePrompts("unused")

    def test_prompts_from_card(self, backends, monkeypatch):
        """What the card documents is what is used (4.1: no language prompt)."""
        card_41 = ("| ASR (with punctuation) | transcribe the speech with proper punctuation "
                   "and capitalization. |\n| ASR (with keyword biasing) | transcribe the speech "
                   "to text. Keywords: <kw1>, <kw2>, ... |")
        card_50 = card_41 + "\n| ASR (with language ID) | The speech is in <language>. ... |"
        g = self._prompts(backends, monkeypatch, card_50)
        assert (g.punctuation, g.keywords, g.language) == (True, True, True)
        assert g.text("fr", g.hotwords_prompt(["Broke", "Jean-Luc Machin"])) == (
            "The speech is in French. Transcribe it exactly; do not translate. "
            "transcribe the speech with proper punctuation and capitalization. "
            "Keywords: Broke, Jean-Luc Machin")
        g = self._prompts(backends, monkeypatch, card_41)
        assert (g.punctuation, g.keywords, g.language) == (True, True, False)
        assert g.text("fr", "Keywords: Broke") == (
            "transcribe the speech with proper punctuation and capitalization. Keywords: Broke")
        g = self._prompts(backends, monkeypatch, "")  # unreadable card: the raw prompt only
        assert g.text("fr", "Keywords: Broke") == "can you transcribe the speech into a written format?"

    def test_extra_weight_files(self, backends):
        """Weights next to model.safetensors that vLLM would load (no index)."""
        assert backends._extra_weight_files(
            ["model.safetensors", "out_llm.safetensors", "config.json", "sub/x.safetensors"]
        ) == ["out_llm.safetensors"]
        assert backends._extra_weight_files(["model.safetensors", "config.json"]) == []
        assert backends._extra_weight_files(  # sharded with an index: vLLM filters itself
            ["model.safetensors", "model-00001-of-00002.safetensors", "model.safetensors.index.json"]
        ) == []

    def test_hotwords_prompt_and_chunk_length(self, backends, monkeypatch):
        """The backend's keyword format and chunk length are used when it has
        them; else Qwen3-ASR's format and the LONG_FILE_* defaults."""
        from types import SimpleNamespace

        from linto_stt.engines.hf.stt.processing import decoding

        granite = SimpleNamespace(hotwords_prompt=backends.GranitePrompts.hotwords_prompt,
                                  long_file_chunk_len=120)
        qwen = SimpleNamespace()
        assert decoding._hotwords_prompt("Broke, Jean-Luc Machin", granite) == (
            "Keywords: Broke, Jean-Luc Machin")
        assert decoding._hotwords_prompt("chambre double", qwen) == "Vocabulary: chambre double."
        assert decoding._hotwords_prompt("", granite) is None
        monkeypatch.setattr(decoding, "LONG_FILE_SETTINGS_FROM_ENV", False)
        assert decoding._long_file_settings(granite) == (120, 120)
        assert decoding._long_file_settings(qwen) == (
            decoding.LONG_FILE_THRESHOLD, decoding.LONG_FILE_CHUNK_LEN)
        monkeypatch.setattr(decoding, "LONG_FILE_SETTINGS_FROM_ENV", True)
        assert decoding._long_file_settings(granite) == (
            decoding.LONG_FILE_THRESHOLD, decoding.LONG_FILE_CHUNK_LEN)


@pytest.fixture(scope="class")
def granite_vllm_server(project_root, server_timeout):
    """Granite Speech 5.0, no LANGUAGE (identified, not given to the model), eager."""
    yield from _class_server(project_root, server_timeout,
                             {"MODEL": GRANITE_MODEL, "DEVICE": "cuda", "BACKEND": "vllm", **EAGER})


class TestHfGraniteVllm:
    """Granite Speech 5.0 on the vllm backend, one server: punctuation and
    capitalization, a given language, hotwords (keyword biasing)."""

    pytestmark = VLLM_MARKS

    def test_integration(self, granite_vllm_server, test_audio_hotel):
        """The language is identified and returned (not given to the model), and
        the punctuation prompt gives a capitalized and punctuated transcript."""
        result = transcribe_http(granite_vllm_server["url"], str(test_audio_hotel), full=True)
        assert "Bonjour, madame." in result["text"], f"Unexpected transcription: {result}"
        assert result["language"] == "fr"

    def test_language_given(self, granite_vllm_server, test_audio_bonjour):
        result = transcribe_http(granite_vllm_server["url"], str(test_audio_bonjour),
                                 language="fr", full=True)
        assert "Bonjour" in result["text"], f"Unexpected transcription: {result}"
        assert result["language"] == "fr"

    def test_hotwords(self, granite_vllm_server, test_audio_hotel):
        """Without hotwords, the spoken "chambre double" is heard as "chambre
        douze"; with them, "double" is written."""
        result = transcribe_http(granite_vllm_server["url"], str(test_audio_hotel))
        assert get_expected_regex(str(test_audio_hotel), "fr").search(result), (
            f"Unexpected transcription: {result!r}"
        )
        assert BIASED_WORD not in result.lower(), (
            f"Baseline unexpectedly already contains {BIASED_WORD!r}: {result!r}"
        )
        result = transcribe_http(granite_vllm_server["url"], str(test_audio_hotel),
                                 hotwords=HOTWORDS)
        assert BIASED_WORD in result.lower(), (
            f"Expected hotword-biased {BIASED_WORD!r} in transcription: {result!r}"
        )


def _long_config(model, id):
    env = {"MODEL": model, "LANGUAGE": "fr", "DEVICE": "cuda", "BACKEND": "vllm"}
    return pytest.param(
        {"engine": "hf", "mode": "http", "port": 0, "env_overrides": env}, id=id
    )


@pytest.mark.hf
@pytest.mark.vllm
@pytest.mark.uv
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.long
@pytest.mark.skipif(not _vllm_installed(), reason="vllm extra not installed")
class TestHfLongAudio:
    """A file of 14 minutes on the vllm backend, with the default memory
    settings. Two cases: a model that vLLM cuts into 30 s windows (Qwen3-ASR:
    the engine sends the whole file) and models that it does not (Voxtral, and
    Granite through the chat endpoint: the engine cuts the file itself, else the
    request exceeds the context)."""

    @pytest.mark.parametrize(
        "uv_server",
        [_long_config(TEST_MODEL, "long-qwen"), _long_config(VOXTRAL_MODEL, "long-voxtral"),
         _long_config(GRANITE_MODEL, "long-granite")],
        indirect=True,
    )
    def test_long_file(self, uv_server, long_audio):
        result = transcribe_http(uv_server["url"], str(long_audio), timeout=900)
        # The whole file must be transcribed, not a truncated part: the dialogue
        # says "chambre" at least once per repetition (allow a few misses).
        count = len(re.findall(r"chambre", result, flags=re.IGNORECASE))
        assert count >= int(0.8 * LONG_AUDIO_REPEATS), (
            f"'chambre' found {count} times for {LONG_AUDIO_REPEATS} repetitions: "
            f"{result[:300]}..."
        )


# ---------------------------------------------------------------------------
# Dedicated `transformers` backend suite (one model, feature matrix + Docker)
# ---------------------------------------------------------------------------


# The in-process `transformers` backend (CPU-capable, the portable fallback), on
# one model (Qwen3-ASR): with/without a forced language, with/without hotwords,
# and the built Docker image. The UV servers follow --device (default cpu); the
# Docker test runs on CUDA. All pin `BACKEND=transformers` (the engine default
# is `vllm`).
def _tf_server(request, project_root, server_timeout, env):
    env = {"MODEL": TEST_MODEL, "BACKEND": "transformers",
           "DEVICE": request.config.getoption("--device"), **env}
    yield from _class_server(project_root, server_timeout, env)


@pytest.fixture(scope="class")
def tf_fr_server(request, project_root, server_timeout):
    """LANGUAGE=fr, no hotwords."""
    yield from _tf_server(request, project_root, server_timeout, {"LANGUAGE": "fr"})


@pytest.fixture(scope="class")
def tf_hotwords_env_server(request, project_root, server_timeout):
    """LANGUAGE=fr, HOTWORDS set in the environment."""
    yield from _tf_server(request, project_root, server_timeout,
                          {"LANGUAGE": "fr", "HOTWORDS": HOTWORDS})


@pytest.mark.hf
@pytest.mark.uv
class TestHfBackend:
    """Qwen3-ASR on the transformers backend, one server (LANGUAGE=fr)."""

    def test_language_forced(self, tf_fr_server, test_audio_bonjour):
        result = transcribe_http(tf_fr_server["url"], str(test_audio_bonjour), full=True)
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result["text"]), (
            f"Unexpected transcription: {result}"
        )
        assert result["language"] == "fr"

    def test_language_identified_per_request(self, tf_fr_server, test_audio_bonjour):
        """Per-request `*` overrides LANGUAGE=fr: the language is identified."""
        result = transcribe_http(tf_fr_server["url"], str(test_audio_bonjour),
                                 language="*", full=True)
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result["text"]), (
            f"Unexpected transcription: {result}"
        )
        assert result["language"] == "fr"

    def test_hotwords_baseline(self, tf_fr_server, test_audio_hotel):
        """Control: without a hotword, the biased word is NOT produced."""
        result = transcribe_http(tf_fr_server["url"], str(test_audio_hotel))
        assert get_expected_regex(str(test_audio_hotel), "fr").search(result), (
            f"Unexpected transcription: {result!r}"
        )
        assert BIASED_WORD not in result.lower(), (
            f"Baseline unexpectedly already contains {BIASED_WORD!r}: {result!r}"
        )

    def test_hotwords(self, tf_fr_server, test_audio_hotel):
        """The per-request hotword biases the transcription toward 'double'."""
        result = transcribe_http(
            tf_fr_server["url"], str(test_audio_hotel), hotwords=HOTWORDS
        )
        assert BIASED_WORD in result.lower(), (
            f"Expected hotword-biased {BIASED_WORD!r} in transcription: {result!r}"
        )


@pytest.mark.hf
@pytest.mark.uv
class TestHfBackendHotwordsEnv:
    """Qwen3-ASR on the transformers backend, HOTWORDS set in the environment."""

    def test_hotwords_env(self, tf_hotwords_env_server, test_audio_hotel):
        """The HOTWORDS env default biases the transcription toward 'double'."""
        result = transcribe_http(tf_hotwords_env_server["url"], str(test_audio_hotel))
        assert BIASED_WORD in result.lower(), (
            f"Expected hotword-biased {BIASED_WORD!r} in transcription: {result!r}"
        )


@pytest.mark.hf
class TestHfBackendDocker:
    """The transformers backend inside the built Docker image (CUDA)."""

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

    def test_websocket_mode_refused(self):
        """The hf engine has no streaming: the server refuses to start in websocket
        mode, before loading anything."""
        import subprocess
        import sys
        from pathlib import Path

        result = subprocess.run([sys.executable, "main.py", "-m", "websocket", "-e", "hf"],
                                cwd=Path(__file__).resolve().parent.parent,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode != 0
        assert "does not support the websocket" in result.stderr


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
        # A language outside Qwen3-ASR's 30 (Kurdish) is rejected by the processor
        # with another message: it must be aligned without a hint as well.
        self._check_bonjour(aligner.align(audio, "Bonjour.", "ku"), audio)
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

    def test_warmup_loads_the_fallback_aligner(self, monkeypatch):
        """The startup warmup loads the language identifier and the fallback
        aligner too, so that no request waits for them (and the GPU memory does
        not grow after startup)."""
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("SERVICE_MODE", "http")
        monkeypatch.setenv("BACKEND", "transformers")
        pytest.importorskip("transformers")
        from linto_stt.engines.hf.stt import processing

        calls = []

        class FakeModel:
            def check_loaded(self):
                calls.append("asr")

            def check_fallback_aligner_loaded(self):
                calls.append("fallback")

            def check_language_identifier_loaded(self):
                calls.append("lid")

        monkeypatch.setattr(processing, "MODEL", FakeModel())
        monkeypatch.setattr(processing, "decode", lambda *a, **k: calls.append("decode") or {})
        processing.warmup()
        assert calls == ["asr", "lid", "decode", "fallback"]

        # Without a fallback, or with alignment disabled, nothing is loaded.
        for primary, fallback in ((processing.ALIGNER_MODEL, None), (None, "mms")):
            model = processing.LazyLoadedModel("unused", device="cpu", dtype="float32",
                                               attn_implementation=None, num_threads=1,
                                               aligner_model_name=primary,
                                               fallback_aligner_model_name=fallback)
            model.check_fallback_aligner_loaded()
            assert model._fallback_aligner is None

    def test_mms_audio_too_short_for_text(self, engine):
        """CTC alignment needs at least one frame per character: on audio too
        short for its text (a few tens of ms, or less than the encoder's 25 ms
        input), the MMS aligner must still return ordered timestamps within the
        audio instead of failing the request."""
        pytest.importorskip("uroman")
        aligners, audio = engine
        aligner = aligners.load_aligner("mms", device="cpu")
        text = "bonjour madame comment allez vous"
        for n_samples in (1600, 160, 0):  # 100 ms, 10 ms, empty
            clip = audio[:n_samples]
            words = aligner.align(clip, text, "fr")
            assert [w["word"] for w in words] == text.split()
            ends = [w["end"] for w in words]
            assert ends == sorted(ends)
            for w in words:
                assert 0 <= w["start"] <= w["end"] <= n_samples / 16000 + 1e-6

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

    def test_fallback_loading_cases(self, monkeypatch, caplog):
        """Which fallback is loaded (at startup or on first need), for every
        combination of primary and fallback aligners. Fake aligners: no model is
        loaded."""
        import logging
        from types import SimpleNamespace

        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("SERVICE_MODE", "http")
        monkeypatch.setenv("BACKEND", "transformers")
        pytest.importorskip("transformers")
        from linto_stt.engines.hf.stt.processing import LazyLoadedModel

        def qwen_like(name):
            return SimpleNamespace(
                name=name, max_duration=300.0, covers_everything=False,
                covers=lambda language, duration=None: (duration is None or duration <= 300)
                and language in (None, "fr", "en"))

        def mms_like(name):
            return SimpleNamespace(name=name, max_duration=None, covers_everything=True,
                                   covers=lambda language, duration=None: True)

        loaded = []

        def load(self, name):
            loaded.append(name)
            return mms_like(name) if name == "mms" else qwen_like(name)

        monkeypatch.setattr(LazyLoadedModel, "_load_aligner", load)

        def model(primary, fallback):
            loaded.clear()
            return LazyLoadedModel("unused", device="cpu", dtype="float32", attn_implementation=None,
                                   num_threads=None, aligner_model_name=primary,
                                   fallback_aligner_model_name=fallback)

        # Default: Qwen aligner, MMS fallback, both loaded.
        m = model("qwen", "mms")
        m.check_fallback_aligner_loaded()
        assert loaded == ["qwen", "mms"]
        # MMS primary covers everything: the fallback (MMS or Qwen) is never loaded.
        for fallback in ("mms", "qwen"):
            m = model("mms", fallback)
            with caplog.at_level(logging.INFO):
                m.check_fallback_aligner_loaded()
            assert loaded == ["mms"] and m._fallback_aligner is None
            assert "not used and not loaded" in caplog.text
            assert m.aligner_for("nl", 4000).name == "mms"
        # Same model as fallback: the primary's instance is reused.
        m = model("qwen", "qwen")
        m.check_fallback_aligner_loaded()
        assert loaded == ["qwen"] and m._fallback_aligner is m._aligner
        assert m.aligner_for("nl", 10).name == "qwen"
        # Two different Qwen-like aligners: a request neither covers goes to the
        # fallback, with a warning.
        m = model("qwen", "qwen-b")
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert m.aligner_for("fr", 400).name == "qwen-b"
        assert "covered by neither" in caplog.text
        # Alignment disabled: nothing is loaded, whatever the fallback.
        m = model(None, "mms")
        m.check_fallback_aligner_loaded()
        assert loaded == []


# ---------------------------------------------------------------------------
# Docker - Celery task mode
# ---------------------------------------------------------------------------


@pytest.mark.hf
@pytest.mark.docker
@pytest.mark.gpu
@pytest.mark.slow
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
