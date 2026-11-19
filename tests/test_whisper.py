import pytest

from helpers.env_setup import get_expected_regex
from helpers.transcription_client import transcribe_http, transcribe_websocket


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------


# A falsy `device` (None/"") leaves DEVICE unset, so the server fixture fills it
# from the --device CLI option (default cpu) — i.e. the test "follows --device".
# Pass an explicit "cpu"/"cuda" only to pin a test to one device regardless of
# the CLI (used by the CPU-/GPU-named classes below).
def _whisper_configs(device, vads, model="tiny", language="fr", servings=("http",)):
    for vad in vads:
        for serving in servings:
            env = {"MODEL": model, "LANGUAGE": language}
            if device:
                env["DEVICE"] = device
            if vad:
                env["VAD"] = vad
            yield pytest.param(
                {"engine": "whisper", "mode": serving, "port": 0, "env_overrides": env},
                id=f"{device or 'auto'}-vad_{vad or 'none'}-{serving}",
            )


def _whisper_docker_configs(
    device,
    vads,
    dockerfile,
    model="tiny",
    language="fr",
    servings=("http",),
    use_gpu=False,
):
    # Same device convention as _whisper_configs.
    for vad in vads:
        for serving in servings:
            env = {"MODEL": model, "LANGUAGE": language}
            if device:
                env["DEVICE"] = device
            if vad:
                env["VAD"] = vad
            yield pytest.param(
                {
                    "engine": "whisper",
                    "mode": serving,
                    "port": 0,
                    "dockerfile": dockerfile,
                    "use_gpu": use_gpu,
                    "env_overrides": env,
                },
                id=f"{dockerfile.split('/')[-1]}-{device or 'auto'}-vad_{vad or 'none'}-{serving}",
            )


# ---------------------------------------------------------------------------
# UV tests - CPU
# ---------------------------------------------------------------------------


@pytest.mark.whisper
@pytest.mark.uv
class TestWhisperCPU:
    """Whisper engine on CPU via UV subprocess."""

    @pytest.mark.parametrize(
        "uv_server",
        list(_whisper_configs("cpu", ["false", "auditok", "silero"])),
        indirect=True,
    )
    def test_integration_cpu(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


@pytest.mark.whisper
@pytest.mark.uv
@pytest.mark.gpu
class TestWhisperGPU:
    """Whisper engine on CUDA via UV subprocess."""

    @pytest.mark.parametrize(
        "uv_server", list(_whisper_configs("cuda", [None])), indirect=True
    )
    def test_integration_cuda(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


@pytest.mark.whisper
@pytest.mark.uv
class TestWhisperNoDevice:
    """Whisper engine without an explicit device pin: the test sets no DEVICE,
    so it runs on whatever --device selects (default cpu)."""

    @pytest.mark.parametrize(
        "uv_server", list(_whisper_configs(None, [None])), indirect=True
    )
    def test_integration_nodevice(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


# ---------------------------------------------------------------------------
# UV tests - Language variations
# ---------------------------------------------------------------------------


@pytest.mark.whisper
@pytest.mark.uv
class TestWhisperLanguages:
    """Whisper language-specific tests."""

    @pytest.mark.parametrize(
        "uv_server",
        [
            pytest.param(
                {
                    "engine": "whisper",
                    "mode": "http",
                    "port": 0,
                    "env_overrides": {"MODEL": "tiny", "LANGUAGE": "*"},
                },
                id="nolanguage",
            ),
        ],
        indirect=True,
    )
    def test_nolanguage(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )

    @pytest.mark.parametrize(
        "uv_server",
        [
            pytest.param(
                {
                    "engine": "whisper",
                    "mode": "http",
                    "port": 0,
                    "env_overrides": {"MODEL": "tiny", "LANGUAGE": "FR-FR"},
                },
                id="languagecode-FR-FR",
            ),
        ],
        indirect=True,
    )
    def test_languagecode(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )

    @pytest.mark.parametrize(
        "uv_server",
        [
            pytest.param(
                {
                    "engine": "whisper",
                    "mode": "http",
                    "port": 0,
                    "env_overrides": {"MODEL": "tiny", "LANGUAGE": "ru"},
                },
                id="russian",
            ),
        ],
        indirect=True,
    )
    def test_russian(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "ru").search(result), (
            f"Unexpected transcription: {result}"
        )

    @pytest.mark.parametrize(
        "uv_server",
        [
            pytest.param(
                {
                    "engine": "whisper",
                    "mode": "http",
                    "port": 0,
                    "env_overrides": {"MODEL": "tiny", "LANGUAGE": "ru"},
                },
                id="language-over-config",
            ),
        ],
        indirect=True,
    )
    def test_language_over_config(self, uv_server, test_audio_bonjour):
        """Config says ru, but request overrides to fr."""
        result = transcribe_http(
            uv_server["url"], str(test_audio_bonjour), language="fr"
        )
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


# ---------------------------------------------------------------------------
# UV tests - Hotwords
# ---------------------------------------------------------------------------


@pytest.mark.whisper
@pytest.mark.uv
class TestWhisperHotwords:
    """Whisper hotwords biasing (faster-whisper / CTranslate2 backend)."""

    @pytest.mark.parametrize(
        "uv_server",
        [
            pytest.param(
                {
                    "engine": "whisper",
                    "mode": "http",
                    "port": 0,
                    "env_overrides": {
                        "MODEL": "tiny",
                        "LANGUAGE": "fr",
                        "VAD": "false",
                        "HOTWORDS": "BonJour AuRevoir PourquoiPas",
                    },
                },
                id="hotwords-bonjour",
            ),
        ],
        indirect=True,
    )
    def test_hotwords_spelling(self, uv_server, test_audio_bonjour):
        """The HOTWORDS list should bias the spelling toward 'BonJour'."""
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert "BonJour" in result, (
            f"Expected hotword spelling 'BonJour' in transcription: {result!r}"
        )

    @pytest.mark.parametrize(
        "uv_server",
        [
            pytest.param(
                {
                    "engine": "whisper",
                    "mode": "http",
                    "port": 0,
                    "env_overrides": {
                        "MODEL": "tiny",
                        "LANGUAGE": "fr",
                        "VAD": "false",
                        # HOTWORDS deliberately NOT set: the biasing must come
                        # from the per-request `hotwords` query parameter.
                    },
                },
                id="hotwords-per-request-bonjour",
            ),
        ],
        indirect=True,
    )
    def test_hotwords_per_request(self, uv_server, test_audio_bonjour):
        """The per-request `hotwords` query param should bias the spelling toward
        'BonJour' even with no HOTWORDS env set, proving it overrides the default."""
        result = transcribe_http(
            uv_server["url"], str(test_audio_bonjour),
            hotwords="BonJour AuRevoir PourquoiPas",
        )
        assert "BonJour" in result, (
            f"Expected request hotword spelling 'BonJour' in transcription: {result!r}"
        )


# ---------------------------------------------------------------------------
# UV tests - Streaming (WebSocket)
# ---------------------------------------------------------------------------


@pytest.mark.whisper
@pytest.mark.uv
class TestWhisperVADNoSpeech:
    """With VAD enabled, an audio with no speech at all must give an empty
    transcription. Before the fix, when the VAD found no segment the whole
    audio was decoded (clip_timestamps default) and Whisper hallucinated
    subtitle credits on silent chunks."""

    @pytest.mark.parametrize(
        "uv_server",
        list(_whisper_configs(None, ["auditok", "silero"])),
        indirect=True,
    )
    def test_no_speech_gives_empty_output(self, uv_server, test_audio_no_speech):
        for name, path in test_audio_no_speech.items():
            result = transcribe_http(uv_server["url"], str(path))
            assert result == "", f"Hallucinated text on {name}: {result!r}"


@pytest.mark.whisper
@pytest.mark.uv
class TestWhisperStreaming:
    """Whisper streaming over the WebSocket serving mode."""

    @pytest.mark.parametrize(
        "uv_server",
        [
            pytest.param(
                {
                    "engine": "whisper",
                    "mode": "websocket",
                    "port": 0,
                    "env_overrides": {"MODEL": "tiny", "LANGUAGE": "fr"},
                },
                id="streaming-bonjour",
            ),
        ],
        indirect=True,
    )
    def test_streaming(self, uv_server, test_audio_bonjour):
        """Stream bonjour.wav over WebSocket and check the transcription."""
        ws_url = uv_server["url"].replace("http://", "ws://", 1)
        result = transcribe_websocket(
            ws_url, str(test_audio_bonjour), timeout=uv_server["timeout"]
        )
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected streaming transcription: {result!r}"
        )


# ---------------------------------------------------------------------------
# UV tests - Model sizes
# ---------------------------------------------------------------------------


@pytest.mark.whisper
@pytest.mark.uv
@pytest.mark.slow
class TestWhisperModels:
    """Whisper model size tests."""

    @pytest.mark.parametrize(
        "uv_server",
        [
            pytest.param(
                {
                    "engine": "whisper",
                    "mode": "http",
                    "port": 0,
                    "env_overrides": {"MODEL": "small", "LANGUAGE": "fr"},
                },
                id="model-small",
            ),
        ],
        indirect=True,
    )
    def test_model_small(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


# ---------------------------------------------------------------------------
# Docker tests
# ---------------------------------------------------------------------------


@pytest.mark.whisper
@pytest.mark.docker
class TestWhisperDockerCPU:
    """Whisper engine via Docker (CPU dockerfiles)."""

    @pytest.mark.parametrize(
        "docker_server",
        list(
            _whisper_docker_configs(
                "cpu", ["false", "auditok", "silero"], dockerfile="Dockerfile"
            )
        ),
        indirect=True,
    )
    def test_integration_cpu(self, docker_server, test_audio_bonjour):
        result = transcribe_http(docker_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


@pytest.mark.whisper
@pytest.mark.docker
@pytest.mark.gpu
class TestWhisperDockerGPU:
    """Whisper engine via Docker (CUDA dockerfiles)."""

    @pytest.mark.parametrize(
        "docker_server",
        list(
            _whisper_docker_configs(
                "cuda", [None], dockerfile="Dockerfile", use_gpu=True
            )
        ),
        indirect=True,
    )
    def test_integration_cuda(self, docker_server, test_audio_bonjour):
        result = transcribe_http(docker_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


# ---------------------------------------------------------------------------
# Docker tests - Language variations (Celery)
# ---------------------------------------------------------------------------


@pytest.mark.whisper
@pytest.mark.docker
class TestWhisperDockerCelery:
    """Whisper Celery task tests via Docker."""

    @pytest.mark.parametrize(
        "docker_server",
        [
            pytest.param(
                {
                    "engine": "whisper",
                    "mode": "task",
                    "port": 0,
                    "env_overrides": {
                        "MODEL": "tiny",
                        "LANGUAGE": "ru",
                        "SERVICES_BROKER": "redis://172.17.0.1:6379",
                    },
                },
                id="russian-task",
            ),
        ],
        indirect=True,
    )
    def test_russian_celery(self, docker_server, test_audio_bonjour, redis_server):
        from helpers.transcription_client import transcribe_celery

        result = transcribe_celery(
            "bonjour.wav", language="ru", broker_url=redis_server
        )
        assert get_expected_regex(str(test_audio_bonjour), "ru").search(result), (
            f"Unexpected transcription: {result}"
        )

    @pytest.mark.parametrize(
        "docker_server",
        [
            pytest.param(
                {
                    "engine": "whisper",
                    "mode": "task",
                    "port": 0,
                    "env_overrides": {
                        "MODEL": "tiny",
                        "LANGUAGE": "ru",
                        "SERVICES_BROKER": "redis://172.17.0.1:6379",
                    },
                },
                id="language-over-config-task",
            ),
        ],
        indirect=True,
    )
    def test_language_over_config_celery(
        self, docker_server, test_audio_bonjour, redis_server
    ):
        from helpers.transcription_client import transcribe_celery

        result = transcribe_celery(
            "bonjour.wav", language="fr", broker_url=redis_server
        )
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


# ---------------------------------------------------------------------------
# Hallucinations: detector and re-decoding (unit tests, no model)
# ---------------------------------------------------------------------------
# Imports are done in fixtures, so that a missing faster-whisper (e.g. Docker
# tests run with only the `test` extra) skips these tests, not the whole file.


@pytest.fixture(scope="module")
def hallucinations():
    """The detector module, loaded from its file (it only needs `regex`)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent.parent / "linto_stt/engines/whisper/stt/processing/hallucinations.py"
    spec = importlib.util.spec_from_file_location("hallucinations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _detected(hallucinations, text, whole_segment=False):
    words = text.split()
    segments = [(0, len(words) - 1)] if whole_segment else None
    return " ".join(words[i] for i in sorted(hallucinations.find_hallucinated_words(words, segments)))


@pytest.mark.whisper
@pytest.mark.uv
class TestWhisperHallucinationDetector:
    """The detector of known hallucinated phrases (hallucinations.py)."""

    @pytest.mark.parametrize("text, expected", [
        # French credits, as written by Whisper (punctuation, glued apostrophes)
        ("Sous-titrage Société Radio-Canada", "Sous-titrage Société Radio-Canada"),
        ("Sous-titrage ST' 501", "Sous-titrage ST' 501"),
        ("Sous-titrage ST'501", "Sous-titrage ST'501"),
        ("Sous-titrage FR ?", "Sous-titrage FR ?"),
        ("Sous-titrage Société", "Sous-titrage Société"),
        ("Sous-titres par Jérémy Diaz", "Sous-titres par Jérémy Diaz"),
        ("Sous-titres réalisés par la communauté d'Amara.org", "Sous-titres réalisés par la communauté d'Amara.org"),
        # glued to real speech
        ("60 euros per month. Sous-titrage Société Radio-Canada", "Sous-titrage Société Radio-Canada"),
        ("commerciales, tu vois. Sous-titrage ST'501 Au-delà des licences", "Sous-titrage ST'501"),
        # outros, including two in a row
        ("Merci d'avoir regardé cette vidéo !", "Merci d'avoir regardé cette vidéo !"),
        ("Merci d'avoir regardé cette vidéo! Merci d'avoir regardé cette vidéo!",
         "Merci d'avoir regardé cette vidéo! Merci d'avoir regardé cette vidéo!"),
        ("Merci à mes Tipeurs et souscripteurs.", "Merci à mes Tipeurs et souscripteurs."),
        # other languages, whatever the language of the speech
        ("Untertitel im Auftrag des ZDF, 2017", "Untertitel im Auftrag des ZDF, 2017"),
        ("Thanks for watching!", "Thanks for watching!"),
        ("Subtitles by the Amara.org community", "Subtitles by the Amara.org community"),
        ("ご視聴ありがとうございました", "ご視聴ありがとうございました"),
    ])
    def test_phrases_detected(self, hallucinations, text, expected):
        assert _detected(hallucinations, text) == expected

    @pytest.mark.parametrize("text", [
        "le sous-titrage en temps réel est disponible",
        "Radio-Canada a diffusé le reportage",
        "merci d'avoir regardé le document",
        "Merci. Thank you. Bye.",
        "on peut sous-titrer la vidéo",
        "gracias por ver",
        "La Société générale",
    ])
    def test_real_speech_not_detected(self, hallucinations, text):
        assert _detected(hallucinations, text) == ""
        assert _detected(hallucinations, text, whole_segment=True) == ""

    @pytest.mark.parametrize("text", ["Sous-titrage", "Sous-titrage.", "Merci d'avoir regardé", "Untertitel"])
    def test_fragments_detected_only_as_a_whole_segment(self, hallucinations, text):
        assert _detected(hallucinations, text, whole_segment=True) == text
        # inside a longer segment, the same words are ordinary speech
        assert _detected(hallucinations, f"on parlait de {text} hier") == ""


_SR = 16000


@pytest.fixture(scope="module")
def redecode():
    """The re-decoding module and faster-whisper's Segment / Word classes."""
    pytest.importorskip("faster_whisper")
    from types import SimpleNamespace

    import numpy as np
    from faster_whisper.transcribe import Segment, Word

    from linto_stt.engines.whisper.stt.processing import redecode as module

    return SimpleNamespace(module=module, Segment=Segment, Word=Word, np=np, info=SimpleNamespace(language="fr"))


def _rd_segment(rd, words, start_id=0):
    """A faster-whisper Segment from [(start, end, word)], words written with a leading space."""
    ws = [rd.Word(start=s, end=e, word=w, probability=0.9) for s, e, w in words]
    return rd.Segment(id=start_id, seek=0, start=ws[0].start, end=ws[-1].end, text="".join(w.word for w in ws),
                      tokens=[], avg_logprob=-0.3, compression_ratio=1.2, no_speech_prob=0.0, words=ws,
                      temperature=0.0)


def _rd_speech(start, end, n, prefix="mot"):
    """n distinct words evenly spread over [start, end)."""
    step = (end - start) / n
    return [(start + i * step, start + i * step + step * 0.8, f" {prefix}{i}") for i in range(n)]


def _rd_text(segments):
    return " ".join(w.word.strip() for s in segments for w in s.words)


_RD_CREDIT = [(7.22, 10.02, " Sous-titrage"), (10.02, 11.10, " Société"), (11.10, 11.10, " Radio"),
              (11.10, 11.10, "-Canada")]
# a credit between two time ranges of real speech, from 10 to 12 s
_RD_CREDIT_GLUED = [(10.0, 11.0, " Sous-titrage"), (11.0, 11.5, " ST'"), (11.5, 12.0, " 501")]


class _FakeModel:
    """Returns the scripted outputs in order; records the calls."""

    def __init__(self, outputs, info):
        self.outputs = list(outputs)
        self.info = info
        self.calls = []

    def transcribe(self, audio, language=None, word_timestamps=None, **kwargs):
        self.calls.append({"duration": len(audio) / _SR, "language": language, "hotwords": kwargs.get("hotwords")})
        return iter(self.outputs.pop(0)), self.info


@pytest.mark.whisper
@pytest.mark.uv
class TestWhisperRedecode:
    """The re-decoding of suspicious time ranges (redecode.py), with a fake model."""

    def test_credit_over_speech_is_redecoded(self, redecode):
        """The 00:50 case of a LinTO Studio meeting: a credit, then nothing until
        the next window. The first attempts hallucinate again; the window shifted
        by 1 s gives the speech."""
        audio = redecode.np.zeros(40 * _SR, dtype="float32")
        segments = [_rd_segment(redecode, _RD_CREDIT), _rd_segment(redecode, _rd_speech(30, 40, 20, "fin"), 1)]
        model = _FakeModel([
            [_rd_segment(redecode, _RD_CREDIT)],                       # start at the time range, with hotwords
            [_rd_segment(redecode, [(7.0, 9.0, " Sous-titrage")])],    # without hotwords: the fragment alone
            [_rd_segment(redecode, _rd_speech(0, 28, 30))],            # start +1 s: speech
        ], redecode.info)
        out = redecode.module.redecode_suspicious(model, audio, segments, redecode.info, [(0.0, 40.0)],
                                                  {"hotwords": "LinTO", "beam_size": 5}, min_gap=6.0)
        text = _rd_text(out)
        assert "Sous-titrage" not in text and "mot0" in text and "fin19" in text
        assert [c["hotwords"] for c in model.calls] == ["LinTO", None, "LinTO"]
        assert all(c["language"] == "fr" for c in model.calls)
        words = [w for s in out for w in s.words]
        assert words == sorted(words, key=lambda w: w.start)
        assert all(0.0 <= w.start < 30.0 for w in words if w.word.strip().startswith("mot"))

    def test_kept_when_no_attempt_is_accepted(self, redecode):
        audio = redecode.np.zeros(40 * _SR, dtype="float32")
        segments = [_rd_segment(redecode, _RD_CREDIT), _rd_segment(redecode, _rd_speech(30, 40, 20, "fin"), 1)]
        model = _FakeModel([[_rd_segment(redecode, _RD_CREDIT)]] * 10, redecode.info)
        out = redecode.module.redecode_suspicious(model, audio, segments, redecode.info, [(0.0, 40.0)],
                                                  {"hotwords": None}, min_gap=6.0)
        assert _rd_text(out) == _rd_text(segments)
        assert len(model.calls) == 3  # same start, +1 s, +2 s (no hotwords to remove)

    def test_repetitive_attempt_rejected(self, redecode):
        audio = redecode.np.zeros(40 * _SR, dtype="float32")
        segments = [_rd_segment(redecode, _rd_speech(30, 40, 20, "fin"))]
        model = _FakeModel([[_rd_segment(redecode, [(i, i + 0.5, " non,") for i in range(20)])],
                            [_rd_segment(redecode, _rd_speech(0, 28, 30))]], redecode.info)
        out = redecode.module.redecode_suspicious(model, audio, segments, redecode.info, [(0.0, 40.0)], {},
                                                  min_gap=6.0)
        assert "non," not in _rd_text(out) and "mot0" in _rd_text(out)

    def test_credit_glued_to_speech_is_redecoded(self, redecode):
        """A credit between two time ranges of real speech (no long gap): its own
        time range is decoded again."""
        audio = redecode.np.zeros(20 * _SR, dtype="float32")
        segments = [_rd_segment(redecode, _rd_speech(0, 10, 20, "avant") + _RD_CREDIT_GLUED + _rd_speech(12, 20, 16, "apres"))]
        model = _FakeModel([[_rd_segment(redecode, [(0.2, 0.8, " vrai"), (0.9, 1.5, " texte")])]], redecode.info)
        out = redecode.module.redecode_suspicious(model, audio, segments, redecode.info, [(0.0, 20.0)], {},
                                                  min_gap=6.0)
        text = _rd_text(out)
        assert "Sous-titrage" not in text and "vrai texte" in text
        assert "avant19" in text and "apres0" in text
        assert len(model.calls) == 1

    def test_nothing_to_do(self, redecode):
        audio = redecode.np.zeros(20 * _SR, dtype="float32")
        segments = [_rd_segment(redecode, _rd_speech(0, 20, 40))]
        model = _FakeModel([], redecode.info)
        assert redecode.module.redecode_suspicious(model, audio, segments, redecode.info, [(0.0, 20.0)], {}) == segments
        assert model.calls == []
        # without word timestamps, nothing is done
        no_words = [redecode.Segment(**{**segments[0].__dict__, "words": None})]
        assert redecode.module.redecode_suspicious(model, audio, no_words, redecode.info, [(0.0, 20.0)], {}) == no_words

    def test_known_phrase_removed_when_retry_finds_no_speech(self, redecode):
        """A credit in a time range where the retries, with context, write no word
        while transcribing the speech after it: no speech there, the credit is
        removed. Without any word after it (nothing heard at all), it is kept."""
        audio = redecode.np.zeros(20 * _SR, dtype="float32")
        segments = [_rd_segment(redecode, _rd_speech(0, 10, 20, "avant") + _RD_CREDIT_GLUED + _rd_speech(12, 20, 16, "apres"))]
        # the speech after the time range, at 12.5 to 14 s, for the attempts starting
        # at 10 s, 11 s (+1 s) and 9 s (-1 s)
        model = _FakeModel([[_rd_segment(redecode, [(12.5 - t, 13.0 - t, " suite"), (13.1 - t, 14.0 - t, " normale")])]
                            for t in (10.0, 11.0, 9.0)], redecode.info)
        out = redecode.module.redecode_suspicious(model, audio, segments, redecode.info, [(0.0, 20.0)], {},
                                                  min_gap=6.0)
        text = _rd_text(out)
        assert "Sous-titrage" not in text and "suite" not in text
        assert "avant19" in text and "apres0" in text
        model = _FakeModel([[]] * 3, redecode.info)
        out = redecode.module.redecode_suspicious(model, audio, segments, redecode.info, [(0.0, 20.0)], {},
                                                  min_gap=6.0)
        assert "Sous-titrage" in _rd_text(out)
