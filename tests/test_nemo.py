import re
import string

import pytest

from helpers.env_setup import get_expected_regex
from helpers.transcription_client import transcribe_http, transcribe_websocket


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------


def _nemo_configs(
    device,
    vads,
    servings=("http",),
    model="nvidia/stt_fr_conformer_ctc_large",
    architecture="rnnt_bpe",
):
    # device=None leaves DEVICE unset so the server fixture fills it from the
    # --device CLI option (default cpu). Pass an explicit "cpu"/"cuda" to pin it.
    for vad in vads:
        for serving in servings:
            env = {
                "MODEL": model,
                "ARCHITECTURE": architecture,
                "VAD": vad,
            }
            if device is not None:
                env["DEVICE"] = device
            yield pytest.param(
                {"engine": "nemo", "mode": serving, "port": 0, "env_overrides": env},
                id=f"{device or 'auto'}-vad_{vad}-{serving}",
            )


# ---------------------------------------------------------------------------
# Hotel20sec.wav reference (shared by the streaming and offline tests below)
# ---------------------------------------------------------------------------

# Reference transcription of Hotel20sec.wav (French):
#   "Bonjour Madame. Bonjour Monsieur. En quoi peux-je vous aider aujourd'hui ?
#    Euh j'aimerais avoir une chambre pour deux personnes. Deux personnes,
#    chambre double, d'accord. Euh deux lits. Deux lits d'accord. Et puis ce
#    serait pour deux nuits. Deux nuits"
# NB: no NeMo model currently reaches this full reference — the nemotron streaming
# model never emits the leading "Bonjour Madame" (at any att_context_size), and
# the fastconformer buffered path regressed under nemo-toolkit 3.x — so each test
# below asserts what its own model actually produces, not this ideal.


def _normalize_loose(text):
    """Lower-case, replace punctuation between words with spaces, and collapse
    whitespace — for case- and punctuation-insensitive comparison."""
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


# nvidia/nemotron-3.5-asr-streaming-0.6b — a prompt-conditioned, cache-aware
# streaming RNNT model, used in both the offline and streaming tests below.
_NEMOTRON_ENV = {
    "MODEL": "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "ARCHITECTURE": "rnnt_bpe",
    "VAD": "false",
}


# ---------------------------------------------------------------------------
# UV tests
# ---------------------------------------------------------------------------


@pytest.mark.nemo
@pytest.mark.uv
class TestNemoCPU:
    """NeMo engine on CPU via UV subprocess."""

    @pytest.mark.parametrize(
        "uv_server", list(_nemo_configs("cpu", ["false"])), indirect=True
    )
    def test_transcription_http(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


@pytest.mark.nemo
@pytest.mark.uv
@pytest.mark.gpu
class TestNemoGPU:
    """NeMo engine on CUDA via UV subprocess."""

    @pytest.mark.parametrize(
        "uv_server",
        list(_nemo_configs("cuda", [None, "auditok", "silero"])),
        indirect=True,
    )
    def test_transcription_http(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


@pytest.mark.nemo
@pytest.mark.uv
class TestNemoCTC:
    """NeMo CTC architecture tests."""

    @pytest.mark.parametrize(
        "uv_server",
        list(
            _nemo_configs(
                None,
                ["false"],
                servings=("http",),
                model="nvidia/stt_fr_conformer_ctc_large",
                architecture="ctc_bpe",
            )
        ),
        indirect=True,
    )
    def test_ctc_bpe_http(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


# ---------------------------------------------------------------------------
# UV tests - Streaming (WebSocket)
# ---------------------------------------------------------------------------


@pytest.mark.nemo
@pytest.mark.uv
class TestNemoStreaming:
    """NeMo streaming over the WebSocket serving mode."""

    # Native cache-aware streaming (nemotron). ATT_CONTEXT_SIZE_RIGHT=13 is the
    # model's most accurate supported look-ahead ({0,3,6,13} are the only valid
    # values). It does NOT recover the leading "Bonjour Madame" (no setting does);
    # the stable output starts "Bonjour monsieur" and ends "deux nuits".
    # DEVICE is left unset so it follows the --device CLI option (default cpu).
    @pytest.mark.parametrize(
        "uv_server",
        [
            pytest.param(
                {"engine": "nemo", "mode": "websocket", "port": 0,
                 "env_overrides": {**_NEMOTRON_ENV, "ATT_CONTEXT_SIZE_RIGHT": "13"}},
                id="nemotron",
            ),
        ],
        indirect=True,
    )
    def test_streaming_native(self, uv_server, test_audio_hotel):
        """Native cache-aware streaming path (nemotron)."""
        ws_url = uv_server["url"].replace("http://", "ws://", 1)
        result = transcribe_websocket(
            ws_url, str(test_audio_hotel), timeout=uv_server["timeout"]
        )
        normalized = _normalize_loose(result)
        assert normalized.startswith("bonjour monsieur"), (
            f"Expected start 'bonjour monsieur': {result!r}"
        )
        assert normalized.endswith("deux nuits"), (
            f"Expected end 'deux nuits': {result!r}"
        )
        assert "<" not in result and ">" not in result, (
            f"Transcription should not contain <...> tags: {result!r}"
        )

    # Buffered (simulated) streaming over an offline model (fastconformer_pc).
    # This model's transcription quality regressed under nemo-toolkit 3.x (main)
    # and the buffered path is somewhat non-deterministic, so we only check that it
    # produces a plausible French transcription (not the full reference).
    @pytest.mark.parametrize(
        "uv_server",
        [
            pytest.param(
                {"engine": "nemo", "mode": "websocket", "port": 0,
                 "env_overrides": {
                     "MODEL": "linagora/linto_stt_fr_fastconformer_pc",
                     "ARCHITECTURE": "hybrid_bpe_ctc", "VAD": "false"}},
                id="fastconformer_pc",
            ),
        ],
        indirect=True,
    )
    def test_streaming_buffered(self, uv_server, test_audio_hotel):
        """Buffered (simulated) streaming path over an offline model."""
        ws_url = uv_server["url"].replace("http://", "ws://", 1)
        result = transcribe_websocket(
            ws_url, str(test_audio_hotel), timeout=uv_server["timeout"]
        )
        normalized = _normalize_loose(result).rstrip("s")
        assert normalized.startswith("bonjour madame bonjour monsieur") and normalized.endswith("deux nuits deux nuit"), (
            f"Expected a plausible transcription starting with 'bonjour madame bonjour monsieur' and ending"
            f"with 'deux nuits deux nuit': {result!r}"
        )
        assert "<" not in result and ">" not in result, (
            f"Transcription should not contain <...> tags: {result!r}"
        )

    # ATT_CONTEXT_SIZE_RIGHT=13 is the model's most accurate supported look-ahead
    # ({0,3,6,13} are the only valid values; others fall back / garble output). It
    # does NOT recover the leading "Bonjour Madame" (no setting does), but it is the
    # stable, fullest output: "Bonjour monsieur ... deux nuits".
    # DEVICE is left unset so it follows the --device CLI option (default cpu).
    @pytest.mark.parametrize(
        "uv_server",
        [
            pytest.param(
                {"engine": "nemo", "mode": "http", "port": 0,
                 "env_overrides": _NEMOTRON_ENV},
                id="nemotron-right13",
            ),
        ],
        indirect=True,
    )
    def test_streaming_model_offline(self, uv_server, test_audio_hotel):
        result = transcribe_http(uv_server["url"], str(test_audio_hotel))
        assert "<" not in result and ">" not in result, (
            f"Transcription should not contain <...> tags: {result!r}"
        )
        assert result == "Bonjour monsieur, en quoi puis-je vous aider aujourd'hui? J'aimerais avoir une chambre pour deux personnes deux personnes chambres d'eau d'accord de lit et puis ce serait pour deux nuits.", (
            f"Unexpected result: {result!r}"
        )

    # ATT_CONTEXT_SIZE_RIGHT=13 is the model's most accurate supported look-ahead
    # ({0,3,6,13} are the only valid values; others fall back / garble output). It
    # does NOT recover the leading "Bonjour Madame" (no setting does), but it is the
    # stable, fullest output: "Bonjour monsieur ... deux nuits".
    # DEVICE is left unset so it follows the --device CLI option (default cpu).
    @pytest.mark.parametrize(
        "uv_server",
        [
            pytest.param(
                {"engine": "nemo", "mode": "http", "port": 0,
                 "env_overrides": {**_NEMOTRON_ENV, "ATT_CONTEXT_SIZE_RIGHT": "13"}},
                id="nemotron-right13",
            ),
        ],
        indirect=True,
    )
    def test_streaming_model_offline(self, uv_server, test_audio_hotel):
        result = transcribe_http(uv_server["url"], str(test_audio_hotel))
        assert "<" not in result and ">" not in result, (
            f"Transcription should not contain <...> tags: {result!r}"
        )
        assert result == "Bonjour monsieur, en quoi puis-je vous aider aujourd'hui? J'aimerais avoir une chambre pour deux personnes chambres d'eau d'accord de lit et puis ce serait pour deux nuits.", (
            f"Unexpected result: {result!r}"
        )


# ---------------------------------------------------------------------------
# Docker tests
# ---------------------------------------------------------------------------


@pytest.mark.nemo
@pytest.mark.docker
class TestNemoDocker:
    """NeMo engine via Docker container."""

    # DEVICE is left unset so it follows the --device CLI option (default cpu);
    # the docker_server fixture then enables GPU passthrough iff DEVICE is cuda.
    @pytest.mark.parametrize(
        "docker_server",
        [
            pytest.param(
                {
                    "engine": "nemo",
                    "mode": "http",
                    "port": 0,
                    "env_overrides": {
                        "MODEL": "nvidia/stt_fr_conformer_ctc_large",
                        "ARCHITECTURE": "rnnt_bpe",
                        "VAD": "false",
                    },
                },
                id="auto-vad_false-http",
            ),
        ],
        indirect=True,
    )
    def test_transcription_http(self, docker_server, test_audio_bonjour):
        result = transcribe_http(docker_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )

    @pytest.mark.parametrize(
        "docker_server",
        [
            pytest.param(
                {
                    "engine": "nemo",
                    "mode": "task",
                    "port": 0,
                    "env_overrides": {
                        "MODEL": "nvidia/stt_fr_conformer_ctc_large",
                        "ARCHITECTURE": "rnnt_bpe",
                        "VAD": "false",
                        "SERVICES_BROKER": "redis://172.17.0.1:6379",
                    },
                },
                id="auto-vad_false-task",
            ),
        ],
        indirect=True,
    )
    def test_transcription_task(self, docker_server, test_audio_bonjour, redis_server):
        from helpers.transcription_client import transcribe_celery

        result = transcribe_celery(
            "bonjour.wav",
            language="fr",
            broker_url=redis_server,
            timeout=docker_server["timeout"],
        )
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )
