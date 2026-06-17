import re
import string

import pytest

from helpers.env_setup import get_expected_regex
from helpers.transcription_client import transcribe_http, transcribe_websocket


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------

def _nemo_configs(device, vads, servings=("http",),
                  model="nvidia/stt_fr_conformer_ctc_large", architecture="rnnt_bpe"):
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
# Tests check only the start and end, since the middle varies by model/decoding.
HOTEL_EXPECTED_START = "bonjour madame bonjour monsieur"
HOTEL_EXPECTED_END = "deux nuits deux nuits"


def _normalize_loose(text):
    """Lower-case, replace punctuation between words with spaces, and collapse
    whitespace — for case- and punctuation-insensitive comparison."""
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# UV tests
# ---------------------------------------------------------------------------

@pytest.mark.nemo
@pytest.mark.uv
class TestNemoCPU:
    """NeMo engine on CPU via UV subprocess."""

    @pytest.mark.parametrize("uv_server",
        list(_nemo_configs("cpu", ["false"])),
        indirect=True)
    def test_transcription_http(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"


@pytest.mark.nemo
@pytest.mark.uv
@pytest.mark.gpu
class TestNemoGPU:
    """NeMo engine on CUDA via UV subprocess."""

    @pytest.mark.parametrize("uv_server",
        list(_nemo_configs("cuda", [None, "auditok", "silero"])),
        indirect=True)
    def test_transcription_http(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"


@pytest.mark.nemo
@pytest.mark.uv
class TestNemoCTC:
    """NeMo CTC architecture tests."""

    @pytest.mark.parametrize("uv_server",
        list(_nemo_configs(None, ["false"], servings=("http",),
             model="nvidia/stt_fr_conformer_ctc_large", architecture="ctc_bpe")),
        indirect=True)
    def test_ctc_bpe_http(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"


# ---------------------------------------------------------------------------
# UV tests - Streaming (WebSocket)
# ---------------------------------------------------------------------------

@pytest.mark.nemo
@pytest.mark.uv
class TestNemoStreaming:
    """NeMo streaming over the WebSocket serving mode."""

    # DEVICE is left unset so it follows the --device CLI option (default cpu).
    @pytest.mark.parametrize("uv_server", [
        pytest.param(
            {"engine": "nemo", "mode": "websocket", "port": 0,
             "env_overrides": {
                 "MODEL": "linagora/linto_stt_fr_fastconformer_pc",
                 "ARCHITECTURE": "hybrid_bpe_ctc", "VAD": "false"}},
            id="fastconformer_pc",
        ),
        pytest.param(
            {"engine": "nemo", "mode": "websocket", "port": 0,
             "env_overrides": {
                 "MODEL": "nvidia/nemotron-3.5-asr-streaming-0.6b",
                 "ARCHITECTURE": "rnnt_bpe", "VAD": "false"}},
            id="nemotron",
        ),
    ], indirect=True)
    def test_streaming(self, uv_server, test_audio_hotel):
        """Stream Hotel20sec.wav over WebSocket and check the transcription."""
        ws_url = uv_server["url"].replace("http://", "ws://", 1)
        result = transcribe_websocket(ws_url, str(test_audio_hotel),
                                      timeout=uv_server["timeout"])
        normalized = _normalize_loose(result)
        assert normalized.startswith(HOTEL_EXPECTED_START), \
            f"Transcription should start with {HOTEL_EXPECTED_START!r}: {result!r}"
        assert normalized.endswith(HOTEL_EXPECTED_END), \
            f"Transcription should end with {HOTEL_EXPECTED_END!r}: {result!r}"


# ---------------------------------------------------------------------------
# UV tests - Offline decoding (HTTP whole-file)
# ---------------------------------------------------------------------------

@pytest.mark.nemo
@pytest.mark.uv
class TestNemoOffline:
    """NeMo offline (whole-file) decoding over the HTTP serving mode."""

    # DEVICE is left unset so it follows the --device CLI option (default cpu).
    @pytest.mark.parametrize("uv_server", [
        pytest.param(
            {"engine": "nemo", "mode": "http", "port": 0,
             "env_overrides": {
                 "MODEL": "nvidia/nemotron-3.5-asr-streaming-0.6b",
                 "ARCHITECTURE": "rnnt_bpe", "VAD": "false"}},
            id="nemotron",
        ),
    ], indirect=True)
    def test_offline(self, uv_server, test_audio_bonjour):
        """Decode bonjour.wav in one shot over HTTP and check the transcription."""
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"


# ---------------------------------------------------------------------------
# Docker tests
# ---------------------------------------------------------------------------

@pytest.mark.nemo
@pytest.mark.docker
class TestNemoDocker:
    """NeMo engine via Docker container."""

    # DEVICE is left unset so it follows the --device CLI option (default cpu);
    # the docker_server fixture then enables GPU passthrough iff DEVICE is cuda.
    @pytest.mark.parametrize("docker_server", [
        pytest.param(
            {"engine": "nemo", "mode": "http", "port": 0,
             "env_overrides": {"MODEL": "nvidia/stt_fr_conformer_ctc_large",
                               "ARCHITECTURE": "rnnt_bpe",
                               "VAD": "false"}},
            id="auto-vad_false-http",
        ),
    ], indirect=True)
    def test_transcription_http(self, docker_server, test_audio_bonjour):
        result = transcribe_http(docker_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"

    @pytest.mark.parametrize("docker_server", [
        pytest.param(
            {"engine": "nemo", "mode": "task", "port": 0,
             "env_overrides": {"MODEL": "nvidia/stt_fr_conformer_ctc_large",
                               "ARCHITECTURE": "rnnt_bpe",
                               "VAD": "false",
                               "SERVICES_BROKER": "redis://172.17.0.1:6379"}},
            id="auto-vad_false-task",
        ),
    ], indirect=True)
    def test_transcription_task(self, docker_server, test_audio_bonjour, redis_server):
        from helpers.transcription_client import transcribe_celery
        result = transcribe_celery("bonjour.wav", language="fr", broker_url=redis_server,
                                   timeout=docker_server["timeout"])
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"
