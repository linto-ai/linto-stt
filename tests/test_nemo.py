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

    # Reference transcription of Hotel20sec.wav (LinTO French FastConformer PC
    # model, CTC decoding). Kept for documentation; the test only checks the
    # start and end (see below), since streaming output varies in the middle:
    #   "Bonjour Madame. Bonjour Monsieur. En quoi peux-je vous aider
    #    aujourd'hui ? Euh j'aimerais avoir une chambre pour deux personnes.
    #    Deux personnes, chambre double, d'accord. Euh deux lits. Deux lits
    #    d'accord. Et puis ce serait pour deux nuits. Deux nuits"
    EXPECTED_START = "bonjour madame bonjour monsieur"
    EXPECTED_END = "deux nuits deux nuits"

    @pytest.mark.parametrize("uv_server",
        list(_nemo_configs(None, ["false"], servings=("websocket",),
             model="linagora/linto_stt_fr_fastconformer_pc",
             architecture="hybrid_bpe_ctc")),
        indirect=True)
    def test_streaming(self, uv_server, test_audio_hotel):
        """Stream Hotel20sec.wav over WebSocket and check the transcription."""
        ws_url = uv_server["url"].replace("http://", "ws://", 1)
        result = transcribe_websocket(ws_url, str(test_audio_hotel),
                                      timeout=uv_server["timeout"])
        # Compare case- and punctuation-insensitively: lower-case and replace
        # any punctuation between words with a space, then collapse whitespace.
        normalized = re.sub(
            rf"[{re.escape(string.punctuation)}]", " ", result.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        assert normalized.startswith(self.EXPECTED_START), \
            f"Transcription should start with {self.EXPECTED_START!r}: {result!r}"
        assert normalized.endswith(self.EXPECTED_END), \
            f"Transcription should end with {self.EXPECTED_END!r}: {result!r}"


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
