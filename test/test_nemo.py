import pytest

from helpers.env_setup import get_expected_regex
from helpers.transcription_client import transcribe_http, transcribe_websocket


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------

def _nemo_configs(device, vads, servings=("http",),
                  model="nvidia/parakeet-tdt-0.6b-v2", architecture="rnnt_bpe"):
    for vad in vads:
        for serving in servings:
            env = {
                "MODEL": model,
                "ARCHITECTURE": architecture,
                "DEVICE": device,
                "VAD": vad,
            }
            yield pytest.param(
                {"backend": "nemo", "mode": serving, "port": 0, "env_overrides": env},
                id=f"{device}-vad_{vad}-{serving}",
            )


# ---------------------------------------------------------------------------
# UV tests
# ---------------------------------------------------------------------------

@pytest.mark.nemo
@pytest.mark.uv
class TestNemoCPU:
    """NeMo backend on CPU via UV subprocess."""

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
    """NeMo backend on CUDA via UV subprocess."""

    @pytest.mark.parametrize("uv_server",
        list(_nemo_configs("cuda", [None, "false", "auditok", "silero"])),
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
        list(_nemo_configs("cpu", ["false"], servings=("http",),
             model="nvidia/stt_fr_conformer_ctc_large", architecture="ctc_bpe")),
        indirect=True)
    def test_ctc_bpe_http(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"


# ---------------------------------------------------------------------------
# Docker tests
# ---------------------------------------------------------------------------

@pytest.mark.nemo
@pytest.mark.docker
class TestNemoDocker:
    """NeMo backend via Docker container."""

    @pytest.mark.parametrize("docker_server", [
        pytest.param(
            {"backend": "nemo", "mode": "http", "port": 0,
             "env_overrides": {"MODEL": "nvidia/parakeet-tdt-0.6b-v2",
                               "ARCHITECTURE": "rnnt_bpe",
                               "DEVICE": "cpu", "VAD": "false"}},
            id="cpu-vad_false-http",
        ),
    ], indirect=True)
    def test_transcription_http(self, docker_server, test_audio_bonjour):
        result = transcribe_http(docker_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"

    @pytest.mark.parametrize("docker_server", [
        pytest.param(
            {"backend": "nemo", "mode": "task", "port": 0,
             "env_overrides": {"MODEL": "nvidia/parakeet-tdt-0.6b-v2",
                               "ARCHITECTURE": "rnnt_bpe",
                               "DEVICE": "cpu", "VAD": "false",
                               "SERVICES_BROKER": "redis://172.17.0.1:6379"}},
            id="cpu-vad_false-task",
        ),
    ], indirect=True)
    def test_transcription_task(self, docker_server, test_audio_bonjour, redis_server):
        from helpers.transcription_client import transcribe_celery
        result = transcribe_celery("bonjour.wav", language="fr", broker_url=redis_server)
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"

    @pytest.mark.gpu
    @pytest.mark.parametrize("docker_server", [
        pytest.param(
            {"backend": "nemo", "mode": "http", "port": 0, "use_gpu": True,
             "env_overrides": {"MODEL": "nvidia/parakeet-tdt-0.6b-v2",
                               "ARCHITECTURE": "rnnt_bpe",
                               "DEVICE": "cuda", "VAD": "false"}},
            id="cuda-vad_false-http",
        ),
    ], indirect=True)
    def test_transcription_gpu(self, docker_server, test_audio_bonjour):
        result = transcribe_http(docker_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"
