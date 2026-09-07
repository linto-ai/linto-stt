import os

import pytest

from helpers.env_setup import get_expected_regex
from helpers.transcription_client import transcribe_http


def _skip_without_kaldi_paths(request):
    """Skip if Kaldi model paths are not provided or don't exist."""
    am = request.config.getoption("--kaldi-am-path")
    lm = request.config.getoption("--kaldi-lm-path")
    if not am or not lm:
        pytest.skip("--kaldi-am-path and --kaldi-lm-path required for Kaldi tests")
    if not os.path.exists(am):
        pytest.skip(f"Kaldi AM path not found: {am}")
    if not os.path.exists(lm):
        pytest.skip(f"Kaldi LM path not found: {lm}")


# ---------------------------------------------------------------------------
# UV tests
# ---------------------------------------------------------------------------


@pytest.mark.kaldi
@pytest.mark.uv
class TestKaldiUV:
    """Kaldi engine via UV subprocess. Requires --kaldi-am-path and --kaldi-lm-path."""

    @pytest.fixture(autouse=True)
    def _check_paths(self, request):
        _skip_without_kaldi_paths(request)

    @pytest.mark.parametrize(
        "uv_server",
        [
            pytest.param(
                {"engine": "kaldi", "mode": "http", "port": 0, "env_overrides": {}},
                id="http",
            ),
        ],
        indirect=True,
    )
    def test_transcription_http(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )


# ---------------------------------------------------------------------------
# Docker tests
# ---------------------------------------------------------------------------


@pytest.mark.kaldi
@pytest.mark.docker
class TestKaldiDocker:
    """Kaldi engine via Docker. Requires --kaldi-am-path and --kaldi-lm-path.
    Volumes are auto-injected by the docker_server fixture in conftest.py.
    """

    @pytest.fixture(autouse=True)
    def _check_paths(self, request):
        _skip_without_kaldi_paths(request)

    @pytest.mark.parametrize(
        "docker_server",
        [
            pytest.param(
                {"engine": "kaldi", "mode": "http", "port": 0, "env_overrides": {}},
                id="http",
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
                    "engine": "kaldi",
                    "mode": "task",
                    "port": 0,
                    "env_overrides": {"SERVICES_BROKER": "redis://172.17.0.1:6379"},
                },
                id="task",
            ),
        ],
        indirect=True,
    )
    def test_transcription_task(self, docker_server, test_audio_bonjour, redis_server):
        from helpers.transcription_client import transcribe_celery

        result = transcribe_celery(
            "bonjour.wav", language="fr", broker_url=redis_server
        )
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription: {result}"
        )
