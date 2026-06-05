import pytest

from helpers.env_setup import get_expected_regex
from helpers.transcription_client import transcribe_http


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------

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
                id=f"{'nodevice' if not device else device}-vad_{vad or 'none'}-{serving}",
            )


def _whisper_docker_configs(device, vads, dockerfile, model="tiny",
                            language="fr", servings=("http",)):
    for vad in vads:
        for serving in servings:
            env = {"MODEL": model, "LANGUAGE": language}
            if device:
                env["DEVICE"] = device
            if vad:
                env["VAD"] = vad
            yield pytest.param(
                {"engine": "whisper", "mode": serving, "port": 0,
                 "dockerfile": dockerfile, "env_overrides": env},
                id=f"{dockerfile.split('/')[-1]}-{'nodevice' if not device else device}-vad_{vad or 'none'}-{serving}",
            )


# ---------------------------------------------------------------------------
# UV tests - CPU
# ---------------------------------------------------------------------------

@pytest.mark.whisper
@pytest.mark.uv
class TestWhisperCPU:
    """Whisper engine on CPU via UV subprocess."""

    @pytest.mark.parametrize("uv_server",
        list(_whisper_configs("cpu", ["false", "auditok", "silero"])),
        indirect=True)
    def test_integration_cpu(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"


@pytest.mark.whisper
@pytest.mark.uv
@pytest.mark.gpu
class TestWhisperGPU:
    """Whisper engine on CUDA via UV subprocess."""

    @pytest.mark.parametrize("uv_server",
        list(_whisper_configs("cuda", [None])),
        indirect=True)
    def test_integration_cuda(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"


@pytest.mark.whisper
@pytest.mark.uv
class TestWhisperNoDevice:
    """Whisper engine with no explicit device."""

    @pytest.mark.parametrize("uv_server",
        list(_whisper_configs(None, [None])),
        indirect=True)
    def test_integration_nodevice(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"


# ---------------------------------------------------------------------------
# UV tests - Language variations
# ---------------------------------------------------------------------------

@pytest.mark.whisper
@pytest.mark.uv
class TestWhisperLanguages:
    """Whisper language-specific tests."""

    @pytest.mark.parametrize("uv_server", [
        pytest.param(
            {"engine": "whisper", "mode": "http", "port": 0,
             "env_overrides": {"MODEL": "tiny", "LANGUAGE": "*"}},
            id="nolanguage",
        ),
    ], indirect=True)
    def test_nolanguage(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"

    @pytest.mark.parametrize("uv_server", [
        pytest.param(
            {"engine": "whisper", "mode": "http", "port": 0,
             "env_overrides": {"MODEL": "tiny", "LANGUAGE": "FR-FR"}},
            id="languagecode-FR-FR",
        ),
    ], indirect=True)
    def test_languagecode(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"

    @pytest.mark.parametrize("uv_server", [
        pytest.param(
            {"engine": "whisper", "mode": "http", "port": 0,
             "env_overrides": {"MODEL": "tiny", "LANGUAGE": "ru"}},
            id="russian",
        ),
    ], indirect=True)
    def test_russian(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "ru").search(result), \
            f"Unexpected transcription: {result}"

    @pytest.mark.parametrize("uv_server", [
        pytest.param(
            {"engine": "whisper", "mode": "http", "port": 0,
             "env_overrides": {"MODEL": "tiny", "LANGUAGE": "ru"}},
            id="language-over-config",
        ),
    ], indirect=True)
    def test_language_over_config(self, uv_server, test_audio_bonjour):
        """Config says ru, but request overrides to fr."""
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour), language="fr")
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"


# ---------------------------------------------------------------------------
# UV tests - Hotwords
# ---------------------------------------------------------------------------

@pytest.mark.whisper
@pytest.mark.uv
class TestWhisperHotwords:
    """Whisper hotwords biasing (faster-whisper / CTranslate2 backend)."""

    @pytest.mark.parametrize("uv_server", [
        pytest.param(
            {"engine": "whisper", "mode": "http", "port": 0,
             "env_overrides": {
                 "MODEL": "tiny",
                 "LANGUAGE": "fr",
                 "DEVICE": "cpu",
                 "VAD": "false",
                 "HOTWORDS": "BonJour AuRevoir PourquoiPas",
             }},
            id="hotwords-bonjour",
        ),
    ], indirect=True)
    def test_hotwords_spelling(self, uv_server, test_audio_bonjour):
        """The HOTWORDS list should bias the spelling toward 'BonJour'."""
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert "BonJour" in result, \
            f"Expected hotword spelling 'BonJour' in transcription: {result!r}"


# ---------------------------------------------------------------------------
# UV tests - Model sizes
# ---------------------------------------------------------------------------

@pytest.mark.whisper
@pytest.mark.uv
@pytest.mark.slow
class TestWhisperModels:
    """Whisper model size tests."""

    @pytest.mark.parametrize("uv_server", [
        pytest.param(
            {"engine": "whisper", "mode": "http", "port": 0,
             "env_overrides": {"MODEL": "small", "LANGUAGE": "fr"}},
            id="model-small",
        ),
    ], indirect=True)
    def test_model_small(self, uv_server, test_audio_bonjour):
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"


# ---------------------------------------------------------------------------
# Docker tests
# ---------------------------------------------------------------------------

@pytest.mark.whisper
@pytest.mark.docker
class TestWhisperDockerCPU:
    """Whisper engine via Docker (CPU dockerfiles)."""

    @pytest.mark.parametrize("docker_server",
        list(_whisper_docker_configs("cpu", ["false", "auditok", "silero"],
             dockerfile="Dockerfile")),
        indirect=True)
    def test_integration_cpu(self, docker_server, test_audio_bonjour):
        result = transcribe_http(docker_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"


@pytest.mark.whisper
@pytest.mark.docker
@pytest.mark.gpu
class TestWhisperDockerGPU:
    """Whisper engine via Docker (CUDA dockerfiles)."""

    @pytest.mark.parametrize("docker_server",
        list(_whisper_docker_configs("cuda", [None],
             dockerfile="Dockerfile")),
        indirect=True)
    def test_integration_cuda(self, docker_server, test_audio_bonjour):
        result = transcribe_http(docker_server["url"], str(test_audio_bonjour))
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"


# ---------------------------------------------------------------------------
# Docker tests - Language variations (Celery)
# ---------------------------------------------------------------------------

@pytest.mark.whisper
@pytest.mark.docker
class TestWhisperDockerCelery:
    """Whisper Celery task tests via Docker."""

    @pytest.mark.parametrize("docker_server", [
        pytest.param(
            {"engine": "whisper", "mode": "task", "port": 0,
             "env_overrides": {"MODEL": "tiny", "LANGUAGE": "ru",
                               "SERVICES_BROKER": "redis://172.17.0.1:6379"}},
            id="russian-task",
        ),
    ], indirect=True)
    def test_russian_celery(self, docker_server, test_audio_bonjour, redis_server):
        from helpers.transcription_client import transcribe_celery
        result = transcribe_celery("bonjour.wav", language="ru", broker_url=redis_server)
        assert get_expected_regex(str(test_audio_bonjour), "ru").search(result), \
            f"Unexpected transcription: {result}"

    @pytest.mark.parametrize("docker_server", [
        pytest.param(
            {"engine": "whisper", "mode": "task", "port": 0,
             "env_overrides": {"MODEL": "tiny", "LANGUAGE": "ru",
                               "SERVICES_BROKER": "redis://172.17.0.1:6379"}},
            id="language-over-config-task",
        ),
    ], indirect=True)
    def test_language_over_config_celery(self, docker_server, test_audio_bonjour, redis_server):
        from helpers.transcription_client import transcribe_celery
        result = transcribe_celery("bonjour.wav", language="fr", broker_url=redis_server)
        assert get_expected_regex(str(test_audio_bonjour), "fr").search(result), \
            f"Unexpected transcription: {result}"
