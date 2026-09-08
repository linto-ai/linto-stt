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
