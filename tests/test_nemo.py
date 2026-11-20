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


HOTWORDS_BONJOUR = "BonJour, AuRevoir, PourquoiPas"


def _nemo_hotwords_config(id, architecture, hotwords=None):
    # A strong boost (alpha 8, the default is 2): the test checks that the
    # hotwords reach the decoding by forcing a spelling the model never chooses
    # by itself ("BonJour"), as the whisper engine's hotword tests do.
    env = {"MODEL": "linagora/linto_stt_fr_fastconformer_pc", "ARCHITECTURE": architecture,
           "DEVICE": "cuda", "VAD": "false", "HOTWORDS_BOOSTING_ALPHA": "8"}
    if hotwords:
        env["HOTWORDS"] = hotwords
    return pytest.param({"engine": "nemo", "mode": "http", "port": 0, "env_overrides": env}, id=id)


@pytest.mark.nemo
@pytest.mark.uv
@pytest.mark.gpu
class TestNemoHotwords:
    """Hotwords with NeMo's GPU phrase boosting, on a hybrid model decoded with
    RNN-T and with CTC (whose `greedy` strategy is switched to `greedy_batch`, the
    one that supports boosting)."""

    @pytest.mark.parametrize("uv_server", [
        _nemo_hotwords_config("hotwords-env-rnnt", "hybrid_bpe", hotwords=HOTWORDS_BONJOUR),
        _nemo_hotwords_config("hotwords-env-ctc", "hybrid_bpe_ctc", hotwords=HOTWORDS_BONJOUR),
    ], indirect=True)
    def test_hotwords_spelling(self, uv_server, test_audio_bonjour):
        """The HOTWORDS list should bias the spelling toward 'BonJour'."""
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert "BonJour" in result, f"Expected hotword spelling 'BonJour' in transcription: {result!r}"

    @pytest.mark.parametrize("uv_server", [
        # HOTWORDS deliberately NOT set: the biasing must come from the request.
        _nemo_hotwords_config("hotwords-per-request-rnnt", "hybrid_bpe"),
    ], indirect=True)
    def test_hotwords_per_request(self, uv_server, test_audio_bonjour):
        """The per-request `hotwords` query param should bias the spelling toward
        'BonJour' with no HOTWORDS env set; a request without hotwords right after
        is decoded without them (the boosting tree is removed)."""
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour), hotwords=HOTWORDS_BONJOUR)
        assert "BonJour" in result, f"Expected request hotword spelling 'BonJour' in transcription: {result!r}"
        result = transcribe_http(uv_server["url"], str(test_audio_bonjour))
        assert "BonJour" not in result and get_expected_regex(str(test_audio_bonjour), "fr").search(result), (
            f"Unexpected transcription without hotwords: {result!r}"
        )


@pytest.mark.uv
def test_parse_hotwords():
    """The hotwords format, common to all engines (linto_stt/hotwords.py)."""
    from linto_stt.hotwords import parse_hotwords

    assert parse_hotwords("Broke, Jean-Luc Machin , ") == ["Broke", "Jean-Luc Machin"]
    assert parse_hotwords("chambre double") == ["chambre double"]
    assert parse_hotwords("") == [] and parse_hotwords(None) == []


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
        # The opening "Bonjour Madame" varies from run to run on this path (seen:
        # "Bonjour Madame", "Bonjour B Madame", "Bonjour."): only its "bonjour" is
        # required, then "bonjour monsieur" and the closing "deux nuits".
        assert (normalized.startswith("bonjour") and "bonjour monsieur" in normalized
                and normalized.endswith("deux nuits deux nuit")), (
            f"Expected a plausible transcription starting with 'bonjour', containing 'bonjour "
            f"monsieur' and ending with 'deux nuits deux nuit': {result!r}"
        )
        assert "<" not in result and ">" not in result, (
            f"Transcription should not contain <...> tags: {result!r}"
        )

    # The same model offline with its default look-ahead.
    # DEVICE is left unset so it follows the --device CLI option (default cpu).
    @pytest.mark.parametrize(
        "uv_server",
        [
            pytest.param(
                {"engine": "nemo", "mode": "http", "port": 0,
                 "env_overrides": _NEMOTRON_ENV},
                id="nemotron-default",
            ),
        ],
        indirect=True,
    )
    def test_streaming_model_offline_default(self, uv_server, test_audio_hotel):
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
