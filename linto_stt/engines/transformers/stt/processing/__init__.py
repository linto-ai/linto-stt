import logging
import os
import tempfile

from filelock import FileLock

from linto_stt.engines.transformers.stt import (
    ALIGNER_FALLBACK_MODEL,
    ALIGNER_MODEL,
    ATTN_IMPLEMENTATION,
    BACKEND,
    DTYPE,
    NUM_THREADS,
    logger,
    set_num_threads,
)

from .decoding import decode
from .aligners import load_aligner
from .load_model import load_asr_backend
from .utils import get_device, get_language, load_audiofile, load_wave_buffer

__all__ = [
    "logger",
    "decode",
    "load_audiofile",
    "load_wave_buffer",
    "MODEL",
    "USE_GPU",
]


def _lock_path(model_name):
    """Path to the model-load lock. Put it in the temp dir (not the current
    working directory) so loading a model never drops a `.lock` file in the repo,
    while staying a stable per-model path shared across workers on the machine."""
    return os.path.join(tempfile.gettempdir(), os.path.basename(model_name) + ".lock")


def warmup():
    MODEL.check_loaded()
    audio = load_audiofile("tests/bonjour.wav")
    transcription = decode(audio, MODEL, True)
    logger.info(f"Warmup result: {transcription}")


class LazyLoadedModel:
    """Load the ASR backend lazily on first use, so importing the engine (and
    reading its config) stays cheap. Exposes `.asr`, the backend selected by
    `BACKEND` ("transformers", "vllm" serve subprocess, or "vllm_offline"
    in-process), whose `.transcribe()` turns a segment into
    (text, detected_language).

    The companion forced aligner(s) (for word-level timestamps) are loaded
    separately and even more lazily: only on the first request that asks for word
    timestamps. `aligner_model_name=None` disables alignment entirely;
    `fallback_aligner_model_name` names a second aligner used for the languages
    the primary one does not cover (see `aligner_for`)."""

    def __init__(self, model_name, device, dtype, attn_implementation, num_threads,
                 aligner_model_name=None, fallback_aligner_model_name=None):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.num_threads = num_threads
        self.aligner_model_name = aligner_model_name
        self.fallback_aligner_model_name = fallback_aligner_model_name
        self._asr = None
        self._aligner = None
        self._fallback_aligner = None
        self.has_set_num_threads = False

    def check_loaded(self):
        if self._asr is None:
            # OS-level flock (released by the kernel if the holder dies), so a
            # crashed load cannot leave a stale lock that deadlocks later loads.
            with FileLock(_lock_path(self.model_name)):
                if self._asr is None:
                    # Constructing the in-process vLLM engine patches transformers'
                    # Qwen3ASR config, after which the transformers forced aligner
                    # can no longer be *loaded*. So load the aligner first (while
                    # the registry is clean); the already-loaded model then runs
                    # fine. The "vllm" serve backend runs vLLM in a subprocess and
                    # does not patch this process, so only "vllm_offline" needs it.
                    if BACKEND == "vllm_offline" and self.aligner_model_name:
                        self.check_aligner_loaded()
                    self._asr = load_asr_backend(
                        BACKEND,
                        self.model_name,
                        device=self.device,
                        dtype=self.dtype,
                        attn_implementation=self.attn_implementation,
                    )

    def check_aligner_loaded(self):
        if self._aligner is None:
            with FileLock(_lock_path(self.aligner_model_name)):
                if self._aligner is None:
                    self._aligner = self._load_aligner(self.aligner_model_name)
                    if self.fallback_aligner_model_name:
                        self._fallback_aligner = self._load_aligner(
                            self.fallback_aligner_model_name
                        )

    def _load_aligner(self, name):
        logger.info(
            f"Loading forced aligner {name} "
            f"({'local' if os.path.exists(name) else 'remote'})..."
        )
        return load_aligner(name, device=self.device, dtype=self.dtype)

    def aligner_for(self, language):
        """The aligner to use for `language`: the primary one when it covers the
        language (or when no fallback is configured), else the fallback."""
        self.check_aligner_loaded()
        if self._fallback_aligner is not None and not self._aligner.covers(language):
            logger.info(
                f"Language {language!r} is not covered by the {self._aligner.name} "
                f"aligner: using the {self._fallback_aligner.name} fallback"
            )
            return self._fallback_aligner
        return self._aligner

    def check_num_threads(self):
        if not self.has_set_num_threads and self.num_threads:
            set_num_threads(self.num_threads)
            self.has_set_num_threads = True

    @property
    def asr(self):
        self.check_loaded()
        self.check_num_threads()
        return self._asr

    @property
    def has_aligner(self):
        return self.aligner_model_name is not None


logger.setLevel(logging.INFO)

device, USE_GPU = get_device()
logger.info(f"Using device {device}")

model_name = os.environ.get("MODEL", "Qwen/Qwen3-ASR-1.7B-hf")

language = get_language()
logger.info(
    f"Using language {language}" if language else "Using automatic language detection"
)

logger.info(
    f"Loading model {model_name} "
    f"({'local' if os.path.exists(model_name) else 'remote'})..."
)
try:
    MODEL = LazyLoadedModel(
        model_name,
        device=device,
        dtype=DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        num_threads=NUM_THREADS,
        aligner_model_name=ALIGNER_MODEL,
        fallback_aligner_model_name=ALIGNER_FALLBACK_MODEL,
    )

    # Warm up (force the load + one decode) on GPU or websocket mode. On http+cpu
    # the server would freeze during a synchronous load, so defer to first use.
    if USE_GPU or os.environ.get("SERVICE_MODE", "http") == "websocket":
        warmup()
except Exception as err:
    import traceback

    logger.error(traceback.format_exc())
    raise Exception(f"Failed to load transcription model: {err}") from err
