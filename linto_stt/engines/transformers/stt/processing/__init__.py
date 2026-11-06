import logging
import os

from filelock import FileLock

from linto_stt.engines.transformers.stt import (
    ALIGNER_MODEL,
    ATTN_IMPLEMENTATION,
    BACKEND,
    DTYPE,
    NUM_THREADS,
    logger,
    set_num_threads,
)

from .decoding import decode
from .load_model import load_aligner_model, load_asr_backend
from .utils import get_device, get_language, load_audiofile, load_wave_buffer

__all__ = [
    "logger",
    "decode",
    "load_audiofile",
    "load_wave_buffer",
    "MODEL",
    "USE_GPU",
]


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

    The companion forced-alignment model (for word-level timestamps) is loaded
    separately and even more lazily: only on the first request that asks for word
    timestamps. `aligner_model_name=None` disables alignment entirely."""

    def __init__(self, model_name, device, dtype, attn_implementation, num_threads,
                 aligner_model_name=None):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.num_threads = num_threads
        self.aligner_model_name = aligner_model_name
        self._asr = None
        self._aligner_processor = None
        self._aligner_model = None
        self.has_set_num_threads = False

    def check_loaded(self):
        if self._asr is None:
            # OS-level flock (released by the kernel if the holder dies), so a
            # crashed load cannot leave a stale lock that deadlocks later loads.
            lockfile = os.path.basename(self.model_name) + ".lock"
            with FileLock(lockfile):
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
        if self._aligner_model is None:
            lockfile = os.path.basename(self.aligner_model_name) + ".lock"
            with FileLock(lockfile):
                if self._aligner_model is None:
                    logger.info(
                        f"Loading forced-alignment model {self.aligner_model_name} "
                        f"({'local' if os.path.exists(self.aligner_model_name) else 'remote'})..."
                    )
                    self._aligner_processor, self._aligner_model = load_aligner_model(
                        self.aligner_model_name,
                        device=self.device,
                        dtype=self.dtype,
                    )

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

    @property
    def aligner_processor(self):
        self.check_aligner_loaded()
        return self._aligner_processor

    @property
    def aligner_model(self):
        self.check_aligner_loaded()
        return self._aligner_model


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
    )

    # Warm up (force the load + one decode) on GPU or websocket mode. On http+cpu
    # the server would freeze during a synchronous load, so defer to first use.
    if USE_GPU or os.environ.get("SERVICE_MODE", "http") == "websocket":
        warmup()
except Exception as err:
    import traceback

    logger.error(traceback.format_exc())
    raise Exception(f"Failed to load transcription model: {err}") from err
