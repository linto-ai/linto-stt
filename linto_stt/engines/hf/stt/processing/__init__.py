import logging
import os
import tempfile

from filelock import FileLock

from linto_stt.engines.hf.stt import (
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
    transcription = decode(audio, MODEL, True)  # also loads the primary aligner
    logger.info(f"Warmup result: {transcription}")
    # Load the fallback aligner now too, rather than on the first request that
    # needs it: that request would otherwise wait for the load (and the download
    # if the model is not cached), and the GPU memory would grow after startup.
    MODEL.check_fallback_aligner_loaded()


class LazyLoadedModel:
    """Load the ASR backend lazily on first use, so importing the engine (and
    reading its config) stays cheap. Exposes `.asr`, the backend selected by
    `BACKEND` ("transformers", "vllm" serve subprocess, or "vllm_offline"
    in-process), whose `.transcribe()` turns a segment into
    (text, detected_language).

    The companion forced aligners (for word-level timestamps) are loaded
    separately: by `warmup()` at startup (on GPU), otherwise on the
    first request that needs them. `aligner_model_name=None` disables alignment
    entirely; `fallback_aligner_model_name` names a second aligner used for the
    requests the primary one does not cover (see `aligner_for`)."""

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
        self._fallback_skip_logged = False
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
                        self.check_fallback_aligner_loaded()
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

    def check_fallback_aligner_loaded(self):
        """Load the fallback aligner, unless it can never be used: no fallback
        configured, alignment disabled (ALIGNER_MODEL=none), or a primary aligner
        that covers every language and duration (MMS). A fallback naming the same
        model as the primary reuses the primary's instance."""
        if self._fallback_aligner is not None or not self.fallback_aligner_model_name:
            return
        if not self.has_aligner:
            return
        self.check_aligner_loaded()
        primary = self._aligner
        if getattr(primary, "covers_everything", False):
            if not self._fallback_skip_logged:
                logger.info(
                    f"The {primary.name} aligner covers every language and duration: "
                    f"the fallback aligner ({self.fallback_aligner_model_name}) is not used "
                    f"and not loaded"
                )
                self._fallback_skip_logged = True
            return
        if self.fallback_aligner_model_name == self.aligner_model_name:
            self._fallback_aligner = primary
            return
        with FileLock(_lock_path(self.fallback_aligner_model_name)):
            if self._fallback_aligner is None:
                self._fallback_aligner = self._load_aligner(self.fallback_aligner_model_name)

    def _load_aligner(self, name):
        logger.info(
            f"Loading forced aligner {name} "
            f"({'local' if os.path.exists(name) else 'remote'})..."
        )
        return load_aligner(name, device=self.device, dtype=self.dtype)

    def aligner_for(self, language, duration=None):
        """The aligner to use for a request: the primary one when it covers the
        language and the audio duration (in seconds), else the fallback (loaded
        here if the warmup did not load it; a warning when it does not cover the
        request either). With no usable fallback the primary is always used (with
        a warning when the audio exceeds its documented limit)."""
        self.check_aligner_loaded()
        primary = self._aligner
        if primary.covers(language, duration):
            return primary
        what = (
            f"language {language!r}, {duration:.0f}s of audio" if duration is not None
            else f"language {language!r}"
        )
        self.check_fallback_aligner_loaded()
        fallback = self._fallback_aligner
        if fallback is not None and fallback is not primary:
            if fallback.covers(language, duration):
                logger.info(
                    f"Request ({what}) is not covered by the {primary.name} aligner: "
                    f"using the {fallback.name} fallback"
                )
            else:
                logger.warning(
                    f"Request ({what}) is covered by neither the {primary.name} aligner "
                    f"nor the {fallback.name} fallback: using the fallback anyway; "
                    f"timestamps may degrade. ALIGNER_FALLBACK_MODEL=mms covers any "
                    f"language and duration."
                )
            return fallback
        if duration is not None and primary.max_duration and duration > primary.max_duration:
            logger.warning(
                f"Audio ({duration:.0f}s) exceeds the {primary.name} aligner's documented "
                f"limit ({primary.max_duration:.0f}s); timestamps may degrade. Set "
                f"ALIGNER_FALLBACK_MODEL=mms to align long audio with the MMS aligner."
            )
        return primary

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

    # Warm up (force the load + one decode) on GPU. On CPU the HTTP server would
    # freeze during a synchronous load, so defer to first use. (No websocket
    # mode for this engine: main() refuses it at startup.)
    if USE_GPU:
        warmup()
except Exception as err:
    import traceback

    logger.error(traceback.format_exc())
    raise Exception(f"Failed to load transcription model: {err}") from err
