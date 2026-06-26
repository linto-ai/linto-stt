import logging
import os

from filelock import FileLock
from linto_stt.engines.nemo.stt import logger, set_num_threads, NUM_THREADS, VAD
from linto_stt.punctuation.recasepunc import load_recasepunc_model

from .decoding import decode
from .load_model import load_nemo_model
from .utils import get_device, load_audiofile, load_wave_buffer, get_language, get_decoding_method

__all__ = [
    "logger",
    "decode",
    "load_audiofile",
    "load_wave_buffer",
    "MODEL",
    "USE_GPU",
]


def warmup():
    model.check_loaded()
    audio = load_audiofile("tests/bonjour.wav")
    transcription = decode(audio, MODEL, True)
    logger.info(f"Warmup result: {transcription}")


class LazyLoadedModel:
    def __init__(self, model_type, device, num_threads, decoding_strategy_if_hybrid="ctc"):
        self.model_type = model_type
        self.decoding_strategy_if_hybrid = decoding_strategy_if_hybrid
        self.device = device
        self.num_threads = num_threads
        self._model = None
        self.has_set_num_threads = False

    def check_loaded(self):
        if self._model is None:
            # filelock uses an OS-level flock that the kernel releases when the
            # holder dies, so a killed/crashed load can't leave a stale lock that
            # deadlocks every later load (the old `lockfile` package did exactly
            # that). Lock-file name kept as "<model>.lock" for continuity.
            lockfile = os.path.basename(self.model_type) + ".lock"
            with FileLock(lockfile):
                self._model = load_nemo_model(self.model_type, device=self.device,
                                              decoding_strategy_if_hybrid=self.decoding_strategy_if_hybrid)

    def check_num_threads(self):
        if not self.has_set_num_threads and self.num_threads:
            set_num_threads(self.num_threads)
            self.has_set_num_threads = True

    def __getattr__(self, name):
        self.check_loaded()
        self.check_num_threads()
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        self.check_loaded()
        self.check_num_threads()
        return self._model(*args, **kwargs)

    @property
    def __class__(self):
        return self._model.__class__

# Set informative log
logger.setLevel(logging.INFO)

# Set device
device, USE_GPU = get_device()
logger.info(f"Using device {device}")

# Load ASR model
model_type = os.environ.get("MODEL", "nvidia/parakeet-tdt-0.6b-v2")
decoding_strategy_if_hybrid = get_decoding_method(
    os.environ.get("ARCHITECTURE", "rnnt_bpe"))

# Check language
language = get_language()
logger.info(f"Using language {language}")

logger.info(f"VAD={VAD}")

logger.info(
    f"Loading Nemo model {model_type} ({'local' if os.path.exists(model_type) else 'remote'})..."
)
try:
    model = LazyLoadedModel(model_type, device=device,
                            num_threads=NUM_THREADS, decoding_strategy_if_hybrid=decoding_strategy_if_hybrid)

    PUNCTUATION_MODEL = load_recasepunc_model()
    MODEL = (model, PUNCTUATION_MODEL)

    # if warmup when http+cpu : server freeze
    # if no warmup when websocket+cpu : model loaded at first connexion
    if USE_GPU or os.environ.get("SERVICE_MODE", "http") == "websocket":
        warmup()
except Exception as err:
    import traceback
    logger.error(traceback.format_exc())
    raise Exception(
        "Failed to load transcription model: {}".format(str(err))) from err
