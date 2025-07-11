import logging
import os

from lockfile import FileLock
from stt import USE_CTRANSLATE2, USE_ACCURATE, VAD, logger, set_num_threads, NUM_THREADS

from .alignment_model import get_alignment_model, load_alignment_model
from .decoding import decode
from .load_model import load_whisper_model
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
    model.check_loaded()
    audio_data = load_audiofile("test/bonjour.wav")
    transcription = decode(audio_data, MODEL, False)
    logger.info(f"Warmup result: {transcription}")
    
class LazyLoadedModel:
    def __init__(self, model_type, device, num_threads):
        self.model_type = model_type
        self.device = device
        self.num_threads = num_threads
        self._model = None
        self.has_set_num_threads = False

    def check_loaded(self):
        if self._model is None:
            lockfile = os.path.basename(self.model_type)
            with FileLock(lockfile):
                self._model = load_whisper_model(self.model_type, device=self.device)

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


# Set informative log
logger.setLevel(logging.INFO)

# Set device
device, USE_GPU = get_device()
logger.info(f"Using device {device}")

# Check language
language = get_language()
logger.info(f"Using language {language}")

logger.info(f"VAD={VAD}")
logger.info(f"USE_CTRANSLATE2={USE_CTRANSLATE2}")
logger.info(f"USE_ACCURATE={USE_ACCURATE}")

# Load alignment model (if any)
alignment_model = get_alignment_model(os.environ.get("alignment_model"), language)
if alignment_model:
    logger.info(
        f"Loading alignment model {alignment_model} ({'local' if os.path.exists(alignment_model) else 'remote'})..."
    )
    alignment_model = load_alignment_model(alignment_model, device=device, download_root="/opt")
elif alignment_model is None:
    logger.info("Alignment will be done using Whisper cross-attention weights")
else:
    logger.info(
        "No alignment model preloaded. It will be loaded on the fly depending on the detected language."
    )
    alignment_model = {}  # Alignement model(s) will be loaded on the fly


# Load ASR model
model_type = os.environ.get("MODEL", "medium")
logger.info(
    f"Loading Whisper model {model_type} ({'local' if os.path.exists(model_type) else 'remote'})..."
)
try:
    model = LazyLoadedModel(model_type, device=device, num_threads=NUM_THREADS)
    MODEL = (model, alignment_model)
    if USE_GPU:
        warmup()
except Exception as err:
    raise Exception("Failed to load transcription model: {}".format(str(err))) from err
