import os
import logging
from lockfile import FileLock

from stt import logger, USE_CTRANSLATE2
from .decoding import decode
from .utils import get_device, get_language, load_wave_buffer, load_audiofile

from .load_model import load_whisper_model
from .alignment_model import load_alignment_model, get_alignment_model

__all__ = ["logger", "decode", "model", "alignment_model",
           "load_audiofile", "load_wave_buffer"]

class LazyLoadedModel:

    def __init__(self, model_type, device):
        self.model_type = model_type
        self.device = device
        self._model = None

    def __getattr__(self, name):
        if self._model is None:
            lockfile = os.path.basename(self.model_type)
            with FileLock(lockfile):
                self._model = load_whisper_model(self.model_type, device=self.device)
        return getattr(self._model, name)
    
# Set informative log
logger.setLevel(logging.INFO)

# Set device
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # GPU in the right order
device, USE_GPU = get_device()
logger.info(f"Using device {device}")

# Check language
language = get_language()
logger.info(f"Using language {language}")

# Load ASR model
model_type = os.environ.get("MODEL", "medium")
logger.info(f"Loading Whisper model {model_type} ({'local' if os.path.exists(model_type) else 'remote'})...")
try:
    MODEL = LazyLoadedModel(model_type, device=device)
    # model = load_whisper_model(model_type, device=device)
except Exception as err:
    raise Exception(
        "Failed to load transcription model: {}".format(str(err))) from err

# Load alignment model (if any)
ALIGNMENT_MODEL = get_alignment_model(os.environ.get("ALIGNMENT_MODEL"), language)
if ALIGNMENT_MODEL:
    logger.info(
        f"Loading alignment model {ALIGNMENT_MODEL} ({'local' if os.path.exists(alignment_model) else 'remote'})...")
    ALIGNMENT_MODEL = load_alignment_model(ALIGNMENT_MODEL, device=device, download_root="/opt")
elif ALIGNMENT_MODEL is None:
    logger.info("Alignment will be done using Whisper cross-attention weights")
else:
    logger.info("No alignment model preloaded. It will be loaded on the fly depending on the detected language.")
    ALIGNMENT_MODEL = {}  # Alignement model(s) will be loaded on the fly
