import os
import logging

from stt import logger
from .decoding import decode, get_language
from .utils import get_device, LANGUAGES, load_wave_buffer, load_audiofile

from .load_model import load_whisper_model
from .alignment_model import load_alignment_model, get_alignment_model

__all__ = ["logger", "decode", "model", "alignment_model",
           "load_audiofile", "load_wave_buffer"]

# Set informative log
logger.setLevel(logging.INFO)

# Set device
device, use_gpu = get_device()
logger.info(f"Using device {device}")

# Check language
language = get_language()
available_languages = \
    list(LANGUAGES.keys()) + \
    [k.lower() for k in LANGUAGES.values()] + \
    [None]
if language not in available_languages:
    raise ValueError(f"Language '{get_language()}' is not available. Available languages are: {available_languages}")
if isinstance(language, str) and language not in LANGUAGES:
    language = {v: k for k, v in LANGUAGES.items()}[language.lower()]
logger.info(f"Using language {language}")

# Load ASR model
model_type = os.environ.get("MODEL", "medium")
logger.info(f"Loading Whisper model {model_type} ({'local' if os.path.exists(model_type) else 'remote'})...")
try:
    model = load_whisper_model(model_type, device=device)
except Exception as err:
    raise Exception(
        "Failed to load transcription model: {}".format(str(err))) from err

# Load alignment model (if any)
alignment_model = get_alignment_model(os.environ.get("ALIGNMENT_MODEL"), language)
if alignment_model:
    logger.info(
        f"Loading alignment model {alignment_model} ({'local' if os.path.exists(alignment_model) else 'remote'})...")
    alignment_model = load_alignment_model(alignment_model, device=device, download_root="/opt")
elif alignment_model is None:
    logger.info("Alignment will be done using Whisper cross-attention weights")
else:
    logger.info("No alignment model preloaded. It will be loaded on the fly depending on the detected language.")
    alignment_model = {}  # Alignement model(s) will be loaded on the fly
