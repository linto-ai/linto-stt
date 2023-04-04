import os
import logging

import torch
import whisper_timestamped as whisper

from stt import logger
from stt.processing.decoding import decode, get_language
from stt.processing.utils import load_wave_buffer, load_audiofile

from .load_model import load_whisper_model, load_alignment_model, get_alignment_model

__all__ = ["logger", "use_gpu", "decode", "model", "alignment_model",
           "load_audiofile", "load_wave_buffer"]

# Set informative log
logger.setLevel(logging.INFO)

# Set device
device = os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
try:
    device = torch.device(device)
except Exception as err:
    raise Exception("Failed to set device: {}".format(str(err))) from err
use_gpu = device.type == "cuda"
logger.info(f"Using device {device}")

# Check language
language = get_language()
available_languages = \
    list(whisper.tokenizer.LANGUAGES.keys()) + \
    [k.title() for k in whisper.tokenizer.TO_LANGUAGE_CODE.keys()] + \
    [None]
if language not in available_languages:
    raise ValueError(f"Language '{get_language()}' is not available. Available languages are: {available_languages}")
if isinstance(language, str):
    language = whisper.tokenizer.TO_LANGUAGE_CODE.get(language.lower(), language)
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
