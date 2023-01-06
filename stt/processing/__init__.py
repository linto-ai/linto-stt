import os
import logging

import torch
import whisper

from stt import logger
from stt.processing.decoding import decode, get_language
from stt.processing.utils import load_wave_buffer, load_audiofile

from .load_model import load_whisper_model, load_alignment_model, get_alignment_model, get_model_type

__all__ = ["logger", "decode", "model", "alignment_model",
           "load_audiofile", "load_wave_buffer"]

# Set informative log
logger.setLevel(logging.INFO)

# Set device
device = os.environ.get(
    "DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
try:
    device = torch.device(device)
except Exception as err:
    raise Exception("Failed to set device: {}".format(str(err))) from err
logger.info(f"Using device {device}")

# Check language
language = get_language()
available_languages = [
    k for k, v in whisper.tokenizer.LANGUAGES.items()] + [None]
if language not in available_languages:
    raise RuntimeError(
        f"Language {get_language()} is not available. Available languages are: {available_languages}")
logger.info(f"Using language {language}")

# Load ASR model
model_type = os.environ.get("MODEL", "medium")
logger.info(f"Loading Whisper model {model_type} ({'local' if os.path.exists(model_type) else 'remote'})...")
try:
    model = load_whisper_model(model_type, device=device)
except Exception as err:
    raise Exception(
        "Failed to load transcription model: {}".format(str(err))) from err

# Load alignment model
alignment_model_name = get_alignment_model(language)
if alignment_model_name:
    logger.info(
        f"Loading alignment model {alignment_model_name} ({'local' if os.path.exists(alignment_model_name) else 'remote'})...")
    alignment_model = load_alignment_model(
        alignment_model_name, device=device, download_root="/opt")
else:
    logger.info("No alignment model preloaded")
    alignment_model = {}  # Alignement model(s) will be loaded on the fly
