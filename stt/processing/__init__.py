import os
import logging
from time import time

import torch
import whisper

from stt import logger
from stt.processing.decoding import decode, get_default_language
from stt.processing.utils import load_wave_buffer, load_audiofile

from .load_model import load_whisper_model, load_speechbrain_model

__all__ = ["logger", "decode", "model", "alignment_model", "load_audiofile", "load_wave_buffer"]

# Set device
device = os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
try:
    device = torch.device(device)
except Exception as err:
    raise Exception("Failed to set device: {}".format(str(err))) from err

# Check language
available_languages = [k for k,v in whisper.tokenizer.LANGUAGES.items()] + [None]
if get_default_language() not in available_languages:
    raise RuntimeError(f"Langaue {get_default_language()} is not available. Available languages are: {available_languages}")

# Load ASR model
model_type = os.environ.get("MODEL", "medium")
logger.info(f"Loading Whisper model {model_type} ({'local' if os.path.isfile(model_type) else 'remote'})...")
start = time()
try:
    model = load_whisper_model(model_type, device = device)
except Exception as err:
    raise Exception("Failed to load transcription model: {}".format(str(err))) from err
logger.info("Model loaded. (t={}s)".format(time() - start))

# Load alignment model
alignment_model_type = os.environ.get("ALIGNMENT_MODEL_TYPE", "/opt/linSTT_speechbrain_fr-FR_v1.0.0")
logger.info(f"Loading alignment model...")
start = time()
alignment_model = load_speechbrain_model(alignment_model_type, device = device, download_root = "/opt")
logger.info("Alignment Model loaded. (t={}s)".format(time() - start))
