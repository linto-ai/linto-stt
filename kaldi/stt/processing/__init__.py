import os
import sys
from time import time

from vosk import Model

from stt import logger
from stt.processing.decoding import decode
from stt.processing.utils import load_wave_buffer, load_audiofile

__all__ = [
    "logger",
    "decode",
    "load_audiofile", "load_wave_buffer",
    "MODEL",
    "USE_GPU",
]

# Model locations (should be mounted)
MODEL_PATH = "/opt/model"

# Load ASR models (acoustic model and decoding graph)
logger.info("Loading acoustic model and decoding graph ...")
start = time()
try:
    MODEL = Model(MODEL_PATH)
except Exception as err:
    raise Exception("Failed to load transcription model: {}".format(str(err))) from err
    sys.exit(-1)
logger.info("Acoustic model and decoding graph loaded. (t={}s)".format(time() - start))

# Not implemented yet in Kaldi
USE_GPU = False