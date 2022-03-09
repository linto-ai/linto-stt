import os
from time import time

from vosk import Model

from stt import logger
from stt.processing.decoding import decode
from stt.processing.utils import load_wave, formatAudio
#from stt.processing.model import loadModel

__all__ = ["model", "logger", "decode", "load_wave", "formatAudio"]

# Model locations (should be mounted)
MODEL_PATH="/opt/model"

# Load ASR models (acoustic model and decoding graph)
logger.info('Loading acoustic model and decoding graph ...')
start = time()
try:
    model = Model(MODEL_PATH)
except Exception as e:
    raise Exception("Failed to load transcription model: {}".format(str(e)))
    exit(-1)
logger.info('Acoustic model and decoding graph loaded. (t={}s)'.format(time() - start))
