import os
import sys
from time import time

from linto_stt.engines.kaldi.stt import logger

from .decoding import decode
from .utils import load_audiofile, load_wave_buffer

from vosk import Model


__all__ = [
    "logger",
    "decode",
    "load_audiofile",
    "load_wave_buffer",
    "MODEL",
    "USE_GPU",
]

# Model locations (should be mounted)
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/model")

if os.environ.get("PUNCTUATION_MODEL"):
    from linto_stt.punctuation.recasepunc import load_recasepunc_model
    PUNCTUATION_MODEL = load_recasepunc_model()
else:
    PUNCTUATION_MODEL = None

# Load ASR models (acoustic model and decoding graph)
logger.info("Loading acoustic model and decoding graph ...")
start = time()
try:
    ASR_MODEL = Model(MODEL_PATH)
except Exception as err:
    raise Exception(
        "Failed to load transcription model: {}".format(str(err))) from err

logger.info(
    "Acoustic model and decoding graph loaded. (t={}s)".format(time() - start))


# Number of CPU threads
NUM_THREADS = int(os.environ.get("NUM_THREADS", os.cpu_count() or 1))
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)

MODEL = (ASR_MODEL, PUNCTUATION_MODEL)


def warmup():
    pass


# Not implemented yet in Kaldi
USE_GPU = False
