import os
import sys
from time import time

from stt import logger
from stt.processing.decoding import decode
from stt.processing.utils import load_audiofile, load_wave_buffer
from punctuation.recasepunc import load_recasepunc_model
from vosk import Model
import torch


__all__ = [
    "logger",
    "decode",
    "load_audiofile",
    "load_wave_buffer",
    "MODEL",
    "USE_GPU",
]

# Model locations (should be mounted)
MODEL_PATH = "/opt/model"

PUNCTUATION_MODEL = load_recasepunc_model()

# Load ASR models (acoustic model and decoding graph)
logger.info("Loading acoustic model and decoding graph ...")
start = time()
try:
    ASR_MODEL = Model(MODEL_PATH)
except Exception as err:
    raise Exception("Failed to load transcription model: {}".format(str(err))) from err

logger.info("Acoustic model and decoding graph loaded. (t={}s)".format(time() - start))


# Number of CPU threads
NUM_THREADS = os.environ.get("NUM_THREADS", torch.get_num_threads())
NUM_THREADS = int(NUM_THREADS)
# This set the number of threads for sklearn
os.environ["OMP_NUM_THREADS"] = str(
    NUM_THREADS
)  # This must be done BEFORE importing packages (sklearn, etc.)
# For Torch, we will set it afterward, because setting that before loading the model can hang the process (see https://github.com/pytorch/pytorch/issues/58962)
torch.set_num_threads(1)

MODEL = (ASR_MODEL, PUNCTUATION_MODEL)

def warmup():
    pass

# Not implemented yet in Kaldi
USE_GPU = False
