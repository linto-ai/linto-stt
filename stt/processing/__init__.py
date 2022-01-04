import os
from time import time

from stt import logger
from stt.processing.model import prepare, loadModel
from stt.processing.decoding import decode
from stt.processing.utils import load_wave, formatAudio

# Model locations (should be mounted)
AM_PATH='/opt/models/AM'
LM_PATH='/opt/models/LM'
CONF_PATH='/opt/config'

# Prepare Model
logger.debug("Setting folders and configuration files")
try:
    prepare(AM_PATH, LM_PATH, CONF_PATH)
except Exception as e:
    logger.error("Could not prepare service: {}".format(str(e)))
    exit(-1)

# Load ASR models (acoustic model and decoding graph)
logger.info('Loading acoustic model and decoding graph ...')
start = time()
try:
    model = loadModel(AM_PATH, LM_PATH, os.path.join(CONF_PATH, "online.conf"))
except Exception as e:
    raise Exception("Failed to load transcription model: {}".format(str(e)))
    exit(-1)
logger.info('Acoustic model and decoding graph loaded. (t={}s)'.format(time() - start))
