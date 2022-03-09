import os
import asyncio

from stt import logger
from stt.processing import model
from celery_app.celeryapp import celery
from stt.processing.utils import load_wave
from stt.processing import decode

@celery.task(name="transcribe_task")
def transcribe_task(file_name: str, with_metadata: bool):
    """ transcribe_task """
    logger.info("Received transcription task for {}".format(file_name))

    # Load wave
    file_path = os.path.join("/opt/audio", file_name)
    try:
        file_content = load_wave(file_path)
    except Exception as e:
        logger.error("Failed to load ressource: {}".format(e))
        raise Exception("Could not open ressource {}".format(file_path))

    # Decode
    try:
        result = decode(file_content, model, 16000, with_metadata)
    except Exception as e:
        logger.error("Failed to decode: {}".format(e))
        raise Exception("Failed to decode {}".format(file_path))

    return result