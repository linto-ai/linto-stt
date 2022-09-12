import asyncio
import os

from celery_app.celeryapp import celery
from stt import logger
from stt.processing import decode, model
from stt.processing.utils import load_wave


@celery.task(name="transcribe_task")
def transcribe_task(file_name: str, with_metadata: bool):
    """transcribe_task"""
    logger.info(f"Received transcription task for {file_name}")

    # Load wave
    file_path = os.path.join("/opt/audio", file_name)
    try:
        file_content = load_wave(file_path)
    except Exception as err:
        logger.error(f"Failed to load ressource: {repr(err)}")
        raise Exception(f"Could not open ressource {file_path}") from err

    # Decode
    try:
        result = decode(file_content, model, 16000, with_metadata)
    except Exception as err:
        logger.error(f"Failed to decode: {repr(err)}")
        raise Exception(f"Failed to decode {file_path}") from err

    return result
