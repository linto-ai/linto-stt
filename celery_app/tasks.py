import asyncio
import os

from celery_app.celeryapp import celery
from stt import logger
from stt.processing import decode, MODEL
from stt.processing.utils import load_audiofile


@celery.task(name="transcribe_task")
def transcribe_task(file_name: str, with_metadata: bool):
    """transcribe_task"""
    logger.info(f"Received transcription task for {file_name}")

    # Load wave
    file_path = os.path.join("/opt/audio", file_name)
    try:
        file_content = load_audiofile(file_path)
    except Exception as err:
        import traceback
        msg = f"{traceback.format_exc()}\nFailed to load ressource {file_path}"
        logger.error(msg)
        raise Exception(msg) # from err

    # Decode
    try:
        result = decode(file_content, MODEL, with_metadata)
    except Exception as err:
        import traceback
        msg = f"{traceback.format_exc()}\nFailed to decode {file_path}"
        logger.error(msg)
        raise Exception(msg) # from err

    return result
