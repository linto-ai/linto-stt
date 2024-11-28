import asyncio
import os

from stt import logger
from stt.processing import MODEL, decode
from stt.processing.utils import load_audiofile

from celery_app.celeryapp import celery
from typing import Optional

@celery.task(name="transcribe_task")
def transcribe_task(file_name: str, with_metadata: bool, language: Optional[str] = None):
    """transcribe_task"""
    logger.info(f"Received transcription task for {file_name}")

    # Load wave
    print(language, type(language))
    file_path = os.path.join("/opt/audio", file_name)
    try:
        file_content = load_audiofile(file_path)
    except Exception as err:
        import traceback

        msg = f"{traceback.format_exc()}\nFailed to load ressource {file_path}"
        logger.error(msg)
        raise Exception(msg)  # from err

    # Decode
    try:
        result = decode(file_content, MODEL, with_metadata, language=language)
    except Exception as err:
        import traceback

        msg = f"{traceback.format_exc()}\nFailed to decode {file_path}"
        logger.error(msg)
        raise Exception(msg)  # from err

    return result
