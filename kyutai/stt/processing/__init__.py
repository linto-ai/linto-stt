import logging
import os

logging.basicConfig(
    format="[%(asctime)s %(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("__stt__")

KYUTAI_URL = os.environ.get("KYUTAI_URL", "ws://localhost:8080")
KYUTAI_API_KEY = os.environ.get("KYUTAI_API_KEY", "public_token")

USE_GPU = False
MODEL = None

from .streaming import wssDecode
from .utils import load_wave_buffer

__all__ = [
    "logger",
    "decode",
    "load_wave_buffer",
    "wssDecode",
    "MODEL",
    "USE_GPU",
]


def warmup():
    logger.info("Kyutai backend uses external server, no warmup needed")


def decode(audio, model, with_word_timestamps, language=None):
    from .client import decode_audio

    return decode_audio(audio)