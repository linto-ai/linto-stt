import logging

logging.basicConfig(
    format="[%(asctime)s,%(msecs)03d %(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("__stt__")

try:
    import faster_whisper
    USE_CTRANSLATE2 = True
except ImportError as err:
    try:
        import whisper
    except:
        raise err
    USE_CTRANSLATE2 = False

try:
    import torch
    USE_TORCH = True
except ImportError:
    USE_TORCH = False

try:
    import torchaudio
    USE_TORCHAUDIO = True
except ImportError:
    USE_TORCHAUDIO = False
