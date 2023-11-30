import logging
import os

logging.basicConfig(
    format="[%(asctime)s,%(msecs)03d %(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("__stt__")

# The following is to have GPU in the right order (as nvidia-smi show them)
# It is important to set that before loading ctranslate2
# see https://github.com/guillaumekln/faster-whisper/issues/150
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU in the right order

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
