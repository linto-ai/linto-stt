import logging
import os

logging.basicConfig(
    format="[%(asctime)s,%(msecs)03d %(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("__stt__")

# The following is to have GPU in the right order (as nvidia-smi show them)
# It is important to set that before loading ctranslate2
# see https://github.com/guillaumekln/faster-whisper/issues/150
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU in the right order
