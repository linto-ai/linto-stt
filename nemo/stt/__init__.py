import logging
import torch
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

def set_num_threads(n):
    torch.set_num_threads(n)

NUM_THREADS = os.environ.get("NUM_THREADS", os.environ.get("OMP_NUM_THREADS"))
if NUM_THREADS is None:
    NUM_THREADS = torch.get_num_threads()
if NUM_THREADS is not None:
    NUM_THREADS = int(NUM_THREADS)

set_num_threads(1)