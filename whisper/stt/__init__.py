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

if os.environ.get("VAD","auditok").lower() in ["true", "1"]:
    VAD = "auditok"
elif os.environ.get("VAD","auditok").lower() in ["false", "0"]:
    VAD = False
else:
    VAD = os.environ.get("VAD","auditok")

VAD_DILATATION = float(os.environ.get("VAD_DILATATION", 0.5))
VAD_MIN_SPEECH_DURATION = float(os.environ.get("VAD_MIN_SPEECH_DURATION", 0.1))
VAD_MIN_SILENCE_DURATION = float(os.environ.get("VAD_MAX_SILENCE_DURATION", 0.1))

NUM_THREADS = os.environ.get("NUM_THREADS", os.environ.get("OMP_NUM_THREADS"))
NUM_THREADS = int(NUM_THREADS)

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

if USE_CTRANSLATE2:
    def set_num_threads(n):
        # os.environ["OMP_NUM_THREADS"] = str(n)
        pass
else:
    import torch
    DEFAULT_NUM_THREADS = torch.get_num_threads()
    def set_num_threads(n):
        torch.set_num_threads(n)

# Number of CPU threads
if NUM_THREADS is None:
    NUM_THREADS = DEFAULT_NUM_THREADS
if NUM_THREADS is not None:
    NUM_THREADS = int(NUM_THREADS)
# For Torch, we will set it afterward, because setting that before loading the model can hang the process (see https://github.com/pytorch/pytorch/issues/58962)
set_num_threads(1)
