import logging

logging.basicConfig(
    format="[%(asctime)s,%(msecs)03d %(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("__stt__")

try:
    import faster_whisper
    USE_CTRANSLATE2 = True
except ImportError:
    USE_CTRANSLATE2 = False

try:
    import torch, torchaudio
    USE_TORCH = True
except ImportError:
    USE_TORCH = False

# TODO: Get rid of that
if USE_TORCH:
    SHOULD_USE_GEVENT = torch.cuda.is_available()
    torch.set_num_threads(1)
else:
    SHOULD_USE_GEVENT = USE_CTRANSLATE2
