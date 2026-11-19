import hashlib
import os
import shutil
import tempfile
from time import time

from filelock import FileLock

from linto_stt.engines.kaldi.stt import logger

from .decoding import decode
from .utils import load_audiofile, load_wave_buffer

from vosk import Model


__all__ = [
    "logger",
    "decode",
    "load_audiofile",
    "load_wave_buffer",
    "MODEL",
    "USE_GPU",
]

# Model locations (should be mounted): a Vosk model (MODEL_PATH), or a LinTO
# acoustic model (AM_PATH) and decoding graph (LM_PATH), converted here.
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/model")
AM_PATH = os.environ.get("AM_PATH", "/opt/AM")
LM_PATH = os.environ.get("LM_PATH", "/opt/LM")
# "vosk", "lin", or unset: "lin" when MODEL_PATH holds no Vosk model and both
# AM_PATH and LM_PATH exist, else "vosk".
MODEL_TYPE = (os.environ.get("MODEL_TYPE") or "").strip().lower()


CONVERSION_VERSION = 2  # 2: word_boundary.int generated when the graph lacks it


def _is_vosk_model(path):
    return os.path.isfile(os.path.join(path, "am", "final.mdl"))


def _resolve_model_path():
    """Folder of the Vosk model to load. LinTO models are converted into a Vosk
    model folder (links to the original files plus a few small configuration
    files), once per pair of AM/LM folders, in the temporary directory."""
    use_lin = MODEL_TYPE == "lin" or (
        MODEL_TYPE != "vosk"
        and not _is_vosk_model(MODEL_PATH)
        and os.path.isdir(AM_PATH)
        and os.path.isdir(LM_PATH)
    )
    if not use_lin:
        return MODEL_PATH
    from linto_stt.engines.kaldi.lin_to_vosk import lin_to_vosk_format

    # The format version is part of the key, so that a change in the conversion
    # never reuses a folder converted by an older version.
    key = hashlib.sha1(
        f"{CONVERSION_VERSION}|{os.path.realpath(AM_PATH)}|{os.path.realpath(LM_PATH)}".encode()
    ).hexdigest()[:12]
    target = os.path.join(tempfile.gettempdir(), f"linto-stt-kaldi-model-{key}")
    done = os.path.join(target, ".converted")
    # Several workers may start at once: one converts, the others wait.
    with FileLock(target + ".lock"):
        if not os.path.exists(done):
            if os.path.exists(target):  # left over from an interrupted conversion
                shutil.rmtree(target)
            logger.info(f"Converting LinTO model ({AM_PATH}, {LM_PATH}) to Vosk format in {target}")
            lin_to_vosk_format(AM_PATH, LM_PATH, target)
            open(done, "w").close()
    return target


MODEL_PATH = _resolve_model_path()

if os.environ.get("PUNCTUATION_MODEL"):
    from linto_stt.punctuation.recasepunc import load_recasepunc_model
    PUNCTUATION_MODEL = load_recasepunc_model()
else:
    PUNCTUATION_MODEL = None

# Load ASR models (acoustic model and decoding graph)
logger.info("Loading acoustic model and decoding graph ...")
start = time()
try:
    ASR_MODEL = Model(MODEL_PATH)
except Exception as err:
    raise Exception(
        "Failed to load transcription model: {}".format(str(err))) from err

logger.info(
    "Acoustic model and decoding graph loaded. (t={}s)".format(time() - start))


# Number of CPU threads
NUM_THREADS = int(os.environ.get("NUM_THREADS", os.cpu_count() or 1))
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)

MODEL = (ASR_MODEL, PUNCTUATION_MODEL)


def warmup():
    pass


# Not implemented yet in Kaldi
USE_GPU = False
