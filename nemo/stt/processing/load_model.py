import os
import shutil
import subprocess
import sys
import time
import nemo.collections.asr as nemo_asr

from stt import logger


def load_nemo_model(model_type_or_file, device="cpu", download_root=None):
    start = time.time()

    logger.info(f"Loading Nemo model {model_type_or_file}...")

    default_cache_root = os.path.join(os.path.expanduser("~"), ".cache")
    if download_root is None:
        download_root = default_cache_root
    if model_type_or_file.endswith(".nemo"):
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(model_type_or_file, map_location=device)
    else:
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_type_or_file, map_location=device)
    
    logger.info(f"Whisper Nemo loaded. (t={time.time() - start:.2f}s)")

    return model


def check_torch_installed():
    try:
        import torch
    except ImportError:
        # Install transformers with torch
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers[torch]>=4.23"])

        # # Re-load ctranslate2
        # import importlib
        # import ctranslate2
        # importlib.reload(ctranslate2)
        # importlib.reload(ctranslate2.converters.transformers)

    # import torch

