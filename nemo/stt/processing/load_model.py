import os
import shutil
import subprocess
import sys
import time
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.asr_confidence_utils import (
    ConfidenceConfig,
    ConfidenceConstants,
    ConfidenceMethodConfig,
    ConfidenceMethodConstants,
)
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig

from stt import logger
import logging
logging.basicConfig(level = logging.INFO)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

def load_nemo_model(model_type_or_file, model_class: nemo_asr.models.EncDecHybridRNNTCTCModel, device="cpu", download_root=None, decoding_strategy_if_hybrid="ctc"):
    start = time.time()
    logger.info(f"Loading Nemo model {model_type_or_file}...")
    default_cache_root = os.path.join(os.path.expanduser("~"), ".cache")
    if download_root is None:
        download_root = default_cache_root
    if model_type_or_file.endswith(".nemo"):
        model = model_class.restore_from(model_type_or_file, map_location=device)
    else:
        model = model_class.from_pretrained(model_type_or_file, map_location=device)
        # model = nemo_asr.models.ASRModel.from_pretrained(model_type_or_file, map_location=device)     # todo: make architecture optional if use remote by using this line
    logger.info(f"Nemo model loaded. (t={time.time() - start:.2f}s)")
    if isinstance(model, nemo_asr.models.EncDecRNNTModel):
        if isinstance(model, nemo_asr.models.EncDecHybridRNNTCTCModel):
            if decoding_strategy_if_hybrid=="ctc":
                logger.info("You are using an hybrid model, changing decoding strategy to ctc")
                model.change_decoding_strategy(decoder_type="ctc")
            else:
                logger.info("You are using an hybrid model, using rnnt decoder")
    elif isinstance(model, nemo_asr.models.EncDecMultiTaskModel):
        decode_cfg = model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        model.change_decoding_strategy(decode_cfg)
    return model
