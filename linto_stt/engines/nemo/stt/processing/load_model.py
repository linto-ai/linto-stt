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

from linto_stt.engines.nemo.stt import logger, ATT_CONTEXT_SIZE, ATT_CONTEXT_SIZE_RIGHT
from .utils import supports_cache_aware_streaming, enable_strip_lang_tags
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)


def load_nemo_model(model_type_or_file, device="cpu", download_root=None, decoding_strategy_if_hybrid="ctc"):
    start = time.time()
    logger.info(f"Loading Nemo model {model_type_or_file}...")
    default_cache_root = os.path.join(os.path.expanduser("~"), ".cache")
    if download_root is None:
        download_root = default_cache_root
    if model_type_or_file.endswith(".nemo"):
        model = nemo_asr.models.ASRModel.restore_from(
            model_type_or_file, map_location=device)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_type_or_file, map_location=device)
    logger.info(f"Nemo model loaded. (t={time.time() - start:.2f}s)")
    if isinstance(model, nemo_asr.models.EncDecRNNTModel):
        if isinstance(model, nemo_asr.models.EncDecHybridRNNTCTCModel):
            if decoding_strategy_if_hybrid == "ctc":
                logger.info(
                    "You are using an hybrid model, changing decoding strategy to ctc")
                model.change_decoding_strategy(decoder_type="ctc")
            else:
                logger.info(
                    "You are using an hybrid model, using rnnt decoder")
    elif isinstance(model, nemo_asr.models.EncDecMultiTaskModel):
        decode_cfg = model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        model.change_decoding_strategy(decode_cfg)

    _configure_attention_context(model)

    # Prompt-conditioned models emit inline language tags ("<fr-FR>"); enable the
    # model's native stripping (no-op for models that don't support it). Covers
    # the offline path; the streaming path re-applies it after it rebuilds the
    # decoding (see _configure_streaming_decoding).
    if enable_strip_lang_tags(model):
        logger.info("Enabled native language-tag stripping (strip_lang_tags).")

    return model


def _configure_attention_context(model):
    """Set the encoder's attention context [left, right] (encoder frames, ~80ms
    each), with model-dependent defaults.

    - Cache-aware streaming models: keep their TRAINED attention and only adjust
      the look-ahead via ``encoder.set_default_att_context_size``. Using
      ``rel_pos_local_attn`` here would REPLACE the trained attention and corrupt
      the streaming cache (it is meant to convert *offline* models). Default left
      is the model's own (ATT_CONTEXT_SIZE unset => unchanged); default right is 0
      (lowest latency, but worst boundary accuracy — raise it, e.g. to 13, for
      better start/end recognition).
    - Other (offline) models: bound the context by switching to local attention
      (``rel_pos_local_attn``). Default left 128; default right same as left.
    """
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return

    if supports_cache_aware_streaming(model):
        if ATT_CONTEXT_SIZE is None and ATT_CONTEXT_SIZE_RIGHT is None:
            # Neither set: keep the model's TRAINED attention/look-ahead. Forcing a
            # value here (especially right=0) badly degrades quality, so we only
            # touch it when the user explicitly asks.
            logger.info(
                "Cache-aware streaming model: keeping trained att_context_size "
                f"({list(encoder.att_context_size)}).")
            return
        if not hasattr(encoder, "set_default_att_context_size"):
            logger.warning(
                "Cache-aware streaming model exposes no set_default_att_context_size; "
                "cannot set the look-ahead context.")
            return
        current = encoder.att_context_size
        default_left = current[0] if isinstance(current[0], int) else current[0][0]
        left = ATT_CONTEXT_SIZE if ATT_CONTEXT_SIZE is not None else default_left
        right = ATT_CONTEXT_SIZE_RIGHT if ATT_CONTEXT_SIZE_RIGHT is not None else 0
        logger.info(
            f"Cache-aware streaming model: setting att_context_size=[{left}, {right}]")
        encoder.set_default_att_context_size(att_context_size=[left, right])
    else:
        left = ATT_CONTEXT_SIZE if ATT_CONTEXT_SIZE is not None else 128
        right = ATT_CONTEXT_SIZE_RIGHT if ATT_CONTEXT_SIZE_RIGHT is not None else left
        if left <= 0 or not hasattr(model, "change_attention_model"):
            return
        logger.info(
            f"Offline model: switching to local attention "
            f"(att_context_size=[{left}, {right}]).")
        model.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=[left, right],
        )
