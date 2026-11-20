"""Hotwords for the nemo engine: NeMo's GPU phrase boosting.

The hotwords are compiled into a boosting tree (a prefix tree of the terms, split
into the model's tokens). During decoding, it adds a bonus to the score of the
tokens that continue one of the terms. It is not a prompt: it is part of the
decoding configuration of the model. Supported (NeMo >= 2.5):

- CTC decoding with the `greedy_batch` strategy (a `greedy` CTC decoding is
  switched to `greedy_batch` while boosting): CTC models, and hybrid models
  decoded with CTC (ARCHITECTURE=hybrid_bpe_ctc);
- RNN-T / TDT decoding with the `greedy` or `greedy_batch` strategy (parakeet,
  hybrid models decoded with RNN-T);
- Canary (multi-task models), whose decoding is a beam search.

The decoding configuration belongs to the model, shared by all requests: setting
the hotwords and decoding happen under LOCK. The boosting tree is rebuilt only
when the hotwords (or the language) change; a request without hotwords restores
the model's own decoding configuration.
"""
import copy
import inspect
import threading

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig
from omegaconf import open_dict

from linto_stt.hotwords import parse_hotwords
from linto_stt.engines.nemo.stt import (
    HOTWORDS_BOOSTING_ALPHA,
    HOTWORDS_BOOSTING_CONTEXT_SCORE,
    HOTWORDS_BOOSTING_DEPTH_SCALING,
    logger,
)

LOCK = threading.Lock()

_GREEDY = ("greedy", "greedy_batch")


def _is_multitask(model):
    return isinstance(model, nemo_asr.models.EncDecMultiTaskModel)


def _uses_ctc(model):
    if isinstance(model, nemo_asr.models.EncDecHybridRNNTCTCModel):
        return getattr(model, "cur_decoder", "rnnt") == "ctc"
    return isinstance(model, nemo_asr.models.EncDecCTCModel)


def _decoding_cfg(model):
    if isinstance(model, nemo_asr.models.EncDecHybridRNNTCTCModel) and _uses_ctc(model):
        return model.cfg.aux_ctc.decoding
    return model.cfg.decoding


def _apply(model, cfg):
    kwargs = {}
    params = inspect.signature(model.change_decoding_strategy).parameters
    if "decoder_type" in params:  # hybrid models
        kwargs["decoder_type"] = "ctc" if _uses_ctc(model) else "rnnt"
    if "verbose" in params:  # not on multi-task models (Canary)
        kwargs["verbose"] = False
    model.change_decoding_strategy(cfg, **kwargs)


class _State:
    def __init__(self):
        self.key = None  # (terms, language) of the boosting tree in place, None without
        self.base = None  # the model's own decoding configuration


_states = {}


def _boosted(model, base, terms, language):
    """The decoding configuration with a boosting tree of `terms`, or None if the
    model's decoding does not support phrase boosting."""
    cfg = copy.deepcopy(base)
    with open_dict(cfg):
        if _is_multitask(model):
            section = cfg.beam
            depth_scaling = 1.0
        elif cfg.strategy in _GREEDY:
            if _uses_ctc(model):
                cfg.strategy = "greedy_batch"  # CTC boosting needs the batched greedy decoding
            section = cfg.greedy
            depth_scaling = 2.0
        else:
            return None
        if HOTWORDS_BOOSTING_DEPTH_SCALING is not None:
            depth_scaling = HOTWORDS_BOOSTING_DEPTH_SCALING
        # NeMo's own config class: some decoding configs are typed (structured)
        # and refuse a plain dict there.
        section.boosting_tree = BoostingTreeModelConfig(
            key_phrases_list=list(terms),
            context_score=HOTWORDS_BOOSTING_CONTEXT_SCORE,
            depth_scaling=depth_scaling,
            # the language of the terms, used only by aggregate tokenizers
            source_lang=language if language and language != "*" else "en",
        )
        section.boosting_tree_alpha = HOTWORDS_BOOSTING_ALPHA
    return cfg


def set_hotwords(model, hotwords, language=None):
    """Set (or remove) the boosting tree of the model's decoding for these
    hotwords. To be called under LOCK, right before decoding."""
    terms = parse_hotwords(hotwords)
    key = (tuple(terms), language) if terms else None
    state = _states.setdefault(id(model), _State())
    if key == state.key:
        return
    if state.base is None:
        state.base = copy.deepcopy(_decoding_cfg(model))
    if key is None:
        _apply(model, state.base)
        state.key = None
        return
    cfg = _boosted(model, state.base, terms, language)
    if cfg is None:
        logger.warning(f"Hotwords ignored: phrase boosting is not supported with the decoding strategy "
                       f"{state.base.get('strategy')!r} of this model")
        return
    logger.info(f"Phrase boosting of {len(terms)} hotword(s): {', '.join(terms)[:200]}")
    _apply(model, cfg)
    state.key = key
