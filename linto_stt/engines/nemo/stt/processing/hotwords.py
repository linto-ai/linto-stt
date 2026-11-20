"""Hotwords for the nemo engine: NeMo's GPU phrase boosting.

The hotwords are compiled into a boosting tree (a prefix tree of the terms, split
into the model's tokens). During decoding, it adds a bonus to the score of the
tokens that continue one of the terms. It is not a prompt: it is part of the
decoding configuration of the model. Supported by all the model types, greedy
or beam-search decoding: CTC (`greedy_batch`, `beam_batch`), RNN-T / TDT
(`greedy_batch`, `malsd_batch`) and Canary (`beam`). A model configured with
another strategy is decoded, while hotwords are given, with the one of the same
kind that supports them (e.g. `greedy` -> `greedy_batch`). Only decodings with an
external language model (CTC `pyctcdecode`, `flashlight`, `wfst`) ignore them.

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

# Decoding strategies that take a boosting tree (pinned NeMo), with the section of
# the decoding config that holds it. The other strategies are replaced, while
# hotwords are given, by the one of the same kind that takes it (greedy: its
# batched form; beam search: the batched beam search, with the same beam size).
# `maes_batch` takes it too, but only for RNN-T (not TDT): replaced as well.
_BOOSTING_SECTION = {
    "ctc": {"greedy_batch": "greedy", "beam_batch": "beam"},
    "rnnt": {"greedy_batch": "greedy", "malsd_batch": "beam"},
    "multitask": {"beam": "beam"},
}
_BOOSTING_STRATEGY = {
    "ctc": {"greedy": "greedy_batch", "beam": "beam_batch"},
    "rnnt": {"greedy": "greedy_batch", "beam": "malsd_batch", "tsd": "malsd_batch",
             "alsd": "malsd_batch", "maes": "malsd_batch", "maes_batch": "malsd_batch"},
    "multitask": {},
}


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


def _decoder_kind(model):
    if _is_multitask(model):
        return "multitask"
    return "ctc" if _uses_ctc(model) else "rnnt"


def _boosted(model, base, terms, language):
    """The decoding configuration with a boosting tree of `terms`, or None if no
    strategy of this decoder takes one (e.g. CTC with an external language model)."""
    kind = _decoder_kind(model)
    strategy = _BOOSTING_STRATEGY[kind].get(base.strategy, base.strategy)
    section_name = _BOOSTING_SECTION[kind].get(strategy)
    if section_name is None:
        return None
    if strategy != base.strategy:
        logger.info(f"Decoding strategy {base.strategy!r} replaced by {strategy!r} for the hotwords")
    cfg = copy.deepcopy(base)
    with open_dict(cfg):
        cfg.strategy = strategy
        section = cfg[section_name]
        if HOTWORDS_BOOSTING_DEPTH_SCALING is not None:
            depth_scaling = HOTWORDS_BOOSTING_DEPTH_SCALING
        else:
            depth_scaling = 1.0 if kind == "multitask" else 2.0  # NeMo's recommendations
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
    if cfg is not None:
        try:
            _apply(model, cfg)
            logger.info(f"Phrase boosting of {len(terms)} hotword(s): {', '.join(terms)[:200]}")
            state.key = key
            return
        except Exception as err:  # e.g. a model type whose decoder refuses the tree
            logger.warning(f"Hotwords ignored: phrase boosting failed with this model ({err})")
    else:
        logger.warning(f"Hotwords ignored: phrase boosting is not supported with the decoding strategy "
                       f"{state.base.get('strategy')!r} of this model")
    # Decode without hotwords (not with the tree of a previous request).
    if state.key is not None:
        _apply(model, state.base)
        state.key = None
