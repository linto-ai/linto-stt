import copy
import os
import time
import regex as re
from typing import Tuple, Union
import nemo.collections.asr as nemo_asr

import numpy as np
from stt import (
    logger
)

from .vad import remove_non_speech
from .text_normalize import normalize_text, remove_emoji, remove_punctuation
from .utils import SAMPLE_RATE, get_language

default_prompt = os.environ.get("PROMPT", None)


def decode(
    audio,
    model_and_alignementmodel,  # Tuple[model, alignment_model]
    with_word_timestamps: bool,
    language: str = None,
    remove_punctuation_from_words=False,
    condition_on_previous_text: bool = False,
    no_speech_threshold: float = 0.6,
    compression_ratio_threshold: float = 2.4,
    prompt: str = default_prompt,
) -> dict:
    language = get_language(language)
    kwargs = copy.copy(locals())
    kwargs.pop("model_and_alignementmodel")
    kwargs["model"], kwargs["alignment_model"] = model_and_alignementmodel

    start_t = time.time()

    res = kwargs["model"].transcribe([audio])[0]

    logger.info(f"Transcription complete (t={time.time() - start_t}s)")

    return res


# def decode(
#     audio,
#     model,
#     with_word_timestamps,
#     language,
#     remove_punctuation_from_words,
#     **kwargs,
# ):
#     kwargs["no_speech_threshold"] = 1  # To avoid empty output
#     if kwargs.get("beam_size") is None:
#         kwargs["beam_size"] = 1
#     if kwargs.get("best_of") is None:
#         kwargs["best_of"] = 1
#     segments, info = model.transcribe(
#         audio,
#         word_timestamps=with_word_timestamps,
#         language=language,
#         # Careful with the following options
#         max_initial_timestamp=10000.0,
#         **kwargs,
#     )
#     segments = list(segments)

#     # return format_faster_whisper_response(
#     #     segments, info, remove_punctuation_from_words=remove_punctuation_from_words
#     # )


def contains_alphanum(text: str) -> bool:
    return re.search(r"[^\W\'\-_]", text)
