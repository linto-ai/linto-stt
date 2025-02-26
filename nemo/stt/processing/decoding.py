import copy
import os
import time
import regex as re
import json
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
    no_speech_threshold: float = 0.6,
    compression_ratio_threshold: float = 2.4,
) -> dict:
    language = get_language(language)
    kwargs = copy.copy(locals())
    kwargs.pop("model_and_alignementmodel")
    kwargs["model"], kwargs["alignment_model"] = model_and_alignementmodel
    kwargs.pop("alignment_model")
    start_t = time.time()

    res = decode_encoder(**kwargs)

    logger.info(f"Transcription complete (t={time.time() - start_t}s)")

    return res


# def decode_canary(
#     audio,
#     model: nemo_asr.models.EncDecMultiTaskModel,
#     with_word_timestamps,
#     language,
#     remove_punctuation_from_words,
#     **kwargs,
# ):
#     config = {
#         "audio_filepath": audio,  # path to the audio file
#         "duration": None,
#         "taskname": "asr",  
#         "source_lang": language, # language of the audio input, set `source_lang`==`target_lang` for ASR, choices=['en','de','es','fr']
#         "target_lang": language, # language of the text output, choices=['en','de','es','fr']
#         "pnc": "yes",  # whether to have PnC output, choices=['yes', 'no']
#         "answer": "na", 
#     }

#     with open('tmp.json', "w") as f:
#         json.dump(config, f)
        
#     hypothesis = model.transcribe('tmp.json', return_hypotheses=False, timestamps=False)[0]
#     hypothesis.language = language
#     return format_nemo_response(hypothesis, remove_punctuation_from_words=remove_punctuation_from_words)

def decode_encoder(
    audio,
    model,
    with_word_timestamps,
    language,
    remove_punctuation_from_words,
    **kwargs,
):
    kwargs["no_speech_threshold"] = 1  # To avoid empty output
    if kwargs.get("beam_size") is None:
        kwargs["beam_size"] = 1
    if kwargs.get("best_of") is None:
        kwargs["best_of"] = 1
        
    
    hypothesis = model.transcribe([audio], return_hypotheses=True, timestamps=True)[0]
    if isinstance(model._model, nemo_asr.models.EncDecHybridRNNTCTCModel):
        hypothesis=hypothesis[0]
    hypothesis.language = language
    return format_nemo_response(hypothesis, remove_punctuation_from_words=remove_punctuation_from_words)

def contains_alphanum(text: str) -> bool:
    return re.search(r"[^\W\'\-_]", text)

def format_nemo_response(
    hypothesis, remove_punctuation_from_words=False
):
    """Format NeMo response."""

    words = []
    if hypothesis.word_confidence:
        for word, conf in zip(hypothesis.timestep['word'], hypothesis.word_confidence):
            words.append({'word': word['word'], 'start': round(word['start'], 2), 'end': round(word['end'], 2), 'conf': conf})
        return {
            "text": hypothesis.text.strip(),
            "language": hypothesis.language,
            "confidence-score": round(np.average([i['conf'] for i in words]), 2) if len(words)>0 else 0.0, # need to change
            "words": words,
        }
    else:
        for word in hypothesis.timestep['word']:
            words.append({'word': word['word'], 'start': round(word['start'], 2), 'end': round(word['end'], 2)})
        return {
            "text": hypothesis.text.strip(),
            "language": "fr",
            "words": words,
        }
    # print(round(np.exp(-np.mean([np.log(i['conf']) for i in words])), 2))
