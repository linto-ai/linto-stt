import copy
import os
import time
import regex as re
import json
from typing import Tuple, Union
import nemo.collections.asr as nemo_asr

import numpy as np
from stt import (
    logger,
    VAD, VAD_DILATATION, VAD_MIN_SILENCE_DURATION, VAD_MIN_SPEECH_DURATION, 
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

def decode_encoder(
    audio,
    model,
    with_word_timestamps,
    language,
    remove_punctuation_from_words,
    **kwargs,
):
    
    if VAD:
        audio_speech, _, _ = remove_non_speech(audio, use_sample=True, method=VAD, dilatation=VAD_DILATATION, \
            min_silence_duration=VAD_MIN_SILENCE_DURATION, min_speech_duration=VAD_MIN_SPEECH_DURATION, avoid_empty_speech=True)
        audio = audio_speech
    hypothesis = model.transcribe([audio], return_hypotheses=True, timestamps=True)[0]      # /!\ Will run out of memory on long audios
    if isinstance(model._model, nemo_asr.models.EncDecHybridRNNTCTCModel):
        hypothesis=hypothesis[0]
    hypothesis.language = language
    return format_nemo_response(hypothesis, remove_punctuation_from_words=remove_punctuation_from_words, with_word_timestamps=with_word_timestamps)

def contains_alphanum(text: str) -> bool:
    return re.search(r"[^\W\'\-_]", text)

def format_nemo_response(
    hypothesis, remove_punctuation_from_words=False, with_word_timestamps=False
):
    words = []
    if with_word_timestamps:
        if hypothesis.word_confidence:
            for word, conf in zip(hypothesis.timestep['word'], hypothesis.word_confidence):
                text = remove_punctuation_from_words(word['word']) if remove_punctuation_from_words else word['word']
                words.append({'word': text, 'start': round(word['start'], 2), 'end': round(word['end'], 2), 'conf': conf})
            return {
                "text": hypothesis.text.strip(),
                "language": hypothesis.language,
                "confidence-score": round(np.average([i['conf'] for i in words]), 2) if len(words)>0 else 0.0, # need to change
                "words": words,
            }
        else:
            for word in hypothesis.timestep['word']:
                text = remove_punctuation_from_words(word['word']) if remove_punctuation_from_words else word['word']
                words.append({'word': text, 'start': round(word['start'], 2), 'end': round(word['end'], 2)})
    return {
        "text": hypothesis.text.strip(),
        "language": hypothesis.language,
        "words": words,
    }