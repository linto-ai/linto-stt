import copy
import os
import time
import regex as re
import json
import math
from typing import Tuple, Union
import nemo.collections.asr as nemo_asr
import torch

import numpy as np
from lintostt.nemo_backend.stt import (
    logger,
    VAD, VAD_DILATATION, VAD_MIN_SILENCE_DURATION, VAD_MIN_SPEECH_DURATION,
    LONG_FILE_THRESHOLD, LONG_FILE_CHUNK_LEN, LONG_FILE_CHUNK_CONTEXT_LEN
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
    if len(audio)>SAMPLE_RATE*LONG_FILE_THRESHOLD:
        logger.info(f"Audio last more than {LONG_FILE_THRESHOLD/60}min, splitting the decoding")
        hypothesis = stream_long_file(audio, model)
        hypothesis['language'] = language
        return format_nemo_response(hypothesis, from_dict=True, remove_punctuation_from_words=remove_punctuation_from_words, with_word_timestamps=with_word_timestamps)
    else:
        hypothesis = model.transcribe([audio], return_hypotheses=True, timestamps=True)[0]      # /!\ Will run out of memory on long audios
        hypothesis.language = language
        return format_nemo_response(hypothesis, from_dict=False, remove_punctuation_from_words=remove_punctuation_from_words, with_word_timestamps=with_word_timestamps)

def contains_alphanum(text: str) -> bool:
    return re.search(r"[^\W\'\-_]", text)

def format_nemo_response(
    hypothesis, from_dict=False, remove_punctuation_from_words=False, with_word_timestamps=False
):
    words = []
    if from_dict:
        if with_word_timestamps:
            if hypothesis.get('word_confidence', False):
                for word, conf in zip(hypothesis['timestamp']['word'], hypothesis['word_confidence']):
                    text = remove_punctuation_from_words(word['word']) if remove_punctuation_from_words else word['word']
                    words.append({'word': text, 'start': round(word['start'], 2), 'end': round(word['end'], 2), 'conf': conf})
                return {
                    "text": hypothesis['text'].strip(),
                    "language": hypothesis.get('language', None),
                    "confidence-score": round(np.average([i['conf'] for i in words]), 2) if len(words)>0 else 0.0, # need to change
                    "words": words,
                }
            else:
                for word in hypothesis['timestamp']['word']:
                    text = remove_punctuation_from_words(word['word']) if remove_punctuation_from_words else word['word']
                    words.append({'word': text, 'start': round(word['start'], 2), 'end': round(word['end'], 2)})
        return {
            "text": hypothesis['text'].strip(),
            "language": hypothesis.get('language', None),
            "words": words,
        }
    else:
        if with_word_timestamps:
            if hypothesis.word_confidence:
                for word, conf in zip(hypothesis.timestamp['word'], hypothesis.word_confidence):
                    text = remove_punctuation_from_words(word['word']) if remove_punctuation_from_words else word['word']
                    words.append({'word': text, 'start': round(word['start'], 2), 'end': round(word['end'], 2), 'conf': conf})
                return {
                    "text": hypothesis.text.strip(),
                    "language": hypothesis.language,
                    "confidence-score": round(np.average([i['conf'] for i in words]), 2) if len(words)>0 else 0.0, # need to change
                    "words": words,
                }
            else:
                for word in hypothesis.timestamp['word']:
                    text = remove_punctuation_from_words(word['word']) if remove_punctuation_from_words else word['word']
                    words.append({'word': text, 'start': round(word['start'], 2), 'end': round(word['end'], 2)})
        return {
            "text": hypothesis.text.strip(),
            "language": hypothesis.language,
            "words": words,
        }
    

def get_chunks(samples, frame_duration, sample_rate, context_duration=0):
    _samples = samples
    _chunk_len = frame_duration*sample_rate
    _start = 0
    _context_len = context_duration*sample_rate
    chunks_list = []
    number_of_chunks = math.ceil(len(_samples) / _chunk_len)
    for i in range(number_of_chunks):
        last = int(_start + _chunk_len)
        start_with_context = max(0, int(_start - _context_len))
        end_with_context = min(len(_samples), int(last + _context_len))
        chunk = _samples[start_with_context: end_with_context]
        _start = last
        chunks_list.append(np.array(chunk))
    return chunks_list

class ChunkBufferDecoder:

    def __init__(self, asr_model, chunk_len_in_secs=1, context_len_in_secs=3):
        self.asr_model = asr_model
        self.asr_model.eval()
        self.buffers = []
        self.all_preds = []
        self.chunk_len = chunk_len_in_secs
        self.context_len = context_len_in_secs
        
    @torch.no_grad()    
    def transcribe_buffers(self, buffers):
        self.buffers = buffers
        self._get_batch_preds(buffers)      
        return self.merge_results()
    
    def _get_batch_preds(self, buffers):
        hypothesis = self.asr_model.transcribe(buffers, return_hypotheses=True, timestamps=True, batch_size=2)
        self.all_preds=hypothesis
    
    def merge_results(self):
        result_text = []
        result_timestep = []
        base_acceptance_per_character = 0.015       # in case words are cut in the middle
        for i, chunk_hypothesis in enumerate(self.all_preds):
            for word in chunk_hypothesis.timestamp['word']:
                acceptance = base_acceptance_per_character * len(word['word'])
                merged_word = dict(word=word['word'], start=word['start']+i*self.chunk_len-self.context_len, end=word['end']+i*self.chunk_len-self.context_len)
                if i==0 and word['end']-acceptance<self.chunk_len+self.context_len:
                    result_text.append(word['word'])
                    merged_word = dict(word=word['word'], start=word['start']+i*self.chunk_len, end=word['end']+i*self.chunk_len)
                    result_timestep.append(merged_word)
                elif i==len(self.all_preds)-1 and word['start']+acceptance>self.context_len:
                    result_text.append(word['word'])
                    result_timestep.append(merged_word)
                elif word['start']+acceptance>self.context_len and word['end']-acceptance<self.chunk_len+self.context_len:
                    result_text.append(word['word'])
                    result_timestep.append(merged_word)
                    
        result = {"text": " ".join(result_text), "timestamp": {"word": result_timestep}}
        return result

def stream_long_file(audio, model):
    buffer_list = get_chunks(audio, LONG_FILE_CHUNK_LEN, SAMPLE_RATE, LONG_FILE_CHUNK_CONTEXT_LEN)
    asr_decoder = ChunkBufferDecoder(model, chunk_len_in_secs=LONG_FILE_CHUNK_LEN, context_len_in_secs=LONG_FILE_CHUNK_CONTEXT_LEN)
    result = asr_decoder.transcribe_buffers(buffer_list)
    return result