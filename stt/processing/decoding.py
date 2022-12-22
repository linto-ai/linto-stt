import os

import whisper
from whisper.audio import SAMPLE_RATE

import math
import numpy as np
import torch

import re
import string
from num2words import num2words

from stt import logger
from .word_alignment import compute_alignment

# TODO: understand and remove this limitations
torch.set_num_threads(1)

def get_default_language():
    return os.environ.get("LANGUAGE", None)

def decode(audio: torch.Tensor,
    model: whisper.model.Whisper,
    alignment_model: "Any",
    with_word_timestamps: bool,
    language: str = None,
    beam_size: int = None,
    no_speech_threshold: float = 0.6,
    logprob_threshold: float = -1.0,
    compression_ratio_threshold: float = 2.4,
    normalize_text_as_words = False,
    ) -> dict:
    """Transcribe the audio data using Whisper with the defined model."""
    result = {"text": "", "confidence-score": 0.0, "words": []}

    fp16 = model.device != torch.device("cpu")

    if language is None:
        language = get_default_language()

    logger.info(f"Transcribing audio with language {language}...")

    whisper_res = model.transcribe(audio,
        language = language,
        fp16 = fp16,
        temperature = 0.0, # For deterministic results
        beam_size = beam_size,
        no_speech_threshold = no_speech_threshold,
        logprob_threshold = logprob_threshold,
        compression_ratio_threshold = compression_ratio_threshold
    )

    text = whisper_res["text"].strip()
    if normalize_text_as_words:
        text = normalize_text(text, language)
        text = remove_punctuation(text)
    segments = whisper_res["segments"]

    result["text"] = text
    result["confidence-score"] = np.exp(np.array([r["avg_logprob"] for r in segments])).mean()
    if not with_word_timestamps:
        if not normalize_text_as_words:
            text = normalize_text(text, language)
            text = remove_punctuation(text)
        result["words"] = text.split()
    else:
        # Compute word timestamps
        result["words"] = []
        max_t = audio.shape[0]
        for segment in segments:
            offset = segment["start"]
            start = min(max_t, round(segment["start"] * SAMPLE_RATE))
            end = min(max_t, round(segment["end"] * SAMPLE_RATE))
            sub_audio = audio[start:end]
            sub_text = segment["text"]
            sub_text = normalize_text(sub_text, language)
            sub_text = remove_punctuation(sub_text)
            labels, emission, trellis, segments, word_segments = compute_alignment(sub_audio, sub_text, alignment_model)
            ratio = len(sub_audio) / (trellis.size(0) * SAMPLE_RATE)
            sub_words = sub_text.split()
            assert len(sub_words) == len(word_segments), f"Unexpected number of words: {len(sub_words)} != {len(word_segments)}"
            for word, segment in zip(sub_words, word_segments):
                result["words"].append({
                    "word": word,
                    "start": segment.start * ratio + offset,
                    "end": segment.end * ratio + offset,
                    "conf": segment.score,
                })

    return result


custom_punctuations = string.punctuation.replace("'", "").replace("-", "")

def remove_punctuation(text: str) -> str:
    # Remove all punctuation except apostrophe
    return text.translate(str.maketrans("", "", custom_punctuations))

_whitespace_re = re.compile(r'[^\S\r\n]+')

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text).strip()


def normalize_text(text: str, lang: str) -> str:
    """ Transform digits into characters... """
    
    # Roman digits
    if re.search(r"[IVX]", text):
        if lang == "en":
            digits = re.findall(r"\b(?=[XVI])M*(XX{0,3})(I[XV]|V?I{0,3})(st|nd|rd|th)?\b", text)
            digits = ["".join(d) for d in digits]
        elif lang == "fr":
            digits = re.findall(r"\b(?=[XVI])M*(XX{0,3})(I[XV]|V?I{0,3})(ème|eme|e|er|ère)?\b", text)
            digits = ["".join(d) for d in digits]
        else:
            digits = []
        if digits:
            digits = sorted(list(set(digits)), reverse=True, key=lambda x: (len(x), x))
            for s in digits:
                filtered = re.sub("[a-z]", "", s)
                ordinal = filtered != s
                digit = romanToDecimal(filtered)
                v = undigit(str(digit), lang=lang, to= "ordinal" if ordinal else "cardinal")
                text = re.sub(r"\b" + s + r"\b", v, text)

    # Ordinal digits
    if lang == "en":
        digits = re.findall(r"\b\d*1(?:st)|\d*2(?:nd)|\d*3(?:rd)|\d+(?:th)\b", text)
    elif lang == "fr":
        digits = re.findall(r"\b1(?:ère|ere|er|re|r)|2(?:nd|nde)|\d+(?:ème|eme|e)\b", text)
    else:
        logger.warn(f"Language {lang} not supported for normalization. Some words might be mis-localized.")
        digits = []
    if digits:
        digits = sorted(list(set(digits)), reverse=True, key=lambda x: (len(x), x))
        for digit in digits:
            word = undigit(re.findall(r"\d+", digit)[0], to= "ordinal", lang = lang)
            text = re.sub(r'\b'+str(digit)+r'\b', word, text)

    # Cardinal digits
    digits = re.findall(r"(?:\-?\b[\d/]*\d+(?: \d\d\d)+\b)|(?:\-?\d[/\d]*)",text)
    digits = list(map(lambda s: s.strip(r"[/ ]"), digits))
    digits = list(set(digits))
    digits = digits + flatten([c.split() for c in digits if " " in c])
    digits = digits + flatten([c.split("/") for c in digits if "/" in c])
    digits = sorted(digits, reverse=True, key=lambda x: (len(x), x))
    for digit in digits:
        digitf = re.sub("/+", "/", digit)
        if not digitf:
            continue
        numslash = len(re.findall("/", digitf))
        if numslash == 0:
            word = undigit(digitf, lang = lang)
        elif numslash == 1: # Fraction or date
            i = digitf.index("/")
            is_date = False
            if len(digitf[i+1:]) == 2:
                try:
                    first = int(digitf[:i])
                    second = int(digitf[i+1:])
                    is_date = first > 0 and first < 32 and second > 0 and second < 13
                except: pass
            if is_date:
                first = undigit(digitf[:i].lstrip("0"), lang = lang)
                if first == "un": first = "premier"
                second = _int_to_month[second]
            else:
                first = undigit(digitf[:i], lang = lang)
                second = undigit(digitf[i+1:], to="denominator", lang = lang)
                if float(digitf[:i]) > 2. and second[-1] != "s":
                    second += "s"
            word = first + " " + second
        elif numslash == 2: # Maybe a date
            i1 = digitf.index("/")
            i2 = digitf.index("/", i1+1)
            is_date = False
            if len(digitf[i1+1:i2]) == 2 and len(digitf[i2+1:]) == 4:
                try:
                    first = int(digitf[:i1])
                    second = int(digitf[i1+1:i2])
                    third = int(digitf[i2+1:])
                    is_date = first > 0 and first < 32 and second > 0 and second < 13 and third > 1000
                except: pass
            third = undigit(digitf[i2+1:], lang = lang)
            if is_date:
                first = undigit(digitf[:i1].lstrip("0"), lang = lang)
                if first == "un": first = "premier"
                second = _int_to_month.get(lang, {}).get(int(digitf[i1+1:i2]), digitf[i1+1:i2])
                word = " ".join([first, second, third])
            else:
                word = " / ".join([undigit(s, lang = lang) for s in digitf.split('/')])
        else:
            word = " / ".join([undigit(s, lang = lang) for s in digitf.split('/')])
        # Replace
        if " " in digit:
            text = re.sub(r'\b'+str(digit)+r'\b', " "+word+" ", text)
        else:
            text = re.sub(str(digit), " "+word+" ", text)

    # TODO: symbols (currencies...)

    return collapse_whitespace(text)

def undigit(str, lang, to="cardinal"):
    str = re.sub(" ","", str)
    if to == "denominator":
        assert lang == "fr"
        if str == "2": return "demi"
        if str == "3": return "tiers"
        if str == "4": return "quart"
        to = "ordinal"
    if str.startswith("0") and to == "cardinal":
        numZeros = len(re.findall(r"0+", str)[0])
        if numZeros < len(str):
            return numZeros * (my_num2words(0, lang=lang, to="cardinal")+" ") + my_num2words(float(str), lang=lang, to=to)
    return my_num2words(float(str), lang=lang, to=to)


def my_num2words(x, lang, to = "cardinal", orig = ""):
    """
    Bugfix for num2words
    """
    try:
        if lang == "fr" and to == "ordinal":
            return num2words(x, lang=lang, to=to).replace("vingtsième", "vingtième")
        else:
            return num2words(x, lang=lang, to=to)
    except OverflowError:
        if x == math.inf: # !
            return " ".join(my_num2words(xi, lang=lang, to=to) for xi in orig)
        if x == -math.inf: # !
            return "moins " + my_num2words(-x, lang=lang, to=to, orig=orig.replace("-" , ""))
        # TODO: print a warning
        return my_num2words(x//10, lang=lang, to=to)

def flatten(l):
    """
    flatten a list of lists
    """
    return [item for sublist in l for item in sublist]

_int_to_month = {
    "fr": {
        1: "janvier",
        2: "février",
        3: "mars",
        4: "avril",
        5: "mai",
        6: "juin",
        7: "juillet",
        8: "août",
        9: "septembre",
        10: "octobre",
        11: "novembre",
        12: "décembre",
    },
    "en": {
        1: "january",
        2: "february",
        3: "march",
        4: "april",
        5: "may",
        6: "june",
        7: "july",
        8: "august",
        9: "september",
        10: "october",
        11: "november",
        12: "december",
    }
}


def romanToDecimal(str):
    def value(r):
        if (r == 'I'):
            return 1
        if (r == 'V'):
            return 5
        if (r == 'X'):
            return 10
        if (r == 'L'):
            return 50
        if (r == 'C'):
            return 100
        if (r == 'D'):
            return 500
        if (r == 'M'):
            return 1000
        return -1

    res = 0
    i = 0
    while (i < len(str)):
        # Getting value of symbol s[i]
        s1 = value(str[i])
        if (i + 1 < len(str)):
            # Getting value of symbol s[i + 1]
            s2 = value(str[i + 1])
            # Comparing both values
            if (s1 >= s2):
                # Value of current symbol is greater
                # or equal to the next symbol
                res = res + s1
                i = i + 1
            else:
                # Value of current symbol is greater
                # or equal to the next symbol
                res = res + s2 - s1
                i = i + 2
        else:
            res = res + s1
            i = i + 1
    return res
