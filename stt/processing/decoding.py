import os

import numpy as np
import copy

from stt import logger, USE_CTRANSLATE2
from .utils import SAMPLE_RATE
from .load_model import load_alignment_model, get_alignment_model
from .text_normalize import remove_punctuation, normalize_text, remove_emoji, _punctuations
from .word_alignment import compute_alignment

if not USE_CTRANSLATE2:
    import torch 
    import whisper_timestamped


def get_language():
    """
    Get the language from the environment variable LANGUAGE, and format as expected by Whisper.
    """
    language = os.environ.get("LANGUAGE", "*")
    # "fr-FR" -> "fr" (language-country code to ISO 639-1 code)
    if len(language) > 2 and language[2] == "-":
        language = language.split("-")[0]
    # "*" means "all languages"
    if language == "*":
        language = None
    return language


def decode(audio,
           model,
           alignment_model: "Any",
           with_word_timestamps: bool,
           language: str = None,
           remove_punctuation_from_words=False,
           beam_size: int = None,
           best_of: int = None,
           temperature: float = 0.0,
           condition_on_previous_text: bool = False,
           no_speech_threshold: float = 0.6,
           compression_ratio_threshold: float = 2.4,
           ) -> dict:

    if language is None:
        language = get_language()

    kwargs = copy.copy(locals())

    logger.info(f"Transcribing audio with language {language}...")

    if USE_CTRANSLATE2:
        kwargs.pop("alignment_model")
        return decode_ct2(**kwargs)
    else:
        return decode_torch(**kwargs)


def decode_ct2(audio,
               model,
               with_word_timestamps,
               language,
               remove_punctuation_from_words,
               **kwargs
               ):

    kwargs["no_speech_threshold"] = 1   # To avoid empty output
    if kwargs.get("beam_size") is None:
        kwargs["beam_size"] = 1
    if kwargs.get("best_of") is None:
        kwargs["best_of"] = 1

    segments, info = model.transcribe(
        audio,
        word_timestamps=with_word_timestamps,
        language=language,
        # Careful with the following options
        max_initial_timestamp=10000.0,
        **kwargs)

    segments = list(segments)

    return format_faster_whisper_response(
        segments, info,
        remove_punctuation_from_words=remove_punctuation_from_words
    )


def decode_torch(audio,
                 model,
                 alignment_model,
                 with_word_timestamps,
                 language,
                 remove_punctuation_from_words,
                 beam_size,
                 best_of,
                 temperature,
                 condition_on_previous_text,
                 no_speech_threshold,
                 compression_ratio_threshold,
                 normalize_text_as_words=False,
                 ):
    """Transcribe the audio data using Whisper with the defined model."""

    fp16 = model.device != torch.device("cpu")

    kwargs = dict(
        language=language,
        fp16=fp16,
        temperature=temperature,
        beam_size=beam_size,
        best_of=best_of,
        condition_on_previous_text=condition_on_previous_text,
        no_speech_threshold=no_speech_threshold,
        compression_ratio_threshold=compression_ratio_threshold
    )

    if alignment_model is None:
        # Use Whisper cross-attention weights
        whisper_res = whisper_timestamped.transcribe(model, audio, **kwargs)
        if language is None:
            language = whisper_res["language"]
            logger.info(f"Detected language: {language}")
        return format_whisper_timestamped_response(whisper_res, remove_punctuation_from_words=remove_punctuation_from_words)

    # Force deterministic results
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    
    whisper_res = model.transcribe(audio, **kwargs)

    text = whisper_res["text"]
    text = remove_emoji(text).strip()
    if normalize_text_as_words:
        text = normalize_text(text, language)
        if remove_punctuation_from_words:
            text = remove_punctuation(text)
    segments = whisper_res["segments"]
    if language is None:
        language = whisper_res["language"]
        logger.info(f"Detected language: {language}")
    if isinstance(alignment_model, dict):
        # Load alignment model on the fly
        if language not in alignment_model:
            alignment_model_name = get_alignment_model(language)
            logger.info(f"Loading alignment model {alignment_model_name} ({'local' if os.path.exists(alignment_model_name) else 'remote'})...")
            alignment_model[language] = load_alignment_model(alignment_model_name, device=model.device, download_root="/opt")
        spec_alignment_model = alignment_model[language]
    else:
        spec_alignment_model = alignment_model

    result = {}
    result["text"] = text
    result["language"] = language
    result["confidence-score"] = np.exp(
        np.array([r["avg_logprob"] for r in segments])
    ).mean() if len(segments) else 0.0

    if not with_word_timestamps:
        if not normalize_text_as_words:
            text = normalize_text(text, language)
            if remove_punctuation_from_words:
                text = remove_punctuation(text)
        result["words"] = text.split()
    else:
        # Compute word timestamps
        result["words"] = []
        max_t = audio.shape[0]

        # Ensure that the segments start / end time are increasing
        # (because there is no guarantee with Whisper)
        previous_start = 0.0
        for segment in segments:
            if segment["start"] < previous_start:
                segment["start"] = previous_start
            if segment["end"] <= segment["start"]:
                segment["end"] = segment["start"] + 1.0
            previous_start = segment["end"]

        for segment in segments:
            offset = segment["start"]
            start = min(max_t, round(segment["start"] * SAMPLE_RATE))
            end = min(max_t, round(segment["end"] * SAMPLE_RATE))
            sub_audio = audio[start:end]
            sub_text = segment["text"]
            logger.debug(f"Aligning text: {sub_text}")
            sub_text = remove_emoji(sub_text).strip()
            sub_text = normalize_text(sub_text, language)
            if remove_punctuation_from_words:
                sub_text = remove_punctuation(sub_text)
            if not sub_text:
                logger.warn(
                    f"Lost text in segment {segment['start']}-{segment['end']}")
                continue
            labels, emission, trellis, segments, word_segments = compute_alignment(
                sub_audio, sub_text, spec_alignment_model)
            ratio = len(sub_audio) / (trellis.size(0) * SAMPLE_RATE)
            sub_words = sub_text.split()
            words = []
            use_original_words = True
            if len(sub_words) != len(word_segments):
                logger.warn(
                    f"Alignment failed. Some words might be mis-rendered.\nNumber of words: {len(sub_words)} != {len(word_segments)}\n>>>\n{sub_words}\n<<<\n{[segment.label for segment in word_segments]}")
                assert len(word_segments) < len(sub_words)
                use_original_words = False
            for word, seg in zip(sub_words, word_segments):
                words.append({
                    "word": word if use_original_words else seg.label,
                    "start": seg.start * ratio + offset,
                    "end": seg.end * ratio + offset,
                    "conf": seg.score,
                })
            # Glue the words inside a segment
            for i, word in enumerate(words):
                if i == 0:
                    word["start"] = segment["start"]
                else:
                    word["start"] = words[i-1]["end"]
                if i == len(words) - 1:
                    word["end"] = segment["end"]
                else:
                    word["end"] = .5 * (words[i+1]["start"] + word["end"])
            # Accumulate results
            result["words"] += words

    return result


def format_whisper_timestamped_response(transcription, remove_punctuation_from_words=False):
    """Format Whisper response."""

    for i, seg in enumerate(transcription["segments"][:-1]):
        for expected_keys in ["start", "end", "words", "avg_logprob"]:
            assert expected_keys in seg, f"Missing '{expected_keys}' in segment {i} (that has keys {list(seg.keys())})"

    words = []

    segments = transcription.get("segments", [])

    for seg in segments:
        for word in seg.get("words", []):
            text = word["text"]
            if remove_punctuation_from_words:
                text = remove_punctuation(text)
            words.append({
                "word": text,
                "start": word["start"],
                "end": word["end"],
                "conf": word["confidence"],
            })

    return {
        "text": transcription["text"].strip(),
        "language": transcription["language"],
        "confidence-score": round(np.exp(np.array([r["avg_logprob"] for r in segments])).mean(), 2) if len(segments) else 0.0,
        "words": words,
    }


def format_faster_whisper_response(segments, info,
                                   remove_punctuation_from_words=False):

    language = info.language
    duration = info.duration

    def checked_timestamps(start, end=None):
        if start > duration or (end is not None and end > duration):
            print("WARNING, timestamp %f is greater than duration %f" % (max(start, end if end else start), duration))
        if end and end <= start:
            if end == start:
                pass # end = start + 0.01
            else:
                print("WARNING, end timestamp %f is smaller than start timestamp %f" % (end, start))
        if end is None:
            return start
        return (start, end)

    segments_list = []
    for segment in segments:
        start, end = checked_timestamps(segment.start, segment.end)

        words = []
        if segment.words:
            for word in segment.words:
                if len(words) and (not(word.word.strip()) or word.word.strip()[0] in _punctuations):
                    words[-1]["text"] += word.word
                    if word.word.strip() not in _punctuations:
                        words[-1]["confidence"].append(word.probability)
                        _, words[-1]["end"] = checked_timestamps(words[-1]["end"], word.end)
                    continue
                words.append(
                    {"text": word.word, "confidence": [word.probability]} | dict(zip(("start", "end"), checked_timestamps(word.start, word.end)))
                )

            for word in words:
                word["text"] = word["text"].strip()
                word["confidence"] = round(np.mean([c for c in word["confidence"]]), 2)

        segments_list.append({
            "text": segment.text.strip(),
            "start": start,
            "end": end,
            "avg_logprob": segment.avg_log_prob,
            "words": words
        })

    assert len(segments_list)
    
    transcription = {
        "text": " ".join(segment["text"] for segment in segments_list),
        "language": language,
        "confidence": round(np.exp(np.mean([segment.avg_log_prob for segment in segments])), 2),
        "segments": segments_list,
    }
    return format_whisper_timestamped_response(transcription, remove_punctuation_from_words=remove_punctuation_from_words)