import os

import whisper
from whisper.audio import SAMPLE_RATE
import whisper_timestamped

import numpy as np
import torch

from stt import logger
from .word_alignment import compute_alignment
from .text_normalize import remove_punctuation, normalize_text, remove_emoji
from .load_model import load_alignment_model, get_alignment_model

# This is to avoid hanging in a multi-threaded environment
torch.set_num_threads(1)


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

def decode(audio: torch.Tensor,
           model: whisper.model.Whisper,
           alignment_model: "Any",
           with_word_timestamps: bool,
           language: str = None,
           beam_size: int = None,
           best_of: int = None,
           temperature: float = 0.0,
           condition_on_previous_text: bool = False,
           no_speech_threshold: float = 0.6,
           logprob_threshold: float = -1.0,
           compression_ratio_threshold: float = 2.4,
           normalize_text_as_words=False,
           remove_punctuation_from_words=False,
           ) -> dict:
    """Transcribe the audio data using Whisper with the defined model."""

    fp16 = model.device != torch.device("cpu")

    if language is None:
        language = get_language()

    logger.info(f"Transcribing audio with language {language}...")

    kwargs = dict(
        language=language,
        fp16=fp16,
        temperature=temperature,
        beam_size=beam_size,
        best_of=best_of,
        condition_on_previous_text=condition_on_previous_text,
        no_speech_threshold=no_speech_threshold,
        logprob_threshold=logprob_threshold,
        compression_ratio_threshold=compression_ratio_threshold
    )

    if alignment_model is None:
        # Use Whisper cross-attention weights
        whisper_res = whisper_timestamped.transcribe(model, audio, **kwargs)
        if language is None:
            language = whisper_res["language"]
            logger.info(f"Detected language: {language}")
        return format_whisper_timestamped_response(whisper_res)

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
    result["confidence-score"] = np.exp(np.array([r["avg_logprob"] for r in segments])).mean() if len(segments) else 0.0

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

def format_whisper_timestamped_response(transcription):
    """Format Whisper response."""

    for i, seg in enumerate(transcription["segments"][:-1]):
        for expected_keys in ["start", "end", "words", "avg_logprob"]:
            assert expected_keys in seg, f"Missing '{expected_keys}' in segment {i} (that has keys {list(seg.keys())})"

    text = transcription["text"].strip()

    words = []

    segments = transcription.get("segments", [])

    for seg in segments:
        for word in seg.get("words", []):
            words.append({
                "word": word["text"],
                "start": word["start"],
                "end": word["end"],
                "conf": word["confidence"],
            })

    return {
        "text": text,
        "language": transcription["language"],
        "confidence-score": np.exp(np.array([r["avg_logprob"] for r in segments])).mean() if len(segments) else 0.0,
        "words": words,
    }