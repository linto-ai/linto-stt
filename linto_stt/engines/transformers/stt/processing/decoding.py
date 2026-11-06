import time

import torch

from linto_stt.engines.transformers.stt import (
    HOTWORDS,
    LONG_FILE_CHUNK_LEN,
    LONG_FILE_THRESHOLD,
    logger,
)

from .utils import SAMPLE_RATE, get_language, normalize_language_code


def _hotwords_prompt(hotwords):
    """Turn a hotwords string into Qwen3-ASR's documented context prompt.

    The model biases decoding toward a context prompt whose documented form is
    ``"Vocabulary: a, b, c."`` (see the Qwen3-ASR model card). Hotwords may be
    given as comma-separated entries (each entry may be a multi-word phrase, e.g.
    ``"Au revoir"``) or, when there is no comma, as whitespace-separated words.
    Returns None when there are no hotwords (so no prompt is sent).
    """
    if not hotwords:
        return None
    if "," in hotwords:
        entries = [e.strip() for e in hotwords.split(",")]
    else:
        entries = hotwords.split()
    entries = [e for e in entries if e]
    if not entries:
        return None
    return "Vocabulary: " + ", ".join(entries) + "."


def decode(audio, model, with_word_timestamps, language=None, hotwords=None):
    """Transcribe a mono 16 kHz float32 numpy array with a Qwen3-ASR model.

    - `language`: None / "*" lets the model auto-detect (and transcribe
      multilingually); a code ("fr") or name ("French") forces the language.
    - `hotwords`: free-text words/phrases to bias the transcription toward
      (proper nouns, jargon). Passed to the model as its context `prompt`.
      `None` falls back to the `HOTWORDS` env default; a falsy value disables
      biasing.
    - `with_word_timestamps`: when True, run the companion forced-alignment model
      to produce word-level timestamps (``words`` = list of {word, start, end}).
      When False, no alignment runs and ``words`` is the plain token list.

    Returns the standard STT dict: {text, language, confidence-score, words}.
    """
    language = get_language(language)
    if hotwords is None:
        hotwords = HOTWORDS

    # Validate the request against what this model can do (capabilities detected
    # from the processor, not the model name), with clear client errors.
    asr = model.asr
    if hotwords and not asr.supports_hotwords:
        raise ValueError(
            "This model does not support hotwords; unset HOTWORDS and the "
            "request `hotwords` parameter."
        )
    if not language and not asr.supports_language_auto:
        raise ValueError(
            "This model has no automatic language detection; set LANGUAGE or the "
            "request `language` parameter (e.g. 'fr', 'en')."
        )
    prompt = _hotwords_prompt(hotwords)

    duration = len(audio) / SAMPLE_RATE
    # Some backends (e.g. the Cohere seq2seq model) split long audio themselves,
    # so the engine must not also chunk them.
    if duration <= LONG_FILE_THRESHOLD or getattr(asr, "handles_long_audio", False):
        text, words, result_language = _transcribe_segment(
            model, audio, with_word_timestamps, language, prompt
        )
    else:
        logger.info(
            f"Audio is {duration:.0f}s (> {LONG_FILE_THRESHOLD}s): transcribing in "
            f"{LONG_FILE_CHUNK_LEN}s chunks"
        )
        text, words, result_language = _transcribe_long(
            model, audio, with_word_timestamps, language, prompt
        )

    return {
        "text": text,
        "language": result_language,
        "confidence-score": None,  # Qwen3-ASR does not expose a confidence
        "words": words,
    }


def _transcribe_segment(model, audio, with_word_timestamps, language, prompt):
    """Transcribe a single audio segment (short enough for one model pass).

    Returns (text, words, result_language). `language` is an ISO code or None,
    `prompt` the already-built context prompt (or None). The ASR step runs on the
    selected backend (`model.asr`); word timestamps always use the transformers
    forced aligner below.
    """
    start_t = time.time()
    text, detected = model.asr.transcribe(audio, language=language, prompt=prompt)
    # Normalise to an ISO code: a detected language comes back as a full English
    # name ("French"), a forced one is already a code ("fr") -- unify them.
    result_language = normalize_language_code(detected or language)
    logger.info(f"Transcription complete (t={time.time() - start_t}s)")

    if not with_word_timestamps:
        words = text.split()
    elif text and model.has_aligner:
        words = _align_words(model, audio, text, result_language or language)
    else:
        # Metadata requested but alignment is disabled or there is no text.
        words = []

    return text, words, result_language


def _transcribe_long(model, audio, with_word_timestamps, language, prompt):
    """Transcribe long audio in consecutive non-overlapping chunks and merge.

    Chunks are cut at fixed boundaries (no overlap), so the concatenated words
    still reconstruct the concatenated text. Word timestamps from each chunk are
    offset by that chunk's start time. A word spoken exactly across a cut may be
    split at the boundary; keep chunks a few minutes long to make this rare.
    """
    chunk_samples = LONG_FILE_CHUNK_LEN * SAMPLE_RATE
    texts = []
    words = []
    languages = []
    for start in range(0, len(audio), chunk_samples):
        segment = audio[start : start + chunk_samples]
        seg_text, seg_words, seg_language = _transcribe_segment(
            model, segment, with_word_timestamps, language, prompt
        )
        if seg_text:
            texts.append(seg_text)
        if seg_language:
            languages.append(seg_language)
        if with_word_timestamps:
            offset = start / SAMPLE_RATE
            for w in seg_words:
                w["start"] += offset
                w["end"] += offset
            words.extend(seg_words)

    text = " ".join(texts)
    if not with_word_timestamps:
        words = text.split()
    # Use the first chunk's detected language (forced language is identical
    # across chunks; auto-detect is normally consistent for one recording).
    result_language = languages[0] if languages else normalize_language_code(language)
    return text, words, result_language


def _align_words(model, audio, text, language):
    """Run the forced-alignment model to get word-level timestamps for `text`.

    Returns a list of {word, start, end} dicts (times in seconds) via the
    Qwen3-ForcedAligner protocol: prepare_forced_aligner_inputs() ->
    token-classification forward -> decode_forced_alignment().
    """
    processor = model.aligner_processor
    aligner = model.aligner_model

    start_t = time.time()
    inputs, word_lists = processor.prepare_forced_aligner_inputs(
        audio=audio, transcript=text, language=language
    )
    inputs = inputs.to(aligner.device, aligner.dtype)

    with torch.inference_mode():
        outputs = aligner(**inputs)

    items = processor.decode_forced_alignment(
        logits=outputs.logits,
        input_ids=inputs["input_ids"],
        word_lists=word_lists,
        timestamp_token_id=aligner.config.timestamp_token_id,
    )[0]
    logger.info(f"Forced alignment complete (t={time.time() - start_t}s)")

    # The aligner emits the "core" word tokens, usually stripped of the
    # surrounding punctuation the transcript carries (e.g. "Bonjour" for
    # "Bonjour.", or a separate "!" / "?" the aligner skips). Always return the
    # transcript's own whitespace tokens (so the words concatenate back to `text`,
    # keeping its exact spelling/punctuation) and assign the aligner timestamps to
    # the content tokens in order; punctuation-only tokens get a zero-width stamp.
    words = []
    j = 0
    last_end = 0.0
    for token in text.split():
        has_content = any(ch.isalnum() for ch in token)
        if has_content and j < len(items):
            start = float(items[j]["start_time"])
            end = float(items[j]["end_time"])
            last_end = end
            j += 1
        else:
            start = end = last_end  # punctuation, or more tokens than aligner words
        words.append({"word": token, "start": start, "end": end})
    return words
