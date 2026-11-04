import time

import torch

from linto_stt.engines.transformers.stt import BACKEND, HOTWORDS, MAX_NEW_TOKENS, logger

from .utils import get_language, normalize_language_code


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
    if BACKEND != "transformers":
        raise NotImplementedError(
            f"BACKEND={BACKEND!r} is not implemented yet "
            "(only 'transformers' is available)."
        )

    processor = model.processor
    gen_model = model.model

    language = get_language(language)
    if hotwords is None:
        hotwords = HOTWORDS

    request_kwargs = {}
    if language:
        request_kwargs["language"] = language
    prompt = _hotwords_prompt(hotwords)
    if prompt:
        # Qwen3-ASR biases decoding toward the words supplied in its context
        # `prompt`. This is how hotwords are implemented for this engine.
        request_kwargs["prompt"] = prompt

    start_t = time.time()
    # Qwen3ASRProcessor builds the chat/transcription request; the feature
    # extractor expects 16 kHz audio, which load_wave_buffer / load_audiofile
    # already guarantee.
    inputs = processor.apply_transcription_request(audio=audio, **request_kwargs)
    inputs = inputs.to(gen_model.device, gen_model.dtype)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = gen_model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    generated_ids = output_ids[:, input_len:]
    # return_format="parsed" yields {transcription, language, ...} per sequence,
    # so we recover both the text and the language the model actually used
    # (important when language is auto-detected — we need it for alignment and
    # for the response).
    parsed = processor.decode(generated_ids, return_format="parsed")[0]
    text = (parsed.get("transcription") or "").strip()
    # Normalise to an ISO code: a detected language comes back as a full English
    # name ("French"), a forced one is already a code ("fr") -- unify them.
    result_language = normalize_language_code(parsed.get("language") or language)
    logger.info(f"Transcription complete (t={time.time() - start_t}s)")

    if not with_word_timestamps:
        words = text.split()
    elif text and model.has_aligner:
        words = _align_words(model, audio, text, result_language or language)
    else:
        # Metadata requested but alignment is disabled or there is no text.
        words = []

    return {
        "text": text,
        "language": result_language,
        "confidence-score": None,  # Qwen3-ASR does not expose a confidence
        "words": words,
    }


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

    # The aligner emits the "core" word tokens, typically stripped of the
    # surrounding punctuation the transcript carries (e.g. "Bonjour" for
    # "Bonjour."). Re-attach the original transcript tokens when they line up 1:1
    # with the aligner segments, so the returned words keep the transcript's
    # exact spelling/punctuation and concatenate back to `text`. Fall back to the
    # aligner tokens otherwise (e.g. languages without whitespace word
    # boundaries, where the counts won't match).
    original_words = text.split()
    use_original = len(original_words) == len(items)
    return [
        {
            "word": original_words[i] if use_original else item["text"],
            "start": float(item["start_time"]),
            "end": float(item["end_time"]),
        }
        for i, item in enumerate(items)
    ]
