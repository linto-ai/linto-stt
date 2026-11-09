"""Forced aligners: word-level timestamps for an existing transcript.

Pluggable like the ASR backends. ``ALIGNER_MODEL`` selects the primary aligner
and ``ALIGNER_FALLBACK_MODEL`` an optional second one, used for the requests the
primary does not cover: a language outside its set, or audio longer than its
limit (see ``LazyLoadedModel.aligner_for``):

- a Qwen3-ForcedAligner checkpoint (default ``Qwen/Qwen3-ForcedAligner-0.6B-hf``):
  accurate, tuned for 11 languages, up to 5 minutes of audio per pass;
- ``mms``: Meta's MMS CTC aligner (torchaudio's ``MMS_FA`` bundle, 300M
  wav2vec2 trained on 1,100+ languages). Language-agnostic and without a
  duration limit: the transcript is romanised with ``uroman`` and aligned
  character by character over windowed emissions.

Every aligner exposes ``name``, ``covers(language, duration)`` and
``align(audio, text, language) -> [{word, start, end}, ...]`` whose words are
exactly ``text.split()`` (so they concatenate back to the transcript).
"""

import torch

from linto_stt.engines.hf.stt import logger

from .load_model import _DTYPES
from .utils import SAMPLE_RATE


def load_aligner(model_name, device, dtype="bfloat16"):
    """Instantiate the aligner named by `model_name`: ``mms`` or a
    Qwen3-ForcedAligner(-family) checkpoint id / path."""
    if model_name.strip().lower() == "mms":
        return MMSForcedAligner(device)
    return QwenForcedAligner(model_name, device, dtype)


class QwenForcedAligner:
    """Qwen3-ForcedAligner: a token-classification model fed with the audio and
    the transcript's words, which predicts a timestamp token per word
    (prepare_forced_aligner_inputs -> forward -> decode_forced_alignment)."""

    name = "qwen"
    # The model card states "up to 5 minutes of speech" per pass.
    max_duration = 300.0

    def __init__(self, model_name, device, dtype="bfloat16"):
        from transformers import AutoModelForTokenClassification, AutoProcessor

        device_map = "auto" if "cuda" in str(device) else str(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, dtype=_DTYPES.get(dtype, torch.bfloat16), device_map=device_map
        ).eval()
        self.languages = _qwen_aligner_languages()

    def covers(self, language, duration=None):
        """True for the languages the aligner is tuned for (and for auto-detect
        with no known language, where the primary aligner is used as is) and
        for audio within its documented duration limit."""
        if duration is not None and duration > self.max_duration:
            return False
        return language is None or language in self.languages

    def align(self, audio, text, language=None):
        processor, model = self.processor, self.model
        try:
            inputs, word_lists = processor.prepare_forced_aligner_inputs(
                audio=audio, transcript=text, language=language
            )
        except ValueError as err:
            if language is None or "not supported by the forced aligner" not in str(err):
                raise
            # The aligner is tuned for 11 languages, but the language is only a
            # hint for splitting the transcript into words (dedicated
            # Japanese/Korean tokenizers; otherwise CJK characters and
            # whitespace-separated words): the same model aligns any transcript.
            # So align without the hint rather than refuse word timestamps.
            logger.warning(
                f"Language {language!r} is outside the forced aligner's tuned set; "
                f"aligning without a language hint (accuracy not validated)"
            )
            inputs, word_lists = processor.prepare_forced_aligner_inputs(
                audio=audio, transcript=text, language=None
            )
        inputs = inputs.to(model.device, model.dtype)

        with torch.inference_mode():
            outputs = model(**inputs)

        items = processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=model.config.timestamp_token_id,
        )[0]

        # The aligner emits the "core" word tokens, usually stripped of the
        # surrounding punctuation the transcript carries (e.g. "Bonjour" for
        # "Bonjour.", or a separate "!" / "?" the aligner skips). Always return
        # the transcript's own whitespace tokens and assign the aligner timestamps
        # to the content tokens in order; punctuation-only tokens get a zero-width
        # stamp at the previous end.
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


def _qwen_aligner_languages():
    """ISO codes of the languages the Qwen3-ForcedAligner is tuned for, read from
    transformers when available (kept in sync with its own validation)."""
    try:
        from transformers.models.qwen3_asr.processing_qwen3_asr import (
            FORCED_ALIGNER_LANGUAGES,
            LANGUAGE_CODE_TO_NAME,
        )

        return frozenset(
            code for code, name in LANGUAGE_CODE_TO_NAME.items() if name in FORCED_ALIGNER_LANGUAGES
        )
    except ImportError:
        return frozenset({"zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"})


class MMSForcedAligner:
    """Meta MMS forced aligner (torchaudio ``MMS_FA``): a wav2vec2 CTC model over
    28 romanised characters, aligned with CTC forced alignment (Viterbi). Covers
    any language: each word is romanised with uroman (e.g. "Привет" -> "privet",
    "ça" -> "sa") and lowercased, characters outside the dictionary are dropped.

    The weights (1.2 GB, float32) are fetched by torchaudio into the torch hub
    cache (``TORCH_HOME``), not the Hugging Face cache.

    Emissions are computed on fixed windows (``EMISSION_WINDOW_S``) and
    concatenated: wav2vec2's attention is global, so one pass over long audio
    costs quadratic memory and time, while per-window emissions are local
    enough for alignment (the usual practice for CTC aligners). The CTC trellis
    then runs once over the whole sequence, so there is no hard limit on the
    audio duration."""

    name = "mms"
    max_duration = None  # windowed emissions: no limit
    EMISSION_WINDOW_S = 30

    def __init__(self, device):
        import torchaudio
        import uroman

        bundle = torchaudio.pipelines.MMS_FA
        self.device = torch.device(device)
        self.model = bundle.get_model(with_star=False).to(self.device).eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        # Characters a word may contain after romanisation. The dictionary's
        # index 0 ("-") is the CTC blank, which must never appear in a target,
        # so hyphens are dropped like any other character outside the set.
        blank = getattr(self.aligner, "blank", 0)
        self.characters = {ch for ch, idx in bundle.get_dict(star=None).items() if idx != blank}
        self.uroman = uroman.Uroman()
        assert bundle.sample_rate == SAMPLE_RATE

    def covers(self, language, duration=None):
        return True

    def _normalize(self, token):
        romanized = self.uroman.romanize_string(token).lower()
        return "".join(ch for ch in romanized if ch in self.characters)

    def _emission(self, waveform):
        """Log-probabilities over the character dictionary, (frames, chars), for
        the whole (1, samples) waveform, computed window by window."""
        window = self.EMISSION_WINDOW_S * SAMPLE_RATE
        starts = list(range(0, waveform.size(1), window))
        bounds = [(start, start + window) for start in starts]
        if len(bounds) > 1 and waveform.size(1) - bounds[-1][0] < SAMPLE_RATE:
            # Fold a trailing window shorter than 1s into the previous one.
            bounds[-2] = (bounds[-2][0], waveform.size(1))
            bounds.pop()
        chunks = []
        with torch.inference_mode():
            for start, end in bounds:
                emission, _ = self.model(waveform[:, start:end].to(self.device))
                chunks.append(emission[0])
        return torch.cat(chunks)

    def align(self, audio, text, language=None):
        tokens = text.split()
        normalized = [self._normalize(token) for token in tokens]
        alignable = [i for i, norm in enumerate(normalized) if norm]
        spans = []
        if alignable:
            waveform = torch.as_tensor(audio, dtype=torch.float32).reshape(1, -1)
            emission = self._emission(waveform)  # the CTC trellis runs on its device
            seconds_per_frame = waveform.size(1) / emission.size(0) / SAMPLE_RATE
            token_spans = self.aligner(
                emission, self.tokenizer([normalized[i] for i in alignable])
            )
            spans = [
                (span[0].start * seconds_per_frame, span[-1].end * seconds_per_frame)
                for span in token_spans
            ]

        # Same contract as the Qwen aligner: one entry per whitespace token;
        # tokens with nothing to align (punctuation) get a zero-width stamp.
        times = dict(zip(alignable, spans))
        words = []
        last_end = 0.0
        for i, token in enumerate(tokens):
            if i in times:
                start, end = times[i]
                last_end = end
            else:
                start = end = last_end
            words.append({"word": token, "start": start, "end": end})
        return words
