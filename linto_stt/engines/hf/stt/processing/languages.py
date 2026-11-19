"""Languages of an ASR model, and language identification (LID).

Two language sets are read from the model files (never from the model name):

- **accepted**: the languages the model itself enforces. A request for another
  language is rejected (HTTP 400), the same on every backend, and the language
  identifier only chooses among these. Sources: the Qwen3-ASR processor module
  (``LANGUAGE_CODE_TO_NAME``, 30 languages), the config's
  ``supported_languages`` (Cohere, 14), Whisper's tokenizer (100). Other models
  (e.g. Voxtral) take any code of Whisper's table, which is also the set vLLM's
  transcription endpoint accepts (it returns HTTP 400 for any other code).
- **recommended**: the languages listed in the model card metadata
  (``language:``), e.g. Voxtral's 8. Advisory only: a language outside them is
  transcribed, with a warning in the logs. None when the card cannot be read
  (offline, no README).

The language identifier is SpeechBrain's ECAPA model trained on VoxLingua107
(107 languages, Apache 2.0); see benchmark/lid/README.md for its evaluation.
"""

import os
import sys
import tempfile
import time

import torch

from linto_stt.engines.hf.stt import (
    LANGUAGE_IDENTIFICATION_MAX_DURATION,
    LANGUAGE_IDENTIFICATION_MODEL,
    logger,
)

from .utils import SAMPLE_RATE


def whisper_language_codes():
    """Whisper's 100 language codes: the codes vLLM's transcription endpoint
    accepts, whatever the model."""
    from transformers.models.whisper.tokenization_whisper import LANGUAGES

    return frozenset(LANGUAGES)


def detect_languages(model_name, processor):
    """(accepted, recommended) language codes of an ASR model; see the module
    docstring. `accepted` is never empty; `recommended` may be None."""
    accepted = _accepted_languages(model_name, processor)
    # Only real codes: Granite Speech's cards also list "multilingual".
    recommended = (_card_languages(model_name) or frozenset()) & accepted
    return accepted, recommended or None


def _accepted_languages(model_name, processor):
    if processor is not None:
        # Qwen3-ASR: the codes its processor maps to the names the model was
        # trained with (it rejects any other).
        module = sys.modules.get(type(processor).__module__)
        codes = getattr(module, "LANGUAGE_CODE_TO_NAME", None)
        if codes:
            return frozenset(codes)
    try:
        from transformers import AutoConfig

        codes = getattr(AutoConfig.from_pretrained(model_name), "supported_languages", None)
    except Exception as err:  # unreadable config: fall back to Whisper's table
        logger.warning(f"Could not read the config of {model_name}: {err}")
        codes = None
    if codes:
        return frozenset(codes)
    return whisper_language_codes()


def _card_languages(model_name):
    try:
        from huggingface_hub import ModelCard

        path = os.path.join(model_name, "README.md") if os.path.isdir(model_name) else model_name
        return parse_card_languages(ModelCard.load(path).data.language)
    except Exception:  # advisory only: never block the model
        return None


def parse_card_languages(languages):
    """The `language:` metadata of a model card as a set of codes, or None."""
    if isinstance(languages, str):
        languages = [languages]
    codes = set()
    for lang in languages or ():
        if lang is False:
            lang = "no"  # YAML reads Norwegian's bare `no` as a boolean (Whisper's card)
        if isinstance(lang, str) and lang.strip():
            codes.add(lang.strip().lower())
    return frozenset(codes) or None


def describe(codes, limit=30):
    codes = sorted(codes)
    return ", ".join(codes) if len(codes) <= limit else f"{len(codes)} languages"


class LanguageIdentifier:
    """SpeechBrain ECAPA (VoxLingua107) language identification."""

    SOURCE = LANGUAGE_IDENTIFICATION_MODEL  # Hugging Face ID or local folder
    # VoxLingua107 labels that differ from the codes the ASR models use.
    _ALIASES = {"iw": "he"}  # Hebrew: legacy code
    # Same language under another code in some models: Tagalog / Filipino.
    _EQUIVALENTS = {"tl": "fil"}

    def __init__(self, device):
        from speechbrain.inference.classifiers import EncoderClassifier

        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", 0)  # SpeechBrain wants an explicit index
        # The weights are downloaded to the Hugging Face cache (HF_HOME, offline
        # mode...). SpeechBrain also fills a "savedir" with *absolute* symlinks to
        # them, which break when the cache is seen under another path (a cache
        # shared between a host and a container): use a fresh temporary savedir,
        # created by this process, and removed as soon as the model is loaded
        # (everything is then in memory), so that a killed worker leaves nothing.
        # A local folder: its hyperparams.yaml still names the Hugging Face repository
        # (`pretrained_path`) to fetch the weights from; point it to the folder.
        overrides = {"pretrained_path": self.SOURCE} if os.path.isdir(self.SOURCE) else {}
        with tempfile.TemporaryDirectory(prefix="linto-stt-lid-") as savedir:
            self.model = EncoderClassifier.from_hparams(
                source=self.SOURCE, savedir=savedir, run_opts={"device": str(device)},
                overrides=overrides,
            )
        self.model.eval()
        self.device = device
        ind2lab = self.model.hparams.label_encoder.ind2lab
        # "fr: French" -> "fr", in the classifier's output order.
        self.codes = [
            self._ALIASES.get(c, c)
            for c in (ind2lab[i].split(":")[0].strip().lower() for i in range(len(ind2lab)))
        ]

    def _code_in(self, code, languages):
        """`code` as written in `languages` (possibly an equivalent), or None."""
        if code in languages:
            return code
        equivalent = self._EQUIVALENTS.get(code)
        return equivalent if equivalent in languages else None

    @torch.inference_mode()
    def identify(self, audio, languages=None):
        """Identify the language of a 16 kHz mono float32 array, on its first
        LANGUAGE_IDENTIFICATION_MAX_DURATION seconds.

        Returns (code, best): `code` is the most likely language among
        `languages` (all 107 when None, or when none of them is known to the
        classifier), `best` the most likely one among all 107."""
        audio = audio[: int(LANGUAGE_IDENTIFICATION_MAX_DURATION * SAMPLE_RATE)]
        wav = torch.as_tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        log_probs = self.model.classify_batch(wav)[0][0].float().cpu()
        best = self.codes[int(log_probs.argmax())]
        if not languages:
            return best, best
        candidates = [
            (float(log_probs[i]), self._code_in(code, languages))
            for i, code in enumerate(self.codes)
            if self._code_in(code, languages)
        ]
        if not candidates:
            return best, best
        return max(candidates)[1], best


def identify_language(identifier, audio, accepted, recommended):
    """Run the language identifier for a request without a language; returns the
    language to use, among the `accepted` ones, with warnings when it is
    uncertain (best guess not accepted) or outside the `recommended` ones."""
    start_t = time.time()
    code, best = identifier.identify(audio, accepted)
    logger.info(f"Identified language: {code} (t={time.time() - start_t:.3f}s)")
    if accepted and identifier._code_in(best, accepted) is None:
        logger.warning(
            f"The most likely language ({best}) is not supported by the model "
            f"({describe(accepted)}): using {code}, the most likely among them"
        )
    if recommended and code not in recommended:
        logger.warning(
            f"Identified language {code} is not among the model's documented languages "
            f"({describe(recommended)}): the transcription may be poor"
        )
    return code
