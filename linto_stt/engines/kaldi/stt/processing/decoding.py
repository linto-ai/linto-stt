import json
from typing import Tuple

from vosk import KaldiRecognizer, Model

from linto_stt.engines.kaldi.stt import logger

try:
    from linto_stt.punctuation.recasepunc import apply_recasepunc
except ImportError:
    def apply_recasepunc(config, text, **kw):
        return text

def decode(audio: Tuple[bytes, int], model: Model, with_metadata: bool, language=None, hotwords=None) -> dict:
    """Transcribe the audio data using the vosk library with the defined model."""
    decoder_result = {"text": "", "confidence-score": 0.0, "words": []}

    if language and language != "*":
        # A Kaldi model is monolingual: the language of a request (always sent by
        # the transcription service, for instance) cannot change it.
        logger.warning(f"Language {language!r} ignored: a Kaldi model transcribes its own language")
    
    audio_data, sampling_rate = audio

    model, punctuation_model = model

    recognizer = KaldiRecognizer(model, sampling_rate)
    recognizer.SetMaxAlternatives(0)  # Set confidence per words
    recognizer.SetWords(with_metadata)

    recognizer.AcceptWaveform(audio_data)
    try:
        decoder_result_raw = recognizer.FinalResult()
    except Exception as err:
        raise Exception("Failed to decode") from err
    try:
        decoder_result = json.loads(decoder_result_raw)
    except Exception:
        return decoder_result

    decoder_result = apply_recasepunc(punctuation_model, decoder_result)

    # Vosk lists the words (with SetWords) under "result"; the API calls them "words".
    words = [w for w in decoder_result.pop("result", []) if w["word"] != "<unk>"]
    decoder_result["words"] = words
    decoder_result["confidence-score"] = sum(w["conf"] for w in words) / len(words) if words else 0.0
    return decoder_result
