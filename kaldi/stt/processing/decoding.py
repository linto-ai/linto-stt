import json
import re

from vosk import KaldiRecognizer, Model

from punctuation.recasepunc import load_model, apply_recasepunc

_PUNCTUATION_MODEL = load_model()

def decode(audio: tuple[bytes, int], model: Model, with_metadata: bool) -> dict:
    """Transcribe the audio data using the vosk library with the defined model."""
    result = {"text": "", "confidence-score": 0.0, "words": []}

    audio_data, sampling_rate = audio

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
        return result

    if _PUNCTUATION_MODEL:
        result = apply_recasepunc(_PUNCTUATION_MODEL, result)

    if "result" in decoder_result:
        result["words"] = [w for w in decoder_result["result"] if w["word"] != "<unk>"]
        if result["words"]:
            result["confidence-score"] = sum([w["conf"] for w in result["words"]]) / len(
                result["words"]
            )
    return result
