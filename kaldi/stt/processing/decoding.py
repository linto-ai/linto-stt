import json
import re

from vosk import KaldiRecognizer, Model

from punctuation.recasepunc import apply_recasepunc

def decode(audio: tuple[bytes, int], model: Model, with_metadata: bool) -> dict:
    """Transcribe the audio data using the vosk library with the defined model."""
    decoder_result = {"text": "", "confidence-score": 0.0, "words": []}

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

    if "decoder_result" in decoder_result:
        decoder_result["words"] = [w for w in decoder_result["decoder_result"] if w["word"] != "<unk>"]
        if decoder_result["words"]:
            decoder_result["confidence-score"] = sum([w["conf"] for w in decoder_result["words"]]) / len(
                decoder_result["words"]
            )
    return decoder_result
