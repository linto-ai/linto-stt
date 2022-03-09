import json
import re

from vosk import KaldiRecognizer, Model

def decode(audio_data: bytes, model: Model, sampling_rate: int, with_metadata: bool) -> dict:
    ''' Transcribe the audio data using the vosk library with the defined model.'''
    result = {'text':'', 'confidence-score': 0.0, 'words':[]}

    recognizer = KaldiRecognizer(model, sampling_rate)
    recognizer.SetMaxAlternatives(0) # Set confidence per words
    recognizer.SetWords(with_metadata)

    recognizer.AcceptWaveform(audio_data)
    try:
        decoder_result_raw = recognizer.FinalResult()
    except Exception as e:
        raise Exception("Failed to decode")
    try:
        decoder_result = json.loads(decoder_result_raw)
    except Exception:
        return result

    result["text"] = re.sub("<unk> " , "", decoder_result["text"])
    if "word" in decoder_result:
        result["words"] = [w for w in decoder_result["result"] if w["word"] != "<unk>"]
    if "confidence" in decoder_result:    
        result["confidence-score"] = sum([w["conf"] for w in words]) / len(words)
    return result