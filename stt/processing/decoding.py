import json

from vosk import KaldiRecognizer, Model

def decode(audio_data: bytes, model: Model, sampling_rate: int, with_metadata: bool) -> dict:
    ''' Transcribe the audio data using the vosk library with the defined model.'''
    result = {'text':'', 'words':[], 'confidence-score': 0.0}

    recognizer = KaldiRecognizer(model, sampling_rate, False)
    recognizer.SetMaxAlternatives(1)
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

    result['text'] = decoder_result['text'].strip()
    if 'words' in decoder_result:
        result['words'] = decoder_result['words']
    if 'confidence' in decoder_result:
        result['confidence-score'] = decoder_result['confidence']

    return result