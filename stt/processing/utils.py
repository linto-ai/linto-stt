from stt import USE_CTRANSLATE2, USE_TORCH, USE_TORCHAUDIO

import io
import wavio
import os
import numpy as np

SAMPLE_RATE = 16000 # whisper.audio.SAMPLE_RATE

if USE_CTRANSLATE2:
    import ctranslate2
    import faster_whisper
else:
    import torch
    import whisper

if USE_TORCHAUDIO:
    import torchaudio

def has_cuda():
    if USE_CTRANSLATE2:
        return ctranslate2.get_cuda_device_count() > 0
    else:
        return torch.cuda.is_available()

def get_device():
    device = os.environ.get("DEVICE", "cuda" if has_cuda() else "cpu")
    use_gpu = "cuda" in device
    
    # The following is to have GPU in the right order (as nvidia-smi show them)
    # But somehow it does not work with ctranslate2: 
    # see https://github.com/guillaumekln/faster-whisper/issues/150
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' # GPU in the right order
    
    if USE_CTRANSLATE2:
        try:
            if device.startswith("cuda:"):
                _ = [int(dev) for dev in device[5:].split(",")]
            else:
                assert device in ["cpu", "cuda"]
        except:
            raise ValueError(f"Invalid DEVICE '{device}' (should be 'cpu' or 'cuda' or 'cuda:<index> or 'cuda:<index1>,<index2>,...')")
    else:
        try:
            device = torch.device(device)
        except Exception as err:
            raise Exception("Failed to set device: {}".format(str(err))) from err
    return device, use_gpu

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
    # Convert French -> fr
    if isinstance(language, str) and language not in LANGUAGES:
        language = {v: k for k, v in LANGUAGES.items()}.get(language.lower(), language)
        # Raise an exception for unknown languages
        if language not in LANGUAGES:
            available_languages = \
                list(LANGUAGES.keys()) + \
                [k[0].upper() + k[1:] for k in LANGUAGES.values()] + \
                ["*", None]
            raise ValueError(f"Language '{language}' is not available. Available languages are: {available_languages}")
    return language

def conform_audio(audio, sample_rate=16_000):
    if sample_rate != SAMPLE_RATE:
        if not USE_TORCHAUDIO:
            raise NotImplementedError("Resampling not available without torchaudio")
        # Down or Up sample to the right sampling rate
        audio = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(audio)
    if audio.shape[0] > 1:
        # Stereo to mono
        # audio = torchaudio.transforms.DownmixMono()(audio, channels_first = True)
        audio = audio.mean(0)
    else:
        audio = audio.squeeze(0)
    return audio


def load_audiofile(path):
    if not os.path.isfile(path):
        raise RuntimeError("File not found: %s" % path)
    elif not os.access(path, os.R_OK):
        raise RuntimeError("Missing reading permission for: %s" % path)
    if USE_CTRANSLATE2:
        return faster_whisper.decode_audio(path, sampling_rate=SAMPLE_RATE)
    audio = whisper.load_audio(path)
    audio = torch.from_numpy(audio)
    return audio


def load_wave_buffer(file_buffer):
    """ Formats audio from a wavFile buffer to a torch array for processing. """
    file_buffer_io = io.BytesIO(file_buffer)
    if USE_CTRANSLATE2:
        return faster_whisper.decode_audio(file_buffer_io, sampling_rate=SAMPLE_RATE)
    file_content = wavio.read(file_buffer_io)
    sample_rate = file_content.rate
    audio = file_content.data.astype(np.float32)/32768
    audio = audio.transpose()
    audio = torch.from_numpy(audio)
    return conform_audio(audio, sample_rate)


def flatten(l):
    """
    flatten a list of lists
    """
    return [item for sublist in l for item in sublist]

LANGUAGES = { # whisper.tokenizer.LANGUAGES
    'en': 'english',
    'zh': 'chinese',
    'de': 'german',
    'es': 'spanish',
    'ru': 'russian',
    'ko': 'korean',
    'fr': 'french',
    'ja': 'japanese',
    'pt': 'portuguese',
    'tr': 'turkish',
    'pl': 'polish',
    'ca': 'catalan',
    'nl': 'dutch',
    'ar': 'arabic',
    'sv': 'swedish',
    'it': 'italian',
    'id': 'indonesian',
    'hi': 'hindi',
    'fi': 'finnish',
    'vi': 'vietnamese',
    'he': 'hebrew',
    'uk': 'ukrainian',
    'el': 'greek',
    'ms': 'malay',
    'cs': 'czech',
    'ro': 'romanian',
    'da': 'danish',
    'hu': 'hungarian',
    'ta': 'tamil',
    'no': 'norwegian',
    'th': 'thai',
    'ur': 'urdu',
    'hr': 'croatian',
    'bg': 'bulgarian',
    'lt': 'lithuanian',
    'la': 'latin',
    'mi': 'maori',
    'ml': 'malayalam',
    'cy': 'welsh',
    'sk': 'slovak',
    'te': 'telugu',
    'fa': 'persian',
    'lv': 'latvian',
    'bn': 'bengali',
    'sr': 'serbian',
    'az': 'azerbaijani',
    'sl': 'slovenian',
    'kn': 'kannada',
    'et': 'estonian',
    'mk': 'macedonian',
    'br': 'breton',
    'eu': 'basque',
    'is': 'icelandic',
    'hy': 'armenian',
    'ne': 'nepali',
    'mn': 'mongolian',
    'bs': 'bosnian',
    'kk': 'kazakh',
    'sq': 'albanian',
    'sw': 'swahili',
    'gl': 'galician',
    'mr': 'marathi',
    'pa': 'punjabi',
    'si': 'sinhala',
    'km': 'khmer',
    'sn': 'shona',
    'yo': 'yoruba',
    'so': 'somali',
    'af': 'afrikaans',
    'oc': 'occitan',
    'ka': 'georgian',
    'be': 'belarusian',
    'tg': 'tajik',
    'sd': 'sindhi',
    'gu': 'gujarati',
    'am': 'amharic',
    'yi': 'yiddish',
    'lo': 'lao',
    'uz': 'uzbek',
    'fo': 'faroese',
    'ht': 'haitian creole',
    'ps': 'pashto',
    'tk': 'turkmen',
    'nn': 'nynorsk',
    'mt': 'maltese',
    'sa': 'sanskrit',
    'lb': 'luxembourgish',
    'my': 'myanmar',
    'bo': 'tibetan',
    'tl': 'tagalog',
    'mg': 'malagasy',
    'as': 'assamese',
    'tt': 'tatar',
    'haw': 'hawaiian',
    'ln': 'lingala',
    'ha': 'hausa',
    'ba': 'bashkir',
    'jw': 'javanese',
    'su': 'sundanese'
}
