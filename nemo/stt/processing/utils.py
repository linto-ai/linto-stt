import io
import os

import numpy as np
import wavio

SAMPLE_RATE = 16000  # whisper.audio.SAMPLE_RATE

import torch
import nemo.collections.asr as nemo_asr
import torchaudio


def has_cuda():
    return torch.cuda.is_available()


def get_device():
    device = os.environ.get("DEVICE", "cuda" if has_cuda() else "cpu")
    use_gpu = "cuda" in device
    try:
        device = torch.device(device)
    except Exception as err:
        raise Exception("Failed to set device: {}".format(str(err))) from err
    return device, use_gpu


def get_language(language = None):
    """
    Get the language from the environment variable LANGUAGE, and format as expected by Whisper.
    """
    if language is None:
        language = os.environ.get("LANGUAGE", None)
    if language is None:
        language = "fr" if "fr" in os.environ.get("MODEL", "nvidia/stt_fr_conformer_ctc_large").split("_") else "en"
    return language

def get_model_class(architecture):
    architecture = architecture.lower()
    model_class = nemo_asr.models.EncDecCTCModelBPE
    if architecture=="ctc":
        model_class = nemo_asr.models.EncDecCTCModel
    elif architecture=="hybrid_bpe" or architecture=="rnnt_ctc_bpe" or architecture=="hybrid_rnnt_ctc_bpe":
        model_class = nemo_asr.models.EncDecHybridRNNTCTCBPEModel
    elif architecture=="enc_dec" or architecture=="canary" or architecture=="multi_task":
        model_class = nemo_asr.models.EncDecMultiTaskModel
    return model_class
        

def conform_audio(audio, sample_rate=16_000):
    if sample_rate != SAMPLE_RATE:
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
    audio = whisper.load_audio(path)
    audio = torch.from_numpy(audio)
    return audio


def load_wave_buffer(file_buffer):
    """Formats audio from a wavFile buffer to a torch array for processing."""
    file_buffer_io = io.BytesIO(file_buffer)
    file_content = wavio.read(file_buffer_io)
    sample_rate = file_content.rate
    audio = file_content.data.astype(np.float32) / 32768
    audio = audio.transpose()
    audio = torch.from_numpy(audio)
    return conform_audio(audio, sample_rate)


def flatten(l):
    """
    flatten a list of lists
    """
    return [item for sublist in l for item in sublist]


LANGUAGES = {  # whisper.tokenizer.LANGUAGES
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}
