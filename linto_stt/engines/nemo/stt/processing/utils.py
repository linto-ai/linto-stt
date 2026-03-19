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
    device = os.environ.get("DEVICE") or ("cuda" if has_cuda() else "cpu")
    use_gpu = "cuda" in device
    try:
        device = torch.device(device)
    except Exception as err:
        raise Exception("Failed to set device: {}".format(str(err))) from err
    return device, use_gpu


def get_language(language = None):
    """
    Get the language from the environment variable LANGUAGE, and format as expected by NeMo (if supported).
    """
    if language is None:
        language = os.environ.get("LANGUAGE", "*")
    # "fr-FR" -> "fr" (language-country code to ISO 639-1 code)
    language_fields = language.split("-")
    if len(language_fields) == 2:
        language = language_fields[0]
    # "*" means "all languages"
    if language == "*":
        language = None
    if language is None:
        language = "unknown"
    return language
        
def get_decoding_method(architecture):
    architecture = architecture.lower()
    if "hybrid" in architecture:
        return "ctc" if "ctc" in architecture else "rnnt"
    else:
        return None

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
    audio, _ = torchaudio.load(path)
    audio = audio.squeeze().numpy()
    return audio


def load_wave_buffer(file_buffer):
    """Formats audio from a wavFile buffer to a torch array for processing."""
    file_buffer_io = io.BytesIO(file_buffer)
    file_content = wavio.read(file_buffer_io)
    sample_rate = file_content.rate
    audio = file_content.data.astype(np.float32) / 32768
    audio = audio.transpose()
    audio = torch.from_numpy(audio)
    return conform_audio(audio, sample_rate).numpy()


def flatten(l):
    """
    flatten a list of lists
    """
    return [item for sublist in l for item in sublist]