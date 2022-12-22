import io
import wavio
import os
import numpy as np
import torch
import torchaudio
import whisper

def conform_audio(audio, sample_rate = 16_000):
    if sample_rate != whisper.audio.SAMPLE_RATE:
        # Down or Up sample to the right sampling rate
        audio = torchaudio.transforms.Resample(sample_rate, whisper.audio.SAMPLE_RATE)(audio)
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
    # audio, sample_rate = torchaudio.load(path)
    # return conform_audio(audio, sample_rate)
    audio = whisper.load_audio(path)
    audio = torch.from_numpy(audio)
    return audio


def load_wave_buffer(file_buffer):
    """ Formats audio from a wavFile buffer to a torch array for processing. """
    file_buffer_io = io.BytesIO(file_buffer)
    file_content = wavio.read(file_buffer_io)
    sample_rate = file_content.rate
    audio = torch.from_numpy(file_content.data.astype(np.float32)/32768)
    audio = audio.transpose(0,1)
    return conform_audio(audio, sample_rate)

def flatten(l):
    """
    flatten a list of lists
    """
    return [item for sublist in l for item in sublist]
