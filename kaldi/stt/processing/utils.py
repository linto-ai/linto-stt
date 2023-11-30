import io

import wavio
from numpy import int16, squeeze, mean


def load_audiofile(file_path):
    """Formats audio from a wavFile buffer to a bytebuffer"""
    audio = squeeze(wavio.read(file_path).data)
    return (audio.tobytes(), 16000)


def load_wave_buffer(file_buffer):
    """Formats audio from a wavFile buffer to a numpy array for processing."""
    file_buffer_io = io.BytesIO(file_buffer)
    file_content = wavio.read(file_buffer_io)
    # if stereo file, convert to mono by computing the mean over the channels
    if file_content.data.ndim == 2:
        if file_content.data.shape[1] == 1:
            data = squeeze(file_content.data)
        elif file_content.data.shape[1] == 2:
            data = mean(file_content.data, axis=1, dtype=int16)
        else:
            raise Exception("Audio Format not supported.")
        return data.tobytes(), file_content.rate
    raise Exception("Audio Format not supported.")
