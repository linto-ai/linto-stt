import io

import numpy as np
import wavio
from scipy.signal import resample_poly

SAMPLE_RATE = 24000


def load_wave_buffer(file_buffer):
    """Load wave buffer and convert to float32 mono at 24kHz."""
    file_buffer_io = io.BytesIO(file_buffer)
    wav = wavio.read(file_buffer_io)
    data = wav.data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    sr = wav.rate
    if sr != SAMPLE_RATE:
        gcd = np.gcd(sr, SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // gcd, sr // gcd)
    data /= 32768.0
    return data, SAMPLE_RATE