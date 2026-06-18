import io
import os
import re

import numpy as np
import wavio

SAMPLE_RATE = 16000  # whisper.audio.SAMPLE_RATE

import torch
import nemo.collections.asr as nemo_asr
import torchaudio


# Prompt-conditioned NeMo models (e.g. nvidia/nemotron-3.5-asr-streaming) emit
# inline language-id markers like "<fr-FR>" in BOTH offline and streaming output.
# These are special tokens, not real transcription, so we strip them. The region
# part covers ISO 3166 alpha-2/alpha-3 (2-4 uppercase, also an uppercased script
# code) and numeric-3 (e.g. "<es-419>"); the strict shape leaves ordinary
# angle-bracketed text untouched.
_LANGUAGE_MARKER_RE = re.compile(r"<[a-z]{2,3}-(?:[A-Z]{2,4}|[0-9]{3})>")


def strip_language_markers(text):
    """Remove inline language-id markers from `text` and tidy the whitespace."""
    if not text:
        return text
    text = _LANGUAGE_MARKER_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def supports_cache_aware_streaming(model):
    """Return True iff the model's encoder was trained for cache-aware streaming.

    The robust, trained-in signal is the Conformer encoder's ``att_context_style``:
    cache-aware streaming models use ``"chunked_limited"`` /
    ``"chunked_limited_with_rc"`` (limited, chunked attention with inter-chunk
    caching), whereas offline models use ``"regular"`` (full attention). Note that
    ``encoder.streaming_cfg`` is *always* set — even for offline models, where it
    computes a full-model lookahead — so it is NOT a valid discriminator; only
    ``att_context_style`` is.

    Such models expose the native streaming API used by NeMo's
    ``speech_to_text_cache_aware_streaming_infer.py`` example
    (``encoder.get_initial_cache_state`` + ``conformer_stream_step``), so we can
    serve them through that path instead of simulating streaming over an offline
    model.
    """
    encoder = getattr(model, "encoder", None)
    return getattr(encoder, "att_context_style", None) in (
        "chunked_limited",
        "chunked_limited_with_rc",
    )

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