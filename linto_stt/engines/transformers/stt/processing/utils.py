import io
import os

import numpy as np
import torch
import torchaudio
import wavio

SAMPLE_RATE = 16000


def has_cuda():
    return torch.cuda.is_available()


def get_device():
    device = os.environ.get("DEVICE") or ("cuda" if has_cuda() else "cpu")
    use_gpu = "cuda" in device
    try:
        device = torch.device(device)
    except Exception as err:
        raise Exception(f"Failed to set device: {err}") from err
    return device, use_gpu


def get_language(language=None):
    """Resolve the request/LANGUAGE-env language into the form Qwen3-ASR expects.

    Returns a short ISO code (e.g. "fr") or None for automatic detection.
    Accepts an ISO code ("fr"), a BCP-47 tag ("fr-FR") or a full English name
    ("French"), as documented for `LANGUAGE`; `*` / unset means auto-detect.
    Always normalise to the code: the vLLM transcription endpoint only accepts
    ISO 639-1 codes, so a full name would be rejected there.
    """
    if language is None:
        language = os.environ.get("LANGUAGE", "*")
    return normalize_language_code(language)


# Full English language names (lowercased) -> ISO code, for the languages the
# current Transformers ASR model (Qwen3-ASR) can report as the detected language.
# Self-contained on purpose (no reliance on transformers internals). Extend it if
# a future model reports other languages; unknown values fall back gracefully.
_LANGUAGE_NAME_TO_CODE = {
    "arabic": "ar", "cantonese": "yue", "chinese": "zh", "czech": "cs",
    "danish": "da", "dutch": "nl", "english": "en", "filipino": "fil",
    "finnish": "fi", "french": "fr", "german": "de", "greek": "el",
    "hindi": "hi", "hungarian": "hu", "indonesian": "id", "italian": "it",
    "japanese": "ja", "korean": "ko", "macedonian": "mk", "malay": "ms",
    "persian": "fa", "polish": "pl", "portuguese": "pt", "romanian": "ro",
    "russian": "ru", "spanish": "es", "swedish": "sv", "thai": "th",
    "turkish": "tr", "vietnamese": "vi",
}


def language_code_to_name(code):
    """ISO code -> the full English language name Qwen3-ASR uses (e.g. "fr" ->
    "French"), or None if unknown. Inverse of `_LANGUAGE_NAME_TO_CODE`; used to
    force the language in the vLLM prompt prefill (`language <Name><asr_text>`)."""
    if not code:
        return None
    code = code.lower()
    for name, c in _LANGUAGE_NAME_TO_CODE.items():
        if c == code:
            return name.title()
    return None


def normalize_language_code(language):
    """Normalise a language string to a short ISO code; None/empty/`*` -> None.

    The model reports a detected language as a full English name (e.g. "French"),
    while a forced language is already a code ("fr"); this maps both to the code
    so the response's `language` field is consistent. A known full English name
    maps to its code; anything else is lowercased and stripped of a region suffix
    ("fr-FR" -> "fr").
    """
    if not language:
        return None
    lang = language.strip()
    if not lang or lang in ("*", "None", "none"):
        return None
    return _LANGUAGE_NAME_TO_CODE.get(lang.lower()) or lang.split("-")[0].lower()


def conform_audio(audio, sample_rate=SAMPLE_RATE):
    """Resample to 16 kHz and downmix to mono; `audio` is a (channels, samples)
    tensor, the return is a 1-D tensor."""
    if sample_rate != SAMPLE_RATE:
        audio = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(0)  # stereo -> mono
    else:
        audio = audio.squeeze(0)
    return audio


def _decode_audio_file(path):
    """Decode an audio file to a (channels, samples) float32 tensor + sample rate.

    Prefer soundfile (libsndfile, bundled in its wheel): it needs no system
    ffmpeg and avoids torchaudio.load, which on torch >= 2.11 routes through
    torchcodec whose native library fails to load without a matching ffmpeg. Fall
    back to torchaudio for any format libsndfile cannot read.
    """
    try:
        import soundfile as sf

        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        # soundfile returns (frames, channels); conform_audio wants (channels, frames)
        return torch.from_numpy(data.T.copy()), sample_rate
    except (ImportError, RuntimeError):
        # RuntimeError covers soundfile.LibsndfileError (unsupported format).
        audio, sample_rate = torchaudio.load(path)
        return audio, sample_rate


def load_audiofile(path):
    """Load an audio file as a 16 kHz mono float32 numpy array (task mode)."""
    if not os.path.isfile(path):
        raise RuntimeError(f"File not found: {path}")
    if not os.access(path, os.R_OK):
        raise RuntimeError(f"Missing reading permission for: {path}")
    audio, sample_rate = _decode_audio_file(path)
    return conform_audio(audio, sample_rate).numpy()


def load_wave_buffer(file_buffer):
    """Format audio from a WAV buffer to a 16 kHz mono float32 numpy array
    (HTTP mode)."""
    file_content = wavio.read(io.BytesIO(file_buffer))
    audio = file_content.data.astype(np.float32) / 32768
    audio = torch.from_numpy(audio.transpose())  # (samples, channels) -> (channels, samples)
    return conform_audio(audio, file_content.rate).numpy()
