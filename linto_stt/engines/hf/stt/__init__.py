import logging
import os

import torch

logging.basicConfig(
    format="[%(asctime)s,%(msecs)03d %(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("__stt__")

# Order GPUs as nvidia-smi shows them (set before any CUDA init).
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# --- Qwen3-ASR decoding knobs (documented in ENV.md and the engine README) ---

# Inference backend (see backends.py). One of:
#   "vllm"          (default) a managed `vllm serve` subprocess, over HTTP
#                   (Qwen3-ASR + Cohere); higher throughput, GPU + `hf-vllm` extra.
#   "transformers"  in-process Hugging Face transformers (Qwen3-ASR + Cohere);
#                   the portable fallback (CPU-capable, `hf` extra).
#   "vllm_offline"  in-process `vllm.LLM` (Qwen3-ASR only); undocumented on
#                   purpose (slower than "vllm" beyond 20 s of audio, fewer
#                   models), kept only to benchmark against the "vllm" serve
#                   backend: see the `VllmASRBackend` docstring in backends.py.
# A seam so the backend is swappable without touching the engine contract.
BACKEND = os.environ.get("BACKEND", "vllm").lower()

# Compute dtype of the model weights.
DTYPE = os.environ.get("DTYPE", "bfloat16")

# Cap on generated tokens per pass (per chunk for long audio). It is only a
# safety cap: generation stops at the end-of-speech token, so a generous value
# costs nothing for short audio but avoids truncating a dense chunk. ~1280 covers
# a dense 4-minute chunk (the default chunk length below).
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1280"))

# Long-audio handling. VRAM grows ~linearly with audio length and MAX_NEW_TOKENS
# caps the transcript, so audio longer than LONG_FILE_THRESHOLD seconds is
# transcribed in consecutive (non-overlapping) chunks of LONG_FILE_CHUNK_LEN
# seconds and the per-chunk results concatenated (word timestamps offset by each
# chunk's start time). The 4-minute default keeps the per-pass footprint modest
# (~8 GB for the 1.7B model, both models loaded) and stays under the forced
# aligner's documented ~5-minute limit. Raise all three (threshold, chunk len,
# MAX_NEW_TOKENS) together for fewer/larger passes on a GPU with spare VRAM.
LONG_FILE_THRESHOLD = int(os.environ.get("LONG_FILE_THRESHOLD", "240"))
LONG_FILE_CHUNK_LEN = int(os.environ.get("LONG_FILE_CHUNK_LEN", "240"))

# Attention kernel, e.g. "flash_attention_2" (needs flash-attn + fp16/bf16).
# None lets transformers pick its default ("sdpa").
ATTN_IMPLEMENTATION = os.environ.get("ATTN_IMPLEMENTATION") or None

# Default hotwords/keywords used to bias decoding (proper nouns, jargon). They
# are passed to the model as its context `prompt`. A falsy value means no
# biasing. Overridable per request via the `hotwords` query parameter / task
# argument (mirrors LANGUAGE/language).
HOTWORDS = os.environ.get("HOTWORDS") or None

# Companion forced aligner used to produce word-level timestamps: a
# Qwen3-ForcedAligner checkpoint, or "mms" (Meta's language-agnostic MMS CTC
# aligner, see processing/aligners.py). It is only loaded and run when a request
# asks for metadata (word timestamps); plain-text requests never touch it, so it
# adds no cost to the fast path. Set ALIGNER_MODEL empty / "none" to disable
# alignment (the `words` list is then returned empty for metadata requests).
# ALIGNER_FALLBACK_MODEL names a second aligner, loaded on first need, used for
# the requests the primary one does not cover (a language outside its set, audio
# over its duration limit): "mms" by default behind the Qwen aligner; "none" to
# always use the primary aligner.


def _optional_model(name):
    return None if name.strip().lower() in ("", "none", "false", "0") else name


ALIGNER_MODEL = _optional_model(
    os.environ.get("ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B-hf")
)
ALIGNER_FALLBACK_MODEL = _optional_model(os.environ.get("ALIGNER_FALLBACK_MODEL", "mms"))


def set_num_threads(n):
    torch.set_num_threads(n)


NUM_THREADS = os.environ.get("NUM_THREADS") or os.environ.get("OMP_NUM_THREADS") or None
if NUM_THREADS is None:
    NUM_THREADS = torch.get_num_threads()
NUM_THREADS = int(NUM_THREADS)

set_num_threads(1)
