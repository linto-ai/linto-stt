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

# Inference backend. Only "transformers" is implemented for now; "vllm" is a
# planned alternative (see decoding.py). Kept as a seam so the backend is
# swappable without touching the engine contract.
BACKEND = os.environ.get("BACKEND", "transformers").lower()

# Compute dtype of the model weights.
DTYPE = os.environ.get("DTYPE", "bfloat16")

# Upper bound on generated tokens per request. Short clips need little; long
# audio needs more (long-form chunking is handled in decoding). ~440 tokens
# covers a few minutes of dense speech.
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "440"))

# Attention kernel, e.g. "flash_attention_2" (needs flash-attn + fp16/bf16).
# None lets transformers pick its default ("sdpa").
ATTN_IMPLEMENTATION = os.environ.get("ATTN_IMPLEMENTATION") or None

# Default hotwords/keywords used to bias decoding (proper nouns, jargon). They
# are passed to the model as its context `prompt`. A falsy value means no
# biasing. Overridable per request via the `hotwords` query parameter / task
# argument (mirrors LANGUAGE/language).
HOTWORDS = os.environ.get("HOTWORDS") or None

# Companion forced-alignment model (Qwen3-ForcedAligner family) used to produce
# word-level timestamps. It is only loaded and run when a request asks for
# metadata (word timestamps); plain-text requests never touch it, so it adds no
# cost to the fast path. Set ALIGNER_MODEL empty / "none" to disable alignment
# (the `words` list is then returned empty for metadata requests).
ALIGNER_MODEL = os.environ.get("ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B-hf")
if ALIGNER_MODEL.strip().lower() in ("", "none", "false", "0"):
    ALIGNER_MODEL = None


def set_num_threads(n):
    torch.set_num_threads(n)


NUM_THREADS = os.environ.get("NUM_THREADS") or os.environ.get("OMP_NUM_THREADS") or None
if NUM_THREADS is None:
    NUM_THREADS = torch.get_num_threads()
NUM_THREADS = int(NUM_THREADS)

set_num_threads(1)
