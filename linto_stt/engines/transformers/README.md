# Transformers engine

Speech-to-text with Hugging Face `transformers`-native ASR models, through an
in-process `transformers` or `vllm` backend. The engine is named after the
library (not a single model): it auto-detects each checkpoint's family and
capabilities from its processor, so it supports several ASR models.

## Supported models

`MODEL` is any transformers-native ASR checkpoint; the family is detected from its
`model_type`.

| Model (`model_type`) | `MODEL` id(s) | vLLM backend | Auto language detect | Force language | Hotwords |
|---|---|:---:|:---:|:---:|:---:|
| **Qwen3-ASR** (`qwen3_asr`) | `Qwen/Qwen3-ASR-1.7B-hf` (default), `Qwen/Qwen3-ASR-0.6B-hf` | ✅ | ✅ | ✅ | ✅ |
| **Cohere Transcribe** (`cohere_asr`) | `CohereLabs/cohere-transcribe-03-2026` | ✅ \* | ❌ | ✅ | ❌ |

- **Word-level timestamps** (via the forced aligner) work for *every* model.
- **Capabilities are detected from the model's processor signature, not its
  name**, so a request asking for an unsupported feature (hotwords on a model
  without them, or auto-detection on a model that requires a language) is
  rejected with a clear error.
- For Qwen the `-hf` (transformers) variant is used, not the plain
  `Qwen/Qwen3-ASR-1.7B` repo, which targets the separate `qwen-asr` package.
- \* On vLLM, Cohere runs only on the `vllm` **serve** backend (it is an
  encoder-decoder); the in-process `vllm_offline` backend is Qwen-only. See
  **Backends** below.

All require **transformers >= 5.13** (the `transformers` / `vllm` extra); seq2seq
models (Cohere) also use `librosa` (included). Adding another transformers-native
ASR model means extending the family dispatch in `load_model.py` / `backends.py`;
candidates: the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
and the [LINAGORA ASR benchmark](https://linagora-labs.github.io/asr_benchmark/).

## Backends

The ASR step runs on a backend selected by `BACKEND`:

- `transformers` (default): in-process Hugging Face `transformers`. Supports every
  model (Qwen3-ASR + Cohere). Install with `uv sync --extra transformers`.
- `vllm`: a managed [vLLM](https://docs.vllm.ai) `vllm serve` **subprocess** that
  the engine starts and talks to over HTTP (`/v1/audio/transcriptions`), for
  higher throughput. One server per worker, started inside the container; no
  external service to run. Supports every vLLM-servable model, **including Cohere**
  (encoder-decoder). **CUDA GPU** only.
- `vllm_offline`: in-process vLLM (`vllm.LLM`). **Qwen3-ASR only** (it builds a
  decoder-only prompt), kept mainly to benchmark the in-process path against the
  `vllm` serve path. CUDA GPU only.

The two vLLM backends install with `uv sync --extra vllm` (vLLM is pinned to a
version that still allows the aligner's transformers). vLLM uses the non-`-hf`
checkpoint, so an `-hf` `MODEL` is mapped to its non-`-hf` id automatically. Tune
with `VLLM_GPU_MEMORY_UTILIZATION` (fraction of VRAM) or
`VLLM_KV_CACHE_MEMORY_BYTES` (absolute KV-cache budget, better on unified-memory /
shared GPUs), and `VLLM_MAX_MODEL_LEN` (context cap, kept within each model's own
limit). The serve backend also reads `VLLM_SERVE_HOST` / `VLLM_SERVE_PORT` and
`VLLM_SERVE_EXTRA_ARGS` (see `ENV.md`).

Everything else (language selection, hotwords, long-audio chunking, and word
timestamps, always via the transformers forced aligner) is shared, so the
backends can be benchmarked against each other by flipping `BACKEND`.

## Language

- `LANGUAGE=*` (or unset): automatic detection; the model transcribes
  multilingually (only for models that support it, see the table above).
- `LANGUAGE=fr` / `LANGUAGE=French`: force the language (ISO code or full
  English name).
- Overridable per request with the `language` query parameter (HTTP) or task
  argument.

The set of languages is model-specific:

| Model | # languages | Languages (model-card order, *not* a quality ranking) |
|---|:---:|---|
| **Qwen3-ASR** | 30 (+ 22 Chinese dialects) | `zh, en, yue, ar, de, fr, es, pt, id, it, ko, ru, th, vi, ja, tr, hi, ms, nl, sv, da, fi, pl, cs, fil, fa, el, hu, mk, ro` |
| **Cohere Transcribe** | 14 | `en, fr, de, it, es, pt, el, nl, pl, zh, ja, ko, vi, ar` |

The order follows each model card and is **not** a published best-to-worst
ranking (Qwen lists Chinese/English first; Cohere groups by region: European,
then APAC, then MENA). Neither card publishes a per-language quality order, so
consult each model card's per-language WER figures for relative accuracy.

Note: the forced aligner used for **word timestamps**
([`Qwen3-ForcedAligner-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf))
covers only 11 languages (`zh, en, yue, fr, de, it, ja, ko, pt, ru, es`).
Transcription of a language outside that set still works; its word timestamps may
be missing or approximate.

## Hotwords

`HOTWORDS` biases the transcription toward given words (proper nouns, jargon).
Give comma-separated entries (each may be a multi-word phrase, e.g.
`Quilter, Au revoir`) or, with no comma, whitespace-separated words. They are
formatted into Qwen3-ASR's documented context prompt (`Vocabulary: a, b, c.`)
and passed via the model's `prompt`. Overridable per request with the `hotwords`
query parameter / task argument.

Note: Qwen3-ASR's context prompt is soft LM conditioning (not literal token
biasing like faster-whisper's hotwords), and the model normalises casing, so
hotwords nudge *which* words are produced rather than forcing an exact spelling.

## Word timestamps

Qwen3-ASR has no native word timestamps, so they are produced by a companion
forced-alignment model, [`Qwen3-ForcedAligner-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)
(`ALIGNER_MODEL`), which aligns the transcript against the audio.

- The aligner is a **second model**. It is loaded lazily and run **only** when a
  request asks for word timestamps, i.e. HTTP `Accept: application/json` and the
  Celery task (both go through the metadata path). Plain-text requests
  (`Accept: text/plain`) never load or run it, so that fast path keeps the best
  RTF.
- Set `ALIGNER_MODEL=` (empty) or `ALIGNER_MODEL=none` to disable alignment; the
  `words` field is then returned empty for metadata requests.

## Running

```sh
uv sync --extra transformers
MODEL=Qwen/Qwen3-ASR-1.7B-hf DEVICE=cuda uv run main.py -m http -e transformers -p 8080 -i 127.0.0.1
# transcribe:
curl -s -X POST "http://127.0.0.1:8080/transcribe?language=fr" -F "file=@tests/bonjour.wav"
```

With the vLLM backend (GPU):

```sh
uv sync --extra vllm
MODEL=Qwen/Qwen3-ASR-1.7B-hf DEVICE=cuda BACKEND=vllm uv run main.py -m http -e transformers -p 8080 -i 127.0.0.1
```

Docker:

```sh
docker build -t linto-stt-transformers --build-arg STT_ENGINE=transformers --build-arg GPU=1 .
```

## Notes / limitations

- **GPU strongly recommended.** The model is an LLM-based ASR; CPU inference is
  slow.
- **No WebSocket/streaming** (it would need vLLM's realtime API, which is not
  wired here; the vLLM backends are used for offline transcription only).
- **Word timestamps** come from the forced-alignment model (see above), which
  runs only on metadata requests. Disable it with `ALIGNER_MODEL=none`.
- **Long audio**: files longer than `LONG_FILE_THRESHOLD` seconds are transcribed
  in consecutive non-overlapping chunks of `LONG_FILE_CHUNK_LEN` seconds (word
  timestamps offset per chunk), avoiding out-of-memory / truncation. A word
  spoken across a chunk boundary may be split there; keep chunks a few minutes
  long to make this rare.
- Efficiency: the default attention (`sdpa`) already uses Flash-Attention
  kernels on supported GPUs, so it is efficient out of the box. Other knobs:
  `DTYPE`, `MAX_NEW_TOKENS`, and `ATTN_IMPLEMENTATION=flash_attention_2` (opt-in;
  requires separately installing the GPU-only `flash-attn` package). See
  `ENV.md`.
