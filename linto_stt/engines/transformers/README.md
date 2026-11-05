# Transformers engine

Speech-to-text with Hugging Face `transformers`-native ASR models. Currently
built around [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf), a
multilingual ASR model; the engine is named after the `transformers` library so
it can grow to cover other transformers-native ASR checkpoints.

## Models

Set `MODEL` to a transformers-native Qwen3-ASR checkpoint (checkpoint model type
`qwen3_asr`):

- `Qwen/Qwen3-ASR-1.7B-hf` (default)
- `Qwen/Qwen3-ASR-0.6B-hf`

These require **transformers >= 5.13** (installed by the `transformers` extra). The
`-hf` (transformers) variant is used — not the plain `Qwen/Qwen3-ASR-1.7B` repo,
which targets the separate `qwen-asr` package.

## Backends

The ASR step runs on a backend selected by `BACKEND`:

- `transformers` (default) — in-process Hugging Face `transformers`. Install with
  `uv sync --extra transformers`.
- `vllm` — in-process [vLLM](https://docs.vllm.ai) (`vllm.LLM`), for higher
  throughput. Install with `uv sync --extra vllm` (vLLM is pinned to a version
  that still allows the aligner's transformers) and run on a **CUDA GPU**. vLLM
  uses the non-`-hf` checkpoint, so an `-hf` `MODEL` is mapped to its non-`-hf`
  id automatically. Tunable via `VLLM_GPU_MEMORY_UTILIZATION` (fraction of VRAM)
  or `VLLM_KV_CACHE_MEMORY_BYTES` (absolute KV-cache budget, better on
  unified-memory / shared GPUs), and `VLLM_MAX_MODEL_LEN` (see `ENV.md`).

Everything else — language selection, hotwords, long-audio chunking, and word
timestamps (always via the transformers forced aligner) — is shared, so the two
backends can be benchmarked against each other by flipping `BACKEND`.

## Language

- `LANGUAGE=*` (or unset) → automatic detection; the model transcribes
  multilingually.
- `LANGUAGE=fr` / `LANGUAGE=French` → force the language (ISO code or full
  English name).
- Overridable per request with the `language` query parameter (HTTP) or task
  argument.

Supported languages (30): `fr, en, zh, de, es, pt, id, it, ko, ru, th, vi, ja,
tr, hi, ms, nl, sv, da, fi, pl, cs, fil, fa, el, hu, mk, ro, ar, yue` (plus
Chinese dialects). See the model card for the full list.

## Hotwords

`HOTWORDS` biases the transcription toward given words (proper nouns, jargon).
Give comma-separated entries (each may be a multi-word phrase, e.g.
`Quilter, Au revoir`) or, with no comma, whitespace-separated words. They are
formatted into Qwen3-ASR's documented context prompt — `Vocabulary: a, b, c.` —
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
  request asks for word timestamps — i.e. HTTP `Accept: application/json` and the
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
  wired here; the `vllm` backend is offline-only).
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
