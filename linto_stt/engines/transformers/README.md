# Transformers engine

Speech-to-text with Hugging Face `transformers`-native ASR models, through a
managed `vllm` server or an in-process `transformers` backend. The engine is named after the
library (not a single model): it auto-detects each checkpoint's family and
capabilities from its processor, so it supports several ASR models.

## Supported models

`MODEL` is any transformers-native ASR checkpoint; the family is detected from its
`model_type`.

| Model (`model_type`) | `MODEL` id(s) | Auto language detect | Force language | Hotwords |
|---|---|:---:|:---:|:---:|
| **Qwen3-ASR** (`qwen3_asr`) | `Qwen/Qwen3-ASR-1.7B-hf` (default), `Qwen/Qwen3-ASR-0.6B-hf` | ✅ | ✅ | ✅ |
| **Cohere Transcribe** (`cohere_asr`) | `CohereLabs/cohere-transcribe-03-2026` | ❌ | ✅ | ❌ |
| **Whisper** (`whisper`) \* | `openai/whisper-large-v3`, `openai/whisper-large-v3-turbo` | ✅ | ✅ | ❌ |
| **Voxtral** (`voxtral`) \* | `mistralai/Voxtral-Mini-3B-2507`, `mistralai/Voxtral-Small-24B-2507` | ✅ | ✅ | ❌ |
| **Granite-Speech** (`granite_speech`) \* | `ibm-granite/granite-speech-4.1-2b`, `ibm-granite/granite-speech-3.3-8b` | ✅ | ❌ | ❌ |

- **Backends.** Qwen3-ASR and Cohere run on **both** the `transformers` and `vllm`
  backends. The **\*-marked** models (Whisper, Voxtral, Granite-Speech) run on the
  **`vllm` backend only** for now (their in-process `transformers` path is not
  wired). vLLM serves them via the same transcription endpoint, so no code change
  is needed to use them, only to list/validate them.
- **Word-level timestamps** (via the forced aligner) work for *every* model.
- **Capabilities are detected from the model's processor signature, not its
  name**, so a request asking for an unsupported feature (hotwords on a model
  without them, or auto-detection on a model that requires a language) is
  rejected with a clear error. For the \*-marked (vLLM-only) models the flags are
  detected and may be conservative (e.g. a model that can force a language but
  does not expose it in its processor signature is marked ❌).
- For Qwen the `-hf` (transformers) variant is used, not the plain
  `Qwen/Qwen3-ASR-1.7B` repo, which targets the separate `qwen-asr` package.
- **VRAM:** the bigger checkpoints need a big GPU (Voxtral-Small is 24B,
  Granite-3.3 is 8B); the Mini/2B ones fit a single mid-range GPU.

All require **transformers >= 5.13** (the `transformers` / `vllm` extra); seq2seq
models (Cohere, Whisper) also use `librosa` (included). The `vllm` backend can
serve any vLLM-servable ASR model, so **more leaderboard models can be added by
listing + validating them** (no code), e.g. MOSS-Transcribe-Diarize
(`OpenMOSS-Team/MOSS-Transcribe-Diarize`). Candidates come from the
[Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
and the [LINAGORA ASR benchmark](https://linagora-labs.github.io/asr_benchmark/).

## Backends

The ASR step runs on a backend selected by `BACKEND`:

- `vllm` (default): a managed [vLLM](https://docs.vllm.ai) `vllm serve`
  **subprocess** that the engine starts and talks to over HTTP
  (`/v1/audio/transcriptions`), for higher throughput. One server per worker,
  started inside the container; no external service to run. Supports every
  vLLM-servable ASR model. **CUDA GPU** only; install with `uv sync --extra vllm`.
- `transformers`: in-process Hugging Face `transformers`. Runs on CPU too (the
  portable fallback), but does not cover `vllm`-only models such as Whisper.
  Install with `uv sync --extra transformers`.

> **On CPU you must set `BACKEND=transformers`.** The default backend is `vllm`,
> which requires a CUDA GPU, so a CPU run (or a CPU Docker image, built without
> `GPU=1`, which does not install vLLM) has to select the `transformers` backend
> explicitly.

The `vllm` backend installs with `uv sync --extra vllm` (vLLM is pinned to a
version that still allows the aligner's transformers). vLLM uses the non-`-hf`
checkpoint, so an `-hf` `MODEL` is mapped to its non-`-hf` id automatically. Tune
with `VLLM_GPU_MEMORY_UTILIZATION` (fraction of VRAM) or
`VLLM_KV_CACHE_MEMORY_BYTES` (absolute KV-cache budget, better on unified-memory /
shared GPUs), and `VLLM_MAX_MODEL_LEN` (context cap, kept within each model's own
limit). The serve backend also reads `VLLM_SERVE_HOST` / `VLLM_SERVE_PORT` and
`VLLM_SERVE_EXTRA_ARGS` (see `ENV.md`).

Everything else (language selection, hotwords, long-audio chunking, and word
timestamps, always via the transformers forced aligner) is shared between the two
backends.

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
| **Whisper** (large-v3) | 99 | multilingual; too many to list, see the [model card](https://huggingface.co/openai/whisper-large-v3) |

The order follows each model card and is **not** a published best-to-worst
ranking (Qwen lists Chinese/English first; Cohere groups by region: European,
then APAC, then MENA). Neither card publishes a per-language quality order, so
consult each model card's per-language WER figures for relative accuracy.

Note: the forced aligner used for **word timestamps**
([`Qwen3-ForcedAligner-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf))
covers only 11 languages (`zh, en, yue, fr, de, it, ja, ko, pt, ru, es`).
For the other languages the aligner is still run, without a language hint (the
hint only drives how the transcript is split into words; the model itself is
language-agnostic), with a warning in the logs: timestamps are produced for every
language, but their accuracy is only validated by Qwen for the 11 above. For a
language-agnostic aligner see `ALIGNER_MODEL=mms` / `ALIGNER_FALLBACK_MODEL=mms`
in the [Word timestamps](#word-timestamps) section.

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

- The aligner is a **second model**. It runs **only** when a request asks for
  word timestamps, i.e. HTTP `Accept: application/json` and the Celery task (both
  go through the metadata path). Plain-text requests (`Accept: text/plain`) never
  run it, so that fast path keeps the best RTF. On GPU the startup warmup
  preloads it (so the first timestamped request is not slow); otherwise it is
  loaded lazily on first use.
- Set `ALIGNER_MODEL=` (empty) or `ALIGNER_MODEL=none` to disable alignment; the
  `words` field is then returned empty for metadata requests.
- **MMS aligner.** `ALIGNER_MODEL=mms` selects Meta's MMS forced aligner instead
  (torchaudio's `MMS_FA` bundle: a 300M wav2vec2 CTC model trained on 1,100+
  languages). It is language-agnostic: each word is romanised with `uroman`
  (`Привет` -> `privet`, `ça` -> `sa`) and aligned character by character, so
  it covers every language the ASR models can output. Its 1.2 GB float32
  weights are fetched by torchaudio into the torch hub cache (`TORCH_HOME`),
  not the Hugging Face cache.
- **Fallback (default).** `ALIGNER_FALLBACK_MODEL` (default `mms`) keeps the
  Qwen aligner for the requests it covers (its 11 tuned languages, audio up to
  its documented 5-minute limit) and routes the others to MMS: another language
  (forced or detected), or a file longer than 5 minutes (on the `vllm` backend
  the whole file is aligned in one pass; MMS has no duration limit). The
  fallback is loaded only on the first request that needs it, so deployments
  that never hit such a request pay nothing; that first request waits for the
  1.2 GB download unless the torch hub cache is already populated (the Docker
  `~/.cache` mount covers it). Set `ALIGNER_FALLBACK_MODEL=none` to always use
  the primary aligner (long audio is then still aligned with it, with a warning
  in the logs).

## Running

Default backend (`vllm`, GPU):

```sh
uv sync --extra vllm
MODEL=Qwen/Qwen3-ASR-1.7B-hf DEVICE=cuda uv run main.py -m http -e transformers -p 8080 -i 127.0.0.1
# transcribe:
curl -s -X POST "http://127.0.0.1:8080/transcribe?language=fr" -F "file=@tests/bonjour.wav"
```

Transformers backend (in-process HF; runs on CPU too):

```sh
uv sync --extra transformers
MODEL=Qwen/Qwen3-ASR-1.7B-hf DEVICE=cuda BACKEND=transformers uv run main.py -m http -e transformers -p 8080 -i 127.0.0.1
```

Docker:

```sh
docker build -t linto-stt-transformers --build-arg STT_ENGINE=transformers --build-arg GPU=1 .
```

A **GPU build** (`--build-arg GPU=1`) installs the `vllm` extra (a superset), so
it runs the default `vllm` backend out of the box, and the in-process
`transformers` backend works in the same image. A **CPU build** (no `GPU`) installs
only the lean `transformers` extra (no vLLM), so that image **must** be run with
`BACKEND=transformers` (vLLM needs a GPU).

When running the GPU image with the `vllm` backend, give the container enough
shared memory (vLLM's workers use `/dev/shm`): add `--shm-size=2g` (or
`--ipc=host`) to `docker run`, or it may fail with a bus error.

## Notes / limitations

- **GPU strongly recommended.** The model is an LLM-based ASR; CPU inference is
  slow. The default `vllm` backend requires a CUDA GPU, so **on CPU you must set
  `BACKEND=transformers`** (and a CPU Docker image, built without `GPU=1`, has
  only that backend).
- **No WebSocket/streaming** (it would need vLLM's realtime API, which is not
  wired here; the vLLM backends are used for offline transcription only).
- **Word timestamps** come from the forced-alignment model (see above), which
  runs only on metadata requests. Disable it with `ALIGNER_MODEL=none`.
- **Long audio**: handling depends on the backend.
  - On the **`vllm`** backend, vLLM chunks long audio **server-side** (~30 s
    windows, split at quiet points with a short overlap), so `LONG_FILE_THRESHOLD`
    / `LONG_FILE_CHUNK_LEN` have **no effect**.
  - On the **`transformers`** backend, only models that do not self-chunk
    (Qwen3-ASR) are chunked by the engine: files longer than `LONG_FILE_THRESHOLD`
    seconds are transcribed in consecutive non-overlapping chunks of
    `LONG_FILE_CHUNK_LEN` seconds (word timestamps offset per chunk), avoiding
    out-of-memory / truncation. A word spoken across a chunk boundary may be split
    there; keep chunks a few minutes long to make this rare. Seq2seq models
    (Cohere) self-chunk, so these vars do not affect them either.
- Efficiency: the default attention (`sdpa`) already uses Flash-Attention
  kernels on supported GPUs, so it is efficient out of the box. Other knobs:
  `DTYPE`, `MAX_NEW_TOKENS`, and `ATTN_IMPLEMENTATION=flash_attention_2` (opt-in;
  requires separately installing the GPU-only `flash-attn` package). See
  `ENV.md`.
