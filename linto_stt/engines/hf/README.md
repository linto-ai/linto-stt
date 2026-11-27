# LinTO-STT-HF

ASR API based on Hugging Face ASR models ([Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf), [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026), ...), served with [vLLM](https://docs.vllm.ai) or in-process `transformers`. Supports offline transcription (HTTP and Celery task), with word timestamps.

> See the [main README](https://github.com/linto-ai/linto-stt/blob/master/README.md) for API docs, Docker options, and serving modes.
> See [ENV.md](https://github.com/linto-ai/linto-stt/blob/master/ENV.md) for all environment variables.

The engine is named after the model format, not after a single model or runtime:
it serves checkpoints in Hugging Face `transformers` format, and detects each
checkpoint's family and capabilities from its processor.

## Quick Start

### Prerequisites

- [Docker](https://www.Docker.com/products/Docker-desktop/)
- A CUDA GPU with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html): the default `vllm` backend needs one (see [GPU memory](#gpu-memory-of-the-vllm-backend)). On CPU, use `BACKEND=transformers`
- About 8GB disk space for the GPU image, plus the downloaded models

### Pull or Build

```sh
docker pull lintoai/linto-stt-hf
```

or

```sh
# GPU: vllm backend (default) and transformers backend
docker build -t linto-stt-hf:latest --build-arg STT_ENGINE=hf --build-arg GPU=1 .

# CPU: transformers backend only
docker build -t linto-stt-hf-cpu:latest --build-arg STT_ENGINE=hf .
```

A **GPU build** (`--build-arg GPU=1`) installs the `hf-vllm` extra (a superset), so
it runs the default `vllm` backend out of the box, and the in-process
`transformers` backend works in the same image. A **CPU build** (no `GPU`) installs
only the lean `hf` extra (no vLLM), so that image **must** be run with
`BACKEND=transformers` (vLLM needs a GPU).

### Run

HTTP file transcription:

```sh
docker run -p 8080:80 --rm \
  -e SERVICE_MODE=http \
  -e MODEL=Qwen/Qwen3-ASR-1.7B-hf \
  -e DEVICE=cuda \
  --gpus all --shm-size=2g \
  --env-file .env \
  lintoai/linto-stt-hf
```

Celery task (needs a broker, see the [main README](https://github.com/linto-ai/linto-stt/blob/master/README.md)):

```sh
docker run --rm \
  -e SERVICE_MODE=task \
  -e SERVICES_BROKER=redis://172.17.0.1:6379 \
  -e MODEL=Qwen/Qwen3-ASR-1.7B-hf \
  -e DEVICE=cuda \
  --gpus all --shm-size=2g \
  --env-file .env \
  lintoai/linto-stt-hf
```

Test with:

```sh
curl -X POST "http://localhost:8080/transcribe?language=fr" -H "accept: application/json" -F "file=@tests/bonjour.wav;type=audio/wav"
```

With the `vllm` backend, give the container enough shared memory (vLLM's
workers use `/dev/shm`): `--shm-size=2g` (or `--ipc=host`), or it may fail with
a bus error. Mount a cache folder to avoid re-downloading the models (see
"Cache mount" in the [main README](https://github.com/linto-ai/linto-stt/blob/master/README.md)).

### Run without internet access

The engine downloads its models from Hugging Face (and one from Meta) on first
use. On a machine without internet access, provide them beforehand, in one of
two ways. Both were tested in a container started with `--network none`.

With the default settings, the models needed are:

| Model | Setting | Needed |
|---|---|---|
| The ASR model, e.g. `Qwen/Qwen3-ASR-1.7B-hf` | `MODEL` | always |
| Its vLLM checkpoint, e.g. `Qwen/Qwen3-ASR-1.7B` | (`MODEL` without `-hf`) | `vllm` backend, Qwen3-ASR only (vLLM serves the non-`-hf` checkpoint) |
| `Qwen/Qwen3-ForcedAligner-0.6B-hf` | `ALIGNER_MODEL` | word timestamps (unless `ALIGNER_MODEL=none` or `mms`) |
| MMS aligner, `model.pt` (1.2 GB) from `https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt` | `ALIGNER_FALLBACK_MODEL` (or `ALIGNER_MODEL=mms`) | unless `ALIGNER_FALLBACK_MODEL=none`; not on Hugging Face: torchaudio keeps it in the torch hub cache, `$TORCH_HOME/hub/checkpoints/model.pt` |
| `speechbrain/lang-id-voxlingua107-ecapa` | `LANGUAGE_IDENTIFICATION_MODEL` | unless `LANGUAGE_IDENTIFICATION=false` |

On GPU these models are loaded at startup, so a missing one makes the server
fail at startup rather than at the first request; the last "Loading ..." line of
the log names the model that could not be found.

**1. Mount pre-filled caches.** Fill a Hugging Face cache and a torch cache on a
machine with internet access, for instance by starting the container once there
with the same settings and the same mounts (on GPU, the startup loads every model
listed above; on CPU, models load at the first request, so send one), or with
`hf download <repository>` and a copy of `model.pt` into
`<torch cache>/hub/checkpoints/`. Then mount both caches and set
`HF_HUB_OFFLINE=1`, so that nothing tries to reach the Hub:

```sh
docker run -p 8080:80 --rm \
  -e SERVICE_MODE=http \
  -e MODEL=Qwen/Qwen3-ASR-1.7B-hf \
  -e DEVICE=cuda --gpus all --shm-size=2g \
  -v /path/to/huggingface_cache:/opt/hf_cache -e HF_HOME=/opt/hf_cache \
  -v /path/to/torch_cache:/opt/torch_cache -e TORCH_HOME=/opt/torch_cache \
  -e HF_HUB_OFFLINE=1 \
  -e USER_ID=$(id -u) -e GROUP_ID=$(id -g) \
  lintoai/linto-stt-hf
```

**2. Mount local model folders** (copies of the Hugging Face repositories, e.g.
made with `git clone https://huggingface.co/<repository>` or
`hf download <repository> --local-dir <folder>`), and give their paths instead
of the repository IDs. For Qwen3-ASR on the `vllm` backend, the folder of the
vLLM checkpoint must sit next to the `MODEL` folder, with the same name without
`-hf`:

```sh
docker run -p 8080:80 --rm \
  -e SERVICE_MODE=http \
  -e DEVICE=cuda --gpus all --shm-size=2g \
  -v /path/to/models:/models:ro \
  -e MODEL=/models/Qwen/Qwen3-ASR-1.7B-hf \
  -e ALIGNER_MODEL=/models/Qwen/Qwen3-ForcedAligner-0.6B-hf \
  -e LANGUAGE_IDENTIFICATION_MODEL=/models/speechbrain/lang-id-voxlingua107-ecapa \
  -v /path/to/torch_cache:/opt/torch_cache -e TORCH_HOME=/opt/torch_cache \
  -e HF_HUB_OFFLINE=1 \
  -e USER_ID=$(id -u) -e GROUP_ID=$(id -g) \
  lintoai/linto-stt-hf
```

Here `/path/to/models` holds `Qwen/Qwen3-ASR-1.7B-hf`, `Qwen/Qwen3-ASR-1.7B` (for
the `vllm` backend), `Qwen/Qwen3-ForcedAligner-0.6B-hf` and
`speechbrain/lang-id-voxlingua107-ecapa`, and the torch cache holds
`hub/checkpoints/model.pt` (MMS). Mounting the caches outside the container
user's home (`/opt/...` here) avoids the entrypoint's ownership change on them;
`USER_ID` / `GROUP_ID` let the container user read them.

### Run without Docker

Default backend (`vllm`, GPU):

```sh
uv sync --extra hf-vllm
MODEL=Qwen/Qwen3-ASR-1.7B-hf DEVICE=cuda uv run main.py -m http -e hf -p 8080 -i 127.0.0.1
```

Transformers backend (in-process; runs on CPU too):

```sh
uv sync --extra hf
MODEL=Qwen/Qwen3-ASR-1.7B-hf DEVICE=cuda BACKEND=transformers uv run main.py -m http -e hf -p 8080 -i 127.0.0.1
```

## Supported models

`MODEL` is any transformers-native ASR checkpoint; the family is detected from its
`model_type`.

| Model (`model_type`) | `MODEL` id(s) | Auto language detect | Force language | Hotwords |
|---|---|:---:|:---:|:---:|
| **Qwen3-ASR** (`qwen3_asr`) | `Qwen/Qwen3-ASR-1.7B-hf` (default), `Qwen/Qwen3-ASR-0.6B-hf` | ✅ | ✅ | ✅ |
| **Cohere Transcribe** (`cohere_asr`) | `CohereLabs/cohere-transcribe-03-2026` | ❌ | ✅ | ❌ |
| **Voxtral** (`voxtral`) \* | `mistralai/Voxtral-Mini-3B-2507`, `mistralai/Voxtral-Small-24B-2507` | ✅ | ✅ | ❌ |
| **Granite-Speech** (`granite_speech`) \* | `ibm-granite/granite-speech-5.0-2b`, `ibm-granite/granite-speech-4.1-2b`, `ibm-granite/granite-speech-3.3-8b` | ✅ | ✅ 5.0 only | ✅ 5.0, 4.1 |
| **Whisper** (`whisper`) | `openai/whisper-large-v3`, `openai/whisper-large-v3-turbo` | ✅ | ✅ | ❌ |

- **Backends.** Qwen3-ASR, Cohere and Whisper run on **both** the `transformers`
  and `vllm` backends (on `transformers`, Whisper decodes audio over 30 s with
  transformers' sequential long-form decoding). The **\*-marked** models
  (Voxtral, Granite-Speech) run on the **`vllm` backend only** for now (their
  in-process `transformers` path is not wired). vLLM serves Voxtral via the
  same transcription endpoint as the others. Granite-Speech goes through vLLM's
  chat endpoint instead (see [Granite-Speech](#granite-speech)).
- **Word-level timestamps** (via the forced aligner) work for *every* model.
- **Capabilities are detected from the model's processor signature, not its
  name**, so a request asking for an unsupported feature (hotwords on a model
  without them, or auto-detection on a model that requires a language, when
  language identification is off) is rejected with a clear error. For the \*-marked (vLLM-only) models the flags are
  detected and may be conservative (e.g. a model that can force a language but
  does not expose it in its processor signature is marked ❌).
- For Qwen the `-hf` (transformers) variant is used, not the plain
  `Qwen/Qwen3-ASR-1.7B` repo, which targets the separate `qwen-asr` package.
- **VRAM:** the bigger checkpoints need a big GPU (Voxtral-Small is 24B,
  Granite-3.3 is 8B); the Mini/2B ones fit a single mid-range GPU.

All require **transformers >= 5.13** (the `hf` / `hf-vllm` extra); seq2seq
models (Cohere, Whisper) also use `librosa` (included). The `vllm` backend can
serve any vLLM-servable ASR model, so **more leaderboard models can be added by
listing + validating them** (no code), e.g. MOSS-Transcribe-Diarize
(`OpenMOSS-Team/MOSS-Transcribe-Diarize`). Candidates come from the
[Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
and the [LINAGORA ASR benchmark](https://linagora-labs.github.io/asr_benchmark/).

### Whisper: prefer the whisper engine

Whisper runs on this engine, but for production use the dedicated
[whisper engine](https://github.com/linto-ai/linto-stt/blob/master/linto_stt/engines/whisper/README.md) (image
[`lintoai/linto-stt-whisper`](https://hub.docker.com/r/lintoai/linto-stt-whisper)),
which is built around faster-whisper and offers what this engine lacks for Whisper:

- **Voice activity detection** before decoding (`VAD`), which helps against
  hallucinations on silence. Here the audio goes to the model as is.
- **Beam search with temperature fallback** (`USE_ACCURATE`). Here decoding is
  greedy only.
- **Hotwords and initial prompt.** Here a request with hotwords is rejected for Whisper.
- **Native word timestamps.** Here they come from a separate forced aligner,
  which costs time and VRAM.
- **Streaming** (WebSocket mode). Not supported by this engine: the server
  refuses to start in that mode.

Speed and VRAM, measured on an RTX 4090 Laptop GPU with word timestamps, on
meeting recordings cut in segments of 30 to 80 s (RTFx = audio duration /
processing time, higher is faster; see `benchmark/`):

| Whisper model | whisper engine | this engine, `vllm` backend |
|---|---|---|
| large-v3-turbo | RTFx 31 (greedy) or 20 (beam 5), 1.5 GB | RTFx 31, 8.7 GB |
| large-v3 | RTFx 6.5 (greedy) or 5.4 (beam 5), 3.3 GB | RTFx 20, 10.3 GB |

So for turbo this engine brings no speed and costs about 6 times the VRAM. For
large-v3 it is about 3 times faster, for 3 times the VRAM and without the
features above; accuracy on spontaneous French was about 0.5 WER point worse.

## HF-Specific Configuration

### Backends

The ASR step runs on a backend selected by `BACKEND`:

- `vllm` (default): a managed [vLLM](https://docs.vllm.ai) `vllm serve`
  **subprocess** that the engine starts and talks to over HTTP
  (`/v1/audio/transcriptions`), for higher throughput. One server per worker,
  started inside the container; no external service to run. Supports every
  vLLM-servable ASR model. **CUDA GPU** only; install with `uv sync --extra hf-vllm`.
- `transformers`: in-process Hugging Face `transformers`. Runs on CPU too (the
  portable fallback), but does not cover `vllm`-only models such as Voxtral.
  Install with `uv sync --extra hf`.

> **On CPU you must set `BACKEND=transformers`.** The default backend is `vllm`,
> which requires a CUDA GPU, so a CPU run (or a CPU Docker image, built without
> `GPU=1`, which does not install vLLM) has to select the `transformers` backend
> explicitly.

The `vllm` backend installs with `uv sync --extra hf-vllm` (vLLM is pinned to a
version that still allows the aligner's transformers). vLLM uses the non-`-hf`
checkpoint, so an `-hf` `MODEL` is mapped to its non-`-hf` id automatically. The
serve backend also reads `VLLM_SERVE_HOST` / `VLLM_SERVE_PORT` and
`VLLM_SERVE_EXTRA_ARGS` (see [ENV.md](https://github.com/linto-ai/linto-stt/blob/master/ENV.md)).

### GPU memory of the vllm backend

vLLM reserves its memory at startup: the model weights plus a KV cache. The
engine pins that cache by default, so that a worker takes what transcription
needs and can share a GPU:

| Variable | Default | Role |
|---|---|---|
| `VLLM_KV_CACHE_MEMORY_BYTES` | `536870912` (512 MiB) | size of the KV cache |
| `VLLM_MAX_MODEL_LEN` | `4096` | context cap per request, kept within the model's own limit (Cohere 1024, Whisper 448) |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.25` with a pinned cache | startup check only: vLLM refuses to start unless this fraction of the card is free |

These defaults fit Qwen3-ASR-1.7B and Cohere Transcribe (and Whisper) on files
of 30 s to 20 minutes. The reasoning:

- The cache must hold at least one request of `VLLM_MAX_MODEL_LEN` tokens,
  otherwise vLLM refuses to start. vLLM decodes audio in windows of about 30 s,
  so a request stays far below 4096 tokens, and 512 MiB holds a 4096-token
  request of Qwen3-ASR-1.7B (which needs 1.75 GiB for 16384 tokens).
- Larger caches (1 GiB, 2 GiB) added their size to the memory use and no
  measurable speed, for one request at a time.
- To size the cache as a fraction of the card instead, set
  `VLLM_KV_CACHE_MEMORY_BYTES=0`: vLLM then reserves
  `VLLM_GPU_MEMORY_UTILIZATION` of the card (0.6 if unset) for weights and cache.

GPU memory of the whole worker with these defaults (vLLM server plus the forced
aligners, word timestamps on, files up to 20 minutes), measured on an RTX 4090
Laptop GPU (16 GB), see `benchmark/`:

| `MODEL` | After loading | Peak |
|---|:--:|:--:|
| `Qwen/Qwen3-ASR-1.7B-hf` | 8.5 GB | 14.6 GB |
| `CohereLabs/cohere-transcribe-03-2026` | 7.4 GB | 12.7 GB |
| `openai/whisper-large-v3` | 6.3 GB | 10.3 GB |
| `openai/whisper-large-v3-turbo` | 4.7 GB | 8.7 GB |

The peak depends on the order of the files: it is reached when a file of about
5 minutes (the heaviest input of the main aligner) follows a 20-minute file
(which brings the vLLM server to its largest size), with both aligners loaded.
The figures above were measured on that sequence.

Voxtral-Mini-3B is not cut into windows by vLLM: the engine cuts long files
itself for it (see "Long audio" below), so the defaults above suit it too. It
took 12.4 GB after loading in the same test.

Everything else (language selection, hotwords, long-audio chunking, and word
timestamps, always via the transformers forced aligner) is shared between the two
backends.

### Language

- `LANGUAGE=fr` / `LANGUAGE=French`: force the language (ISO code or full
  English name). It must be one the model supports (see below), otherwise the
  request is rejected (HTTP 400, or a failed Celery task), on every backend.
  It is returned as `language` in the result.
- `LANGUAGE=*` (or unset): the language is unknown.
    - With **language identification** (`LANGUAGE_IDENTIFICATION=true`, the
      default), the language is first identified by SpeechBrain's ECAPA model
      (trained on VoxLingua107, 107 languages, Apache 2.0) on the first 30 s of
      the audio, choosing **only among the languages of the model**: by default
      the languages documented on its model card when it lists them, else all
      those it supports (`LANGUAGE_IDENTIFICATION_LANGUAGES=supported`: always
      all those it supports). It is then forced for the transcription, used to
      choose the aligner, and returned as `language`. Models that require a language (Cohere) then work
      without one. When the most likely language of the 107 is not
      one of them (audio in another language), a warning is logged and the most
      likely of them is used. Cost: about 20 ms per request on GPU,
      81 MB of weights (0.5 GB of GPU memory at peak), loaded at startup; the
      weights (86 MB) are downloaded to the Hugging Face cache on first use.
    - Without (`LANGUAGE_IDENTIFICATION=false`), the model detects the language
      itself when it can (Qwen3-ASR, Whisper, Voxtral) and may transcribe
      multilingually; a model that requires a language (Cohere) rejects the
      request. The returned `language` is the one the model reports on the
      `transformers` backend, and `*` on the `vllm` backend (whose
      transcription endpoint does not return it).
- Overridable per request with the `language` query parameter (HTTP) or task
  argument.

Each model has two language lists, read from its files (never from its name):

- **Supported**: a request for another language is rejected (HTTP 400), and
  language identification chooses only among these. Taken from what the model
  itself enforces: Qwen3-ASR's processor, Cohere's config, Whisper's tokenizer.
  The other models accept any of Whisper's 100 codes, which is what vLLM's
  transcription endpoint accepts (it rejects any other code).
- **Documented**: the languages of the model card metadata (`language:`). A
  supported language outside them is still transcribed, with a warning in the
  logs (for Voxtral, a forced language outside its 8 is respected: with `ru`,
  French speech comes out in Cyrillic letters).

| Model | Supported (rejected outside) | Documented (warning outside) | Without a language, identification off |
|---|---|---|---|
| **Qwen3-ASR** | 30: `zh, en, yue, ar, de, fr, es, pt, id, it, ko, ru, th, vi, ja, tr, hi, ms, nl, sv, da, fi, pl, cs, fil, fa, el, hu, mk, ro` (+ 22 Chinese dialects, per its card) | same 30 | detects it (multilingual audio possible) |
| **Cohere Transcribe** | 14: `en, fr, de, it, es, pt, el, nl, pl, zh, ja, ko, vi, ar` | same 14 | request rejected (a language is required) |
| **Whisper** (large-v3, large-v3-turbo) | 100 (Whisper's tokenizer) | 99 (no `yue`) | detects it |
| **Voxtral** (Mini-3B, Small-24B) | 100 (vLLM) | 8: `en, es, fr, pt, hi, de, nl, it` | detects it |
| **Granite-Speech** | 100 (vLLM). A given language is passed to 5.0 only, and rejected by 4.1 and 3.3; an identified one is returned but never passed to the model (a wrong language makes it translate) | 5.0-2b, 4.1-2b: `en, fr, de, es, pt, ja`; 3.3-8b: `en, fr, de, es, pt` | detects it |

With identification on (the default), a request without a language gets one
of the **documented** languages, or of the **supported** ones when the card
lists none (`LANGUAGE_IDENTIFICATION_LANGUAGES=supported`: always among the
supported ones, with a warning when it is outside the documented ones). On
models that accept vLLM's 100 languages, this avoids rare languages: on a 1 s
French "Bonjour", the most likely of the 100 is Breton. Language identification (ECAPA) knows 107 languages: a
supported language it does not know cannot be identified (`yue`: ECAPA only
has `zh`); `fil` is identified through ECAPA's `tl` (Tagalog).

The order of the lists follows each model card and is **not** a published
best-to-worst ranking (Qwen lists Chinese/English first; Cohere groups by
region: European, then APAC, then MENA). Neither card publishes a per-language
quality order, so consult each model card's per-language WER figures for
relative accuracy.

The choice of ECAPA and its accuracy (98.7% correct on VoxLingua107's
development set restricted to Qwen3-ASR's 30 languages, against 94.3% among
all 107) come from a benchmark in this repository (`benchmark/lid/README.md`).

Note: the forced aligner used for **word timestamps**
([`Qwen3-ForcedAligner-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf))
covers only 11 languages (`zh, en, yue, fr, de, it, ja, ko, pt, ru, es`).
For the other languages the aligner is still run, without a language hint (the
hint only drives how the transcript is split into words; the model itself is
language-agnostic), with a warning in the logs: timestamps are produced for every
language, but their accuracy is only validated by Qwen for the 11 above. For a
language-agnostic aligner see `ALIGNER_MODEL=mms` / `ALIGNER_FALLBACK_MODEL=mms`
in the [Word timestamps](#word-timestamps) section.

### Hotwords

`HOTWORDS` biases the transcription toward given words (proper nouns, jargon).
Give comma-separated terms, each possibly made of several words (e.g.
`Quilter, Au revoir`); without a comma, the whole string is one term (the
format common to all engines). They are formatted into the prompt the model's
card documents: `Vocabulary: a, b, c.` for Qwen3-ASR (its context prompt),
`Keywords: a, b, c` for Granite-Speech (see below). Overridable per request
with the `hotwords` query parameter / task argument.

Note: Qwen3-ASR's context prompt is soft LM conditioning (not literal token
biasing like faster-whisper's hotwords), and the model normalises casing, so
hotwords nudge *which* words are produced rather than forcing an exact spelling.

### Granite-Speech

vLLM's transcription endpoint gives Granite-Speech a fixed prompt (a lowercase
transcript without punctuation) and ignores the language and the hotwords. The
engine therefore sends these models to vLLM's chat endpoint, with the prompts
of the "Preferred prompt by task" table of their model card, read from the card
(no setting):

| Prompt | 5.0-2b | 4.1-2b | 3.3-8b |
|---|:---:|:---:|:---:|
| `transcribe the speech with proper punctuation and capitalization.` (else `can you transcribe the speech into a written format?`) | ✅ | ✅ | ❌ |
| hotwords: `Keywords: a, b, c` (up to 2000 keywords for 5.0, per its card) | ✅ | ✅ | ❌ |
| a given language: `The speech is in French. Transcribe it exactly; do not translate.` | ✅ | ❌ | ❌ |

- A given language (`LANGUAGE` or the request) is passed to 5.0. A wrong one
  makes the model translate (French speech, given English, comes out in
  English), so a language found by language identification is returned but not
  passed to the model, which then recognizes the language itself.
- The chat endpoint does not cut long audio: the engine cuts files longer than
  2 minutes into 2-minute chunks (the card advises segments of up to about 2
  minutes), unless `LONG_FILE_THRESHOLD` / `LONG_FILE_CHUNK_LEN` are set.
- The 5.0-2b repository also holds `out_llm.safetensors`, an extra layer that
  is not part of the model, which vLLM would load and reject. The engine serves
  the model from a temporary folder of links to its files, without that one.

### Word timestamps

Qwen3-ASR has no native word timestamps, so they are produced by a companion
forced-alignment model, [`Qwen3-ForcedAligner-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)
(`ALIGNER_MODEL`), which aligns the transcript against the audio.

- The aligner is a **second model**. It runs **only** when a request asks for
  word timestamps, i.e. HTTP `Accept: application/json` and the Celery task (both
  go through the metadata path). Plain-text requests (`Accept: text/plain`) never
  run it, so that fast path keeps the best RTF. On GPU the startup warmup
  preloads it, and the fallback aligner below (so no request waits for an
  aligner to load); otherwise they are loaded on first use.
- Set `ALIGNER_MODEL=` (empty) or `ALIGNER_MODEL=none` to disable alignment; the
  `words` field is then returned empty for metadata requests.
- **MMS aligner.** `ALIGNER_MODEL=mms` selects Meta's MMS forced aligner instead
  (torchaudio's `MMS_FA` bundle: a 300M wav2vec2 CTC model trained on 1,100+
  languages). It is language-agnostic: each word is romanised with `uroman`
  (`Привет` -> `privet`, `ça` -> `sa`) and aligned character by character, so
  it covers every language the ASR models can output. Tokens that romanise to
  nothing (digits, times such as `06:30`, symbols) cannot be aligned and get a
  zero-width stamp at the previous word's end. Word starts are reliable; the
  end of a word followed by a long pause may stretch into the silence (a CTC
  aligner trait). Its 1.2 GB float32 weights are
  fetched by torchaudio into the torch hub cache (`TORCH_HOME`), not the
  Hugging Face cache.
- **Fallback (default).** `ALIGNER_FALLBACK_MODEL` (default `mms`) keeps the
  Qwen aligner for the requests it covers (its 11 tuned languages, audio up to
  its documented 5-minute limit) and routes the others to MMS: another language
  (forced or detected), or a file longer than 5 minutes (on the `vllm` backend
  the whole file is aligned in one pass; MMS has no duration limit). The
  fallback is loaded at startup with the primary aligner (on GPU), so the
  worker's GPU memory does not grow later; the first start downloads its
  1.2 GB unless the torch hub cache is already populated (the Docker
  `~/.cache` mount covers it). Set `ALIGNER_FALLBACK_MODEL=none` to always use
  the primary aligner (long audio is then still aligned with it, with a warning
  in the logs).

## Notes and Limitations

- **GPU strongly recommended.** The model is an LLM-based ASR; CPU inference is
  slow. The default `vllm` backend requires a CUDA GPU, so **on CPU you must set
  `BACKEND=transformers`** (and a CPU Docker image, built without `GPU=1`, has
  only that backend).
- **No WebSocket/streaming.** Only the `http` and `task` modes are supported:
  with `SERVICE_MODE=websocket` (or `-m websocket`) the server refuses to start.
  Streaming would need vLLM's realtime API, which is not wired here (the vLLM
  backends are used for offline transcription only).
- **Audio input: WAV.** HTTP requests must send WAV. Celery tasks read files
  with libsndfile, so WAV works, as do FLAC, OGG/Vorbis and MP3; the Docker image
  ships no ffmpeg, so other containers (M4A/AAC, WebM, MP4...) are rejected
  with an explicit error. Convert such files to WAV upstream.
- **Word timestamps** come from the forced-alignment model (see above), which
  runs only on metadata requests. Disable it with `ALIGNER_MODEL=none`.
- **Long audio**: handling depends on the backend.
  - On the **`vllm`** backend, vLLM chunks long audio **server-side** (~30 s
    windows, split at quiet points with a short overlap), so `LONG_FILE_THRESHOLD`
    / `LONG_FILE_CHUNK_LEN` have **no effect**. Exceptions: Voxtral, which vLLM
    feeds whole (its context grows with the audio duration), and Granite-Speech
    (sent to the chat endpoint, see [Granite-Speech](#granite-speech)). For them
    the engine cuts files longer than `LONG_FILE_THRESHOLD` into
    `LONG_FILE_CHUNK_LEN` pieces itself, as on the `transformers` backend (for
    Granite-Speech, 120 s by default instead of 240 s), so they work on any
    length with the default memory settings.
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
  [ENV.md](https://github.com/linto-ai/linto-stt/blob/master/ENV.md).

## License

AGPLv3 (see LICENSE).

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Qwen3-ASR and Qwen3-ForcedAligner](https://github.com/QwenLM/Qwen3-ASR)
- [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
- [MMS forced aligner](https://docs.pytorch.org/audio/stable/pipelines.html) (torchaudio) and [uroman](https://github.com/isi-nlp/uroman)
