# LinTO-STT-Whisper

ASR API based on [Whisper models](https://openai.com/research/whisper). Supports offline and real-time transcription.

> See the [main README](https://github.com/linto-ai/linto-stt/blob/master/README.md) for API docs, Docker options, and serving modes.
> See [ENV.md](https://github.com/linto-ai/linto-stt/blob/master/ENV.md) for all environment variables.

## Quick Start

### Prerequisites

- [Docker](https://www.Docker.com/products/Docker-desktop/)
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU
- At least 8GB disk space for the image, plus up to 5GB per model
- Up to 7GB RAM depending on model size

### VRAM Usage

Approximate GPU VRAM peak usage by model size and backend:

| Model size                     | ct2/faster_whisper int8 | ct2/faster_whisper float16 | ct2/faster_whisper float32 | torch/whisper_timestamped float32 |
| ------------------------------ | ----------------------- | -------------------------- | -------------------------- | --------------------------------- |
| tiny                           | 1.5G                    | 1.5G                       | 1.5G                       | 1.5G                              |
| distil-whisper/distil-large-v2 | 2.2G                    | 3.2G                       | 4.8G                       | 4.4G                              |
| large (large-v3, ...)          | 2.8G                    | 4.8G                       | 8.2G                       | 10.4G                             |
| large-v3-turbo                 | 1.3G                    | 2.0G                       | 4.0G                       | 6.0G                              |

### Pull or Build

```sh
docker pull lintoai/linto-stt-whisper
```

The published image is the GPU build (it includes the CUDA runtime: cuBLAS + cuDNN
for ctranslate2). Without a GPU, it runs on CPU.

or

```sh
# GPU (same as the published image)
docker build -t linto-stt-whisper:latest --build-arg STT_ENGINE=whisper --build-arg GPU=1 .

# CPU only (~1.3GB smaller, without cuBLAS + cuDNN)
docker build -t linto-stt-whisper-cpu:latest --build-arg STT_ENGINE=whisper .
```

### Run

HTTP file transcription:

```sh
docker run -p 8080:80 --rm \
  -e SERVICE_MODE=http \
  -e MODEL=large-v3 \
  --env-file .env \
  lintoai/linto-stt-whisper
```

WebSocket streaming:

```sh
docker run -p 8080:80 --rm \
  -e SERVICE_MODE=websocket \
  -e MODEL=large-v3-turbo \
  --env-file .env \
  lintoai/linto-stt-whisper
```

On GPU, add `--gpus all -e DEVICE=cuda`:

```sh
docker run -p 8080:80 --rm \
  -e SERVICE_MODE=http \
  -e MODEL=large-v3 \
  -e DEVICE=cuda \
  --gpus all \
  --env-file .env \
  lintoai/linto-stt-whisper
```

Mount cache with `-v ~/.cache:/root/.cache` to avoid re-downloads.

## Whisper Models

The model is downloaded on first transcription and cached. You can specify a model by size name, HuggingFace identifier, or local path.

### OpenAI Whisper Models

Multi-lingual:
[tiny](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt) |
[base](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt) |
[small](https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt) |
[medium](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt) |
[large-v1](https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt) |
[large-v2](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt) |
[large-v3](https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt) |
[large-v3-turbo](https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt)

English-only:
[tiny.en](https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt) |
[base.en](https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt) |
[small.en](https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt) |
[medium.en](https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt)

HuggingFace models like [distil-whisper/distil-large-v2](https://huggingface.co/distil-whisper/distil-large-v2) are also supported (use the identifier directly or download locally).

If you used Whisper locally before, models are in `~/.cache/whisper`.

### Alignment Model (deprecated)

The `ALIGNMENT_MODEL` variable for wav2vec word alignment is deprecated and no longer tested. We advise not to use it.

## Whisper-Specific Configuration

### LANGUAGE

The `LANGUAGE` variable sets the default recognition language (can be overridden per request).

Values:

- `*` — automatic language detection
- Language code: `fr`, `en`, `yue`, ...
- BCP-47 code: `fr-FR`, `en-US`, `yue-HK`, ...
- Language name: `French`, `English`, `Cantonese`, ...

Supported languages:
`af`(afrikaans), `am`(amharic), `ar`(arabic), `as`(assamese), `az`(azerbaijani),
`ba`(bashkir), `be`(belarusian), `bg`(bulgarian), `bn`(bengali), `bo`(tibetan), `br`(breton), `bs`(bosnian),
`ca`(catalan), `cs`(czech), `cy`(welsh), `da`(danish), `de`(german), `el`(greek), `en`(english), `es`(spanish),
`et`(estonian), `eu`(basque), `fa`(persian), `fi`(finnish), `fo`(faroese), `fr`(french), `gl`(galician),
`gu`(gujarati), `ha`(hausa), `haw`(hawaiian), `he`(hebrew), `hi`(hindi), `hr`(croatian), `ht`(haitian creole),
`hu`(hungarian), `hy`(armenian), `id`(indonesian), `is`(icelandic), `it`(italian), `ja`(japanese),
`jw`(javanese), `ka`(georgian), `kk`(kazakh), `km`(khmer), `kn`(kannada), `ko`(korean), `la`(latin),
`lb`(luxembourgish), `ln`(lingala), `lo`(lao), `lt`(lithuanian), `lv`(latvian), `mg`(malagasy), `mi`(maori),
`mk`(macedonian), `ml`(malayalam), `mn`(mongolian), `mr`(marathi), `ms`(malay), `mt`(maltese), `my`(myanmar),
`ne`(nepali), `nl`(dutch), `nn`(nynorsk), `no`(norwegian), `oc`(occitan), `pa`(punjabi), `pl`(polish),
`ps`(pashto), `pt`(portuguese), `ro`(romanian), `ru`(russian), `sa`(sanskrit), `sd`(sindhi), `si`(sinhala),
`sk`(slovak), `sl`(slovenian), `sn`(shona), `so`(somali), `sq`(albanian), `sr`(serbian), `su`(sundanese),
`sv`(swedish), `sw`(swahili), `ta`(tamil), `te`(telugu), `tg`(tajik), `th`(thai), `tk`(turkmen), `tl`(tagalog),
`tr`(turkish), `tt`(tatar), `uk`(ukrainian), `ur`(urdu), `uz`(uzbek), `vi`(vietnamese), `yi`(yiddish),
`yo`(yoruba), `zh`(chinese).

Model `large-v3` and derivatives also support `yue`(cantonese).

### Hallucinations

On some 30 s windows, even over speech, Whisper writes a subtitle credit or a
video outro learned from its training data ("Sous-titrage Société
Radio-Canada", "Sous-titrage ST' 501", "Merci d'avoir regardé cette vidéo")
instead of what is said. It does so with a high confidence, so the temperature
fallback does not retry, and the rest of the window can be lost (up to 30 s of
speech). Whether it happens depends on where the window starts, and hotwords
make it more frequent.

With `HALLUCINATION_REDECODE=true` (the default; faster-whisper backend;
offline decoding, not streaming, except plain-text HTTP requests (`Accept:
text/plain`), which have no word timestamps), after the normal decoding:

- **Suspicious time ranges** are the known hallucinated phrases (a multilingual
  list of subtitle credits and video outros, matched whatever the language;
  truncated forms such as "Sous-titrage" only when they make up a whole
  segment), and the time ranges of at least `HALLUCINATION_REDECODE_MIN_GAP`
  seconds (6 by default) of speech, according to the VAD, without any word
  (only with a VAD: without one, pauses and music would count as lost speech).
- Each one is **decoded again** as a separate excerpt running 5 s past the
  time range (for context; only the words inside the time range are kept), in the
  language of the whole audio: starting at the time range (with, then without the
  hotwords), 1 s later (with, then without the hotwords), then 1 s earlier.
- The first attempt with no known hallucinated phrase, not mostly repeated
  words, and a plausible amount of words (0.5 per second of speech in the time
  range, at least one) **replaces** the time range.
- Where a known phrase was and no attempt reaches that amount, any clean attempt
  is better than the phrase: the one with the most words replaces it, or else
  one that gives no word at all in the time range while transcribing the speech
  after it (there is no speech there: the phrase is **removed**).
- Otherwise the time range keeps what Whisper first wrote.

Cost: nothing on audio without suspicious time range; on a one-hour meeting where
Whisper hallucinated about ten times, decoding took up to about 1.8 times
longer on the affected chunks.

### Streaming Tuning

We recommend running streaming on GPU with `large-v3-turbo` or smaller models (avoid `large-v3` as it is very expensive). Using a VAD on the server side (e.g. `silero`) is recommended.

How to choose `STREAMING_MIN_CHUNK_SIZE` and `STREAMING_BUFFER_TRIMMING_SEC`:

- **Low latency** (2-5s on a RTX 4090): Set `STREAMING_MIN_CHUNK_SIZE=0.5` and `STREAMING_BUFFER_TRIMMING_SEC` around 10 (compromise between latency and accuracy). Range: 6 to 15 depending on hardware/model.
- **High latency** (~30s, minimize GPU activity): Set `STREAMING_MIN_CHUNK_SIZE=26`. `STREAMING_BUFFER_TRIMMING_SEC` must be lower (6-12). Lower values reduce GPU usage but may degrade accuracy.

## License

AGPLv3 (see LICENSE).

## Acknowledgments

- [Ctranslate2](https://github.com/OpenNMT/CTranslate2) / [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [OpenAI Whisper](https://github.com/openai/whisper) / [Whisper-Timestamped](https://github.com/linto-ai/whisper-timestamped)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Whisper_Streaming](https://github.com/ufal/whisper_streaming)
