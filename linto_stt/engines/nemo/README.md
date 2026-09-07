# LinTO-STT-NeMo

ASR API built on the [NeMo toolkit](https://github.com/NVIDIA/NeMo). Supports offline and real-time transcription.

> See the [main README](https://github.com/linto-ai/linto-stt/blob/master/README.md) for API docs, Docker options, and serving modes.
> See [ENV.md](https://github.com/linto-ai/linto-stt/blob/master/ENV.md) for all environment variables.

## Quick Start

### Prerequisites

- [Docker](https://www.Docker.com/products/Docker-desktop/)
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU (cuda 12.6+)
- At least 15GB disk space for the Docker image, plus 500MB-5GB per model

### Pull or Build

```sh
docker pull lintoai/linto-stt-nemo
```

or

```sh
docker build -t linto-stt-nemo:latest --build-arg STT_ENGINE=nemo .
```

### Run File Transcription (HTTP)

English:

```sh
docker run -p 8080:80 -it --name linto-stt-nemo \
  -e SERVICE_MODE=http \
  -e MODEL=nvidia/parakeet-tdt-0.6b-v2 \
  -e ARCHITECTURE=rnnt_bpe \
  lintoai/linto-stt-nemo
```

French:

```sh
docker run -p 8080:80 -it --name linto-stt-nemo \
  -e SERVICE_MODE=http \
  -e MODEL=linagora/linto_stt_fr_fastconformer \
  -e ARCHITECTURE=hybrid_bpe \
  lintoai/linto-stt-nemo
```

Add `--gpus all` for GPU. Test with:

```sh
curl -X POST "http://localhost:8080/transcribe" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@tests/bonjour.wav;type=audio/wav"
```

### Run Streaming (WebSocket)

English:

```sh
docker run -p 8080:80 -it --name linto-stt-nemo \
  -e SERVICE_MODE=websocket \
  -e MODEL=nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms_pc \
  -e ARCHITECTURE=hybrid_bpe \
  lintoai/linto-stt-nemo
```

French (no built-in punctuation, see [Punctuation Model](../../../README.md#punctuation-model-recasepunc)):

```sh
docker run -p 8080:80 -it --name linto-stt-nemo \
  -e SERVICE_MODE=websocket \
  -e MODEL=linagora/linto_stt_fr_fastconformer \
  -e ARCHITECTURE=hybrid_bpe \
  lintoai/linto-stt-nemo
```

Multilingual, native cache-aware streaming:

```sh
docker run -p 8080:80 -it --name linto-stt-nemo \
  -e SERVICE_MODE=websocket \
  -e MODEL=nvidia/nemotron-3.5-asr-streaming-0.6b \
  -e ARCHITECTURE=rnnt_bpe \
  lintoai/linto-stt-nemo
```

Add `-e DEVICE=cuda --gpus all` for GPU. Test with:

```sh
python test/test_streaming.py -v --audio_file tests/Hotel20sec.wav
```

### Run Celery Task

```sh
docker run -p 8080:80 -it --name linto-stt-nemo \
  -e SERVICE_MODE=task \
  -e SERVICE_NAME=stt \
  -e SERVICES_BROKER=redis://172.17.0.1:6379 \
  -e MODEL=nvidia/parakeet-tdt-0.6b-v2 \
  -e ARCHITECTURE=rnnt_bpe \
  -e USER_ID=$(id -u) \
  -e GROUP_ID=$(id -g) \
  -v ~/.cache:/home/appuser/.cache \
  -v ~/data/audio:/opt/audio \
  lintoai/linto-stt-nemo
```

## NeMo Models

The model is downloaded from HuggingFace to the cache folder and loaded at startup.

| Model                                                                                                | HuggingFace ID                                 | Lang | Punctuation | Architecture      | WER (Common Voice) | RTFx GPU (RTX 4090) | RTFx CPU (16 threads) | VRAM/RAM (GB) |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ---- | ----------- | ----------------- | ------------------ | ------------------- | --------------------- | ------------- |
| [LinTO French Fast Conformer](https://huggingface.co/linagora/linto_stt_fr_fastconformer)            | `linagora/linto_stt_fr_fastconformer`          | fr   | No          | `hybrid_bpe_rnnt` | 8.96               | 318                 | 48                    | 0.8           |
| [LinTO French Fast Conformer](https://huggingface.co/linagora/linto_stt_fr_fastconformer)            | `linagora/linto_stt_fr_fastconformer`          | fr   | No          | `hybrid_bpe_ctc`  | 10.53              | 734                 | 60                    | 0.8           |
| [NVIDIA French Fast Conformer](https://huggingface.co/nvidia/stt_fr_fastconformer_hybrid_large_pc)   | `nvidia/stt_fr_fastconformer_hybrid_large_pc`  | fr   | Yes         | `hybrid_bpe_rnnt` | 10.04              | 318                 | 48                    | 0.8           |
| [NVIDIA English Fast Conformer](https://huggingface.co/nvidia/stt_en_fastconformer_transducer_large) | `nvidia/stt_en_fastconformer_transducer_large` | en   | No          | `rnnt_bpe`        | 7.5                | 367                 | 48                    | 0.8           |
| [NVIDIA Parakeet TDT 0.6b](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)                       | `nvidia/parakeet-tdt-0.6b-v2`                  | en   | Yes         | `rnnt_bpe`        | Best EN            | 252                 | 16                    | 2.7           |
| [NVIDIA Parakeet CTC 1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b)                          | `nvidia/parakeet-ctc-1.1b`                     | en   | No          | `ctc_bpe`         | 6.53               | 180                 | 12                    | 4.4           |
| [NVIDIA Nemotron ASR Streaming 0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)  | `nvidia/nemotron-3.5-asr-streaming-0.6b`       | multi | Yes        | `rnnt_bpe`        | —                  | —                   | —                     | ~2.5          |
| [NVIDIA Parakeet TDT 0.6b v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)                    | `nvidia/parakeet-tdt-0.6b-v3`                  | multi (25 EU) | Yes | `rnnt_bpe`        | —                  | —                   | —                     | ~2.7          |
| [NVIDIA Canary 1b Flash](https://huggingface.co/nvidia/canary-1b-flash)                              | `nvidia/canary-1b-flash`                       | en/de/es/fr | Yes   | multitask¹        | —                  | —                   | —                     | ~4.5          |

¹ Canary is an encoder-decoder multitask model (ASR + speech translation, `EncDecMultiTaskModel`). `ARCHITECTURE` is ignored for it; pick the spoken language with `LANGUAGE` (e.g. `LANGUAGE=fr`). `—` = not benchmarked here (see the model card).

More models available on [NVIDIA HuggingFace](https://huggingface.co/nvidia).

Hybrid models can do both CTC and RNNT decoding. Add `_ctc` or `_rnnt` to `hybrid_bpe` to choose. CTC is less accurate but faster (see table above).

**Streaming-capable models.** Models trained for cache-aware streaming — e.g. `nvidia/nemotron-3.5-asr-streaming-0.6b` and the `nvidia/stt_*_fastconformer_*_streaming_*` family — are served in `websocket` mode through NeMo's **native** cache-aware decoding (low latency, accurate). Every other (offline) model can still be served in `websocket` mode through a **buffered/simulated** streaming path. The engine detects which to use automatically from the model's attention type (see [Streaming Tuning](#streaming-tuning)).

## NeMo-Specific Configuration

### NUM_THREADS

Number of threads per worker when running on CPU. Transcription speed does not scale linearly:

| NUM_THREADS | Time (4m30s file, `linagora/linto_stt_fr_fastconformer`) |
| ----------- | -------------------------------------------------------- |
| 2           | 38s                                                      |
| 4           | 25.4s                                                    |
| 8           | 18.1s                                                    |
| 16          | 16s                                                      |

### CONCURRENCY

Maximum number of parallel requests plus one (`CONCURRENCY=0` = 1 worker, `CONCURRENCY=1` = 2 workers).

- **CPU**: `NUM_THREADS * CONCURRENCY <= host threads`. Example: `NUM_THREADS=4` on 8-thread machine → max `CONCURRENCY=1`.
- **GPU**: Use `CONCURRENCY=0` (no parallel requests). For multi-file, run 1 container per GPU with `SERVICE_MODE=task`.

### LONG_FILE Parameters

Splits long files to avoid OOM. Audio is processed in parallel (2 chunks at a time). Values depend on available VRAM/RAM and should be as high as possible.

Example with 16GB VRAM GPU and `linagora/linto_stt_fr_fastconformer`:

- `LONG_FILE_THRESHOLD=540` (9 minutes)
- `LONG_FILE_CHUNK_LEN=360` (6 minutes)
- `LONG_FILE_CHUNK_CONTEXT_LEN=5` (5s overlap at each boundary)

### Attention Context (ATT_CONTEXT_SIZE / ATT_CONTEXT_SIZE_RIGHT)

The encoder's attention context, as `[left, right]` in encoder frames (~80 ms each).
`ATT_CONTEXT_SIZE` is the left context, `ATT_CONTEXT_SIZE_RIGHT` the right (look-ahead).
A larger right context means **more accuracy but more latency**. Defaults are
model-dependent (resolved at load time):

- **Offline models**: `ATT_CONTEXT_SIZE` defaults to `128` (the model is converted to
  local attention via `rel_pos_local_attn`), `ATT_CONTEXT_SIZE_RIGHT` to the same value.
- **Cache-aware streaming models**: the model keeps its **trained** attention when
  neither is set. Set `ATT_CONTEXT_SIZE_RIGHT` to trade latency for accuracy — but
  only specific values are valid (model-specific). For example,
  `nvidia/nemotron-3.5-asr-streaming-0.6b` supports `0, 3, 6, 13`
  (80 ms → 1.12 s look-ahead); `13` is the most accurate.

## Streaming Tuning

In `websocket` mode the engine uses one of two paths, chosen automatically:

- **Native cache-aware streaming** for models trained for it (see *Streaming-capable
  models* above). The model decodes natively; latency/accuracy is governed by
  `ATT_CONTEXT_SIZE_RIGHT`, and `STREAMING_NATIVE_PARTIAL_INTERVAL` sets how often
  partial results are emitted. The `STREAMING_*` knobs below do **not** apply.
- **Buffered (simulated) streaming** for offline models, re-transcribing a sliding
  buffer and committing stable words. The `STREAMING_*` knobs below tune it.

`STREAMING_PAUSE_FOR_FINAL` controls silence duration before a final result. If the model emits punctuation, finals are primarily triggered by punctuation; otherwise this is the main trigger. Adjust based on speech type.

### Low Latency Example (English)

```
SERVICE_MODE=websocket
MODEL=nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms_pc
ARCHITECTURE=hybrid_bpe_ctc
DEVICE=cuda
STREAMING_MIN_CHUNK_SIZE=0.5
STREAMING_BUFFER_TRIMMING_SEC=5
STREAMING_PAUSE_FOR_FINAL=1.2
STREAMING_MAX_WORDS_IN_BUFFER=6
STREAMING_MAX_PARTIAL_ACTUALIZATION_PER_SECOND=4
```

- Punctuation: yes | Latency: ~1s GPU, ~2s CPU (16 threads) | VRAM: ~2GB

### Low Latency Example (French)

```
SERVICE_MODE=websocket
MODEL=linagora/linto_stt_fr_fastconformer
ARCHITECTURE=hybrid_bpe_ctc
DEVICE=cuda
STREAMING_MIN_CHUNK_SIZE=0.5
STREAMING_BUFFER_TRIMMING_SEC=8
STREAMING_PAUSE_FOR_FINAL=1.5
STREAMING_MAX_WORDS_IN_BUFFER=5
STREAMING_MAX_PARTIAL_ACTUALIZATION_PER_SECOND=4
```

- Punctuation: no ([add recasepunc](../../../README.md#punctuation-model-recasepunc)) | Latency: ~1.4s GPU, ~2.4s CPU (16 threads) | VRAM: ~2.5GB

### High Latency Example (English, better accuracy)

```
SERVICE_MODE=websocket
MODEL=nvidia/parakeet-tdt-0.6b-v2
ARCHITECTURE=rnnt_bpe
DEVICE=cuda
STREAMING_MIN_CHUNK_SIZE=1
STREAMING_BUFFER_TRIMMING_SEC=15
STREAMING_PAUSE_FOR_FINAL=1.0
STREAMING_MAX_WORDS_IN_BUFFER=10
STREAMING_MAX_PARTIAL_ACTUALIZATION_PER_SECOND=3
```

- Punctuation: yes | Latency: ~2.5s | VRAM: ~4.5GB

## License

AGPLv3 (see LICENSE).

## Acknowledgments

- [NeMo](https://github.com/NVIDIA/NeMo)
- [SpeechBrain](https://github.com/speechbrain/speechbrain)
- [TorchAudio](https://github.com/pytorch/audio)
- [Whisper_Streaming](https://github.com/ufal/whisper_streaming)
