# LinTO-STT-NeMo

ASR API built on the [NeMo toolkit](https://github.com/NVIDIA/NeMo). Supports offline and real-time transcription.

> See the [main README](../../../README.md) for API docs, Docker options, and serving modes.
> See [ENV.md](../../../ENV.md) for all environment variables.

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

Add `-e DEVICE=cuda --gpus all` for GPU. Test with:

```sh
python test/test_streaming.py -v --audio_file tests/bonjour.wav
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

More models available on [NVIDIA HuggingFace](https://huggingface.co/nvidia).

Hybrid models can do both CTC and RNNT decoding. Add `_ctc` or `_rnnt` to `hybrid_bpe` to choose. CTC is less accurate but faster (see table above).

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

## Streaming Tuning

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
