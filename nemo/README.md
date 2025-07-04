# LinTO-STT-NeMo

LinTO-STT-NeMo is an Automatic Speech Recognition (ASR) API built on the [NeMo toolkit](https://github.com/NVIDIA/NeMo).

You can use LinTO-STT-NeMo as a standalone transcription service or integrate it into a microservices infrastructure via a message broker connector. 

It supports both offline and real-time transcription modes.

Try the LinTO-STT NeMo API—powered by the [LinTO French Fast Conformer model](https://huggingface.co/linagora/linto_stt_fr_fastconformer), directly in your browser via LinTO Studio.

## Quick Start

### Prerequisites

- Install [Docker](https://www.Docker.com/products/Docker-desktop/) and ensure it is running properly.

- To enable GPU capabilities, you must also install
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). You need at least cuda 12.6.

- Ensure at least 15GB of disk space is available to build the Docker image.

- Smaller models require at least 500MB of disk space, while larger ones may need up to 5GB.

### Pull or build

Pull the image :
```sh
docker pull lintoai/linto-stt-nemo
```
or build it
```sh
docker build . -f nemo/Dockerfile -t linto-stt-nemo
```

### Run file transcription API

This API allows you to transcribe audio files through standard HTTP requests. Default API values are defined in [.envdefault](https://github.com/linto-ai/linto-stt/blob/master/nemo/.envdefault), which can serve as a template for your own configuration.

Run the API to transcribe in English using in:
```sh
docker run -p 8080:80 --name linto-stt-nemo -e SERVICE_MODE=http -e MODEL=nvidia/parakeet-tdt-0.6b-v2 -e ARCHITECTURE=rnnt_bpe lintoai/linto-stt-nemo
```

or transcribe in French using:
```sh
docker run -p 8080:80 --name linto-stt-nemo -e SERVICE_MODE=http -e MODEL=linagora/linto_stt_fr_fastconformer -e ARCHITECTURE=hybrid_bpe lintoai/linto-stt-nemo
```

If a GPU is available, add `--gpus all` to the command before the image name.

Once the API is running, you can test it using:
```sh
curl -X POST "http://localhost:8080/transcribe" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@test/bonjour.wav;type=audio/wav"
```

### Run streaming transcription API

The real-time transcription (streaming) API is accessible via a WebSocket connection ([see](#websocket---streaming)). Default API values are defined in [.envdefault](https://github.com/linto-ai/linto-stt/blob/master/nemo/.envdefault), which can serve as a template for your own configuration.

Run the API to transcribe in English in real-time using:
```sh
docker run -p 8080:80 --name linto-stt-nemo -e SERVICE_MODE=websocket -e MODEL=nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms_pc -e ARCHITECTURE=hybrid_bpe lintoai/linto-stt-nemo
```

or transcribe in French using:
```sh
docker run --rm -it -p 8080:80 --name linto-stt-nemo -e SERVICE_MODE=websocket -e MODEL=linagora/linto_stt_fr_fastconformer -e ARCHITECTURE=hybrid_bpe lintoai/linto-stt-nemo
```
Note: this French model does not include punctuation. See [PUNCTUATION_MODEL environment variable](#punctuation_model-environment-variable) if you want to add punctuation.

If you have a GPU, you can add `-e DEVICE=cuda --gpus all` to the command. 


Once the API is running, you can test it using:
```sh
python test/test_streaming.py -v --audio_file test/bonjour.wav
```

# Scenarios

### SERVICE_MODE
![Serving Modes](https://i.ibb.co/qrtv3Z6/platform-stt.png)

STT service can be used in three ways:
* Through an [HTTP API](#http---offline) using the **http**'s mode.
* Through a [message broker](#celery-task---offline) using the **task**'s mode.
* Through a [websocket server](#websocket---streaming) using the **websocket**'s mode.

Mode is specified using the .env value or environment variable ```SERVICE_MODE```.
```bash
SERVICE_MODE=http
```

## Common setup for all services

### Docker options

- To enable GPU capabilities, add ```--gpus all```.
  Note that you can use environment variable `DEVICE=cuda` to make sure GPU is used (and maybe set `CUDA_VISIBLE_DEVICES` if there are several available GPU cards).
- To mount a local cache folder `<CACHE_PATH>` (e.g. "`$HOME/.cache`") and avoid downloading models each time,
  use ```-v <CACHE_PATH>:/var/www/.cache```.
- If you use `MODEL=/opt/model.nemo` environment variable, you may want to mount the model file (or folder) with option ```-v <ASR_PATH>:/opt/model.nemo```.
- To avoid file permission issues with mounted volumes you can set `USER_ID` and `GROUP_ID` environment variables (default to `33`, `www-data` user). If specified, the cache folder will be in `/home/appuser/.cache` instead of `/var/www/.cache`.

For example:
```sh
docker run -p 8080:80 --name linto-stt-nemo -e SERVICE_MODE=websocket -e MODEL=/opt/models/linto_stt_fr_fastconformer.nemo -e ARCHITECTURE=hybrid_bpe -e DEVICE=cuda -e PUNCTUATION_MODEL=/opt/models/fr.24000 -e USER_ID=$(id -u) -e GROUP_ID=$(id -g) --gpus all -v ~/models:/opt/models lintoai/linto-stt-nemo
```

Will launch a Websocket server as the host user on port 8080, using the GPU and the French model `linagora/linto_stt_fr_fastconformer`, and mount the folder `~/models` to `/opt/models`. So you must have the model `linto_stt_fr_fastconformer.nemo` in `~/models` and the punctuation model `fr.24000` in `~/models`. See [PUNCTUATION_MODEL environment variable](#punctuation_model-environment-variable) for more details about punctuation models.

### Parameters

| PARAMETER | DESCRIPTION | EXAMPLE |
|---|---|---|
| [MODEL](#model-environment-variable) | (Required) Path to a NeMo model or HuggingFace identifier. | `nvidia/parakeet-ctc-1.1b` \| `linagora/linto_stt_fr_fastconformer` \| \<ASR_PATH\> \| ... |
| ARCHITECTURE | (Required) The architecture of the model used. Suported (and tested) architectures are CTC, Hybrid and RNNT models. | `hybrid_bpe` \| `ctc_bpe` \| `rnnt_bpe` \| ... |
| DEVICE | Device to use for the model (by default, GPU/CUDA is used if it is available, CPU otherwise) | `cpu` \| `cuda` |
| [NUM_THREADS](#num_threads-environment-variable) | Number of threads (maximum) to speed up the transcription, when running on CPU. Default is `torch.get_num_threads()` | `1` \| `4` \| ... |
| CUDA_VISIBLE_DEVICES | GPU device index to use, when running on GPU/CUDA. We also recommend to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` on multi-GPU machines | `0` \| `1` \| `2` \| ... |
| USER_ID | User ID to run the service as. Default is `33` | `1000` |
| GROUP_ID | Group ID to run the service as. Default is `33` | `1000` |
| VAD | Voice Activity Detection method. VAD is used to detect the presence of human speech in an audio stream. Use "false" to disable. Default is `auditok`. | `true` \| `false` \| `1` \| `0` \| `auditok` \| `silero`
| VAD_DILATATION | How much (in sec) to enlarge each speech segment detected by the VAD. Default is `0.5` | `0.1` \| `0.5` \| ...
| VAD_MIN_SPEECH_DURATION | Minimum duration (in sec) of a speech segment. Default is `0.1` | `0.1` \| `0.5` \| ...
| VAD_MIN_SILENCE_DURATION | Minimum duration (in sec) of a silence segment. Default is `0.1` | `0.1` \| `0.5` \| ...

#### MODEL environment variable

The model will be downloaded (if not already present) from Hugging Face to `/var/www/.cache` with rights `drwxr-xr-x` (owned by `www-data`, see [`USER_ID` and `GROUP_ID`](#docker-options)) and loaded into memory when the server starts.
To preload the model,
you may want to download one of NeMo models:
| Model name | Huggingface ID | Language | Uppercase letters and punctuation | Architecture | [WER (lower is better)](https://en.wikipedia.org/wiki/Word_error_rate) on Common Voice| [RTFx (higher is better)](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) (on a RTX 4090 laptop) | RTFX on CPU (on 16 threads) | VRAM/RAM (GB) |
|---|---|---|---|---|---|---|---|---|
| [LinTO French Large Fast Conformer (by LINAGORA)](https://huggingface.co/linagora/linto_stt_fr_fastconformer) | `MODEL=linagora/linto_stt_fr_fastconformer` | fr | X | `ARCHITECTURE=hybrid_bpe_rnnt` | 8.96 | 318 | 48 | 0.8 |
| [LinTO French Large Fast Conformer (by LINAGORA)](https://huggingface.co/linagora/linto_stt_fr_fastconformer) | `MODEL=linagora/linto_stt_fr_fastconformer` | fr | X | `ARCHITECTURE=hybrid_bpe_ctc` | 10.53 | 734 | 60 | 0.8|
| [French Large Fast Conformer (by NVIDIA)](https://huggingface.co/nvidia/stt_fr_fastconformer_hybrid_large_pc) | `MODEL=nvidia/stt_fr_fastconformer_hybrid_large_pc` | fr | V | `ARCHITECTURE=hybrid_bpe_rnnt` | 10.04| 318 | 48 |0.8|
| [English Large Fast Conformer (by NVIDIA)](https://huggingface.co/nvidia/stt_en_fastconformer_transducer_large) | `MODEL=nvidia/stt_en_fastconformer_transducer_large` | en | X | `ARCHITECTURE=rnnt_bpe` | 7.5 | 367 | 48 | 0.8 |
| [English XL Fast Conformer (by NVIDIA)](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | `MODEL=nvidia/parakeet-tdt-0.6b-v2` | en | V | `ARCHITECTURE=rnnt_bpe` | Should be the best in English | 252 | 16 | 2.7 |
| [English XXL Fast Conformer (by NVIDIA)](https://huggingface.co/nvidia/parakeet-ctc-1.1b) | `MODEL=nvidia/parakeet-ctc-1.1b` | en | X | `ARCHITECTURE=ctc_bpe` | 6.53 | 180 | 12 | 4.4 |

More stt models are available in [NVIDIA](https://huggingface.co/nvidia) huggingface.

Hybrid models like [LinTO French Large Fast Conformer (by LINAGORA)](https://huggingface.co/linagora/linto_stt_fr_fastconformer) can do both `ctc` and `rnnt` decoding methods, so you can choose which one you want to use by adding `ctc` to get `hybrid_bpe_ctc` for example. `ctc` is less accurate but it runs faster `rnnt` (see table above).

#### NUM_THREADS environment variable

Set the number of threads per [worker](#concurrency-environment-variable) using the `NUM_THREADS` environment variable. with the `NUM_THREADS` environment variable. The default value is the number of threads available on the host machine. Having a higher `NUM_THREADS` will speed up the transcription. Note that transcription speed does not scale linearly with the number of threads.
For example, transcribing 4mins30s with model `MODEL=linagora/linto_stt_fr_fastconformer` took:
- 38 seconds with `NUM_THREADS=2`
- 25.4 seconds with `NUM_THREADS=4`
- 18.1 seconds with `NUM_THREADS=8`
- 16 seconds with `NUM_THREADS=16`

## HTTP serving mode - File transcription

The HTTP serving mode deploys a HTTP server and a swagger-ui to allow transcription request on a dedicated route. You can send WAV files to the server. Default API values are defined in [.envdefault](https://github.com/linto-ai/linto-stt/blob/master/nemo/.envdefault), which can serve as a template for your own configuration.

The SERVICE_MODE value in the .env should be set to ```http```.

### File transcription Parameters

See [Common parameters](#parameters) for other parameters.
| PARAMETER | DESCRIPTION | EXAMPLE |
|---|---|---|
| [CONCURRENCY](#concurrency-environment-variable) | Maximum number of parallel requests plus one. For example CONCURRENCY=0 means 1 worker, CONCURRENCY=1 means 2 workers, etc. | `2` |
| [LONG_FILE_THRESHOLD](#long_file-environment-variables) | A file longer than that will be split into smaller chunks to avoid Out of Memory issues. This value depends on your VRAM/RAM amount. Default is 480s (8mins) | `480` \| `240`
| LONG_FILE_CHUNK_LEN | For long form transcription, size of the chunks into which the audio is splitted. This value depends on your VRAM/RAM amount. Default is 240s (4mins) | `240` \| `120`
| LONG_FILE_CHUNK_CONTEXT_LEN | For long form transcription, size of the context added at the beginning and end of each chunk. Default is 10s (10s before and after the chunk) | \| `5` \| `3`

#### CONCURRENCY environment variable

As said in the table above, it is the maximum number of parallel requests plus one. For example CONCURRENCY=0 means 1 worker, CONCURRENCY=1 means 2 workers, etc.
How to choose the number of workers ?
- On CPU : `NUM_THREADS*CONCURRENCY<=Number of threads of the host machine`. For example, with `NUM_THREADS=4` and the host machine has 8 threads, then you can have up to 2 workers, so CONCURRENCY=1.
- ON GPU : CONCURRENCY=0 because there are no parallel requests on GPU. If you want to transcribe multiple files at the same time, you can run 1 container per GPU using `SERVICE_MODE=task` and use [LinTO Transcription Service](https://github.com/linto-ai/linto-transcription-service) to handle the requests.

### LONG_FILE environment variables

The goal of these variables is to avoid Out of Memory issues when transcribing long files. The idea is to split the file into smaller chunks and transcribe them separately. The audio is processed in parallel, transcribing two chunks at a time before merging them into the final transcription. It is faster to transcribe 2 smaller chunks than 1 big one, that's why `LONG_FILE_CHUNK_LEN` exists. `LONG_FILE_CHUNK_LEN` should be smaller than `LONG_FILE_THRESHOLD`. The context is added at the beginning and end of each chunk to avoid losing words at the beginning and end of the chunk. The values of these variables are in seconds.
Their value should be the highest possible. For example, with a GPU with 16GB of VRAM and `MODEL=linagora/linto_stt_fr_fastconformer`, you can set:
- `LONG_FILE_THRESHOLD=540` (9mins)
- `LONG_FILE_CHUNK_LEN=360` (6mins)
- `LONG_FILE_CHUNK_CONTEXT_LEN=5` (5s before and after each chunk)


### Usages
#### /healthcheck
Returns the state of the API

Method: GET

Returns "1" if healthcheck passes.

#### /transcribe
Transcription API

* Method: POST
* Response content: text/plain or application/json
* File: An Wave file 16b 16Khz
* Language (optional): Override environment variable `LANGUAGE`

Return the transcribed text using "text/plain" or a json object when using "application/json" structure as followed:
```json
{
    "text" : "This is the transcription as text",
    "words": [
        {
        "word" : "This",
        "start": 0.0,
        "end": 0.124,
        "conf": 0.82341
        },
        ...
    ],
    "language": "en",
    "confidence-score": 0.879
}
```

#### /docs
The /docs route offers a OpenAPI/swagger interface.

## CELERY TASK - File transcription

### (micro-service) Service broker and shared folder
In task mode, STT operations are triggered via tasks sent through a message broker. Supported message brokers include RabbitMQ, Redis, and Amazon SQS.
Additionally, to prevent large audio files from passing through the message broker, a shared storage folder (SHARED_FOLDER) is used. through the message broker, STT-Worker use a shared storage folder (SHARED_FOLDER).

The celery tasks can be managed using [LinTO Transcription service](https://github.com/linto-ai/linto-transcription-service).

See [Common parameters](#parameters) for other parameters.
| PARAMETER | DESCRIPTION | EXAMPLE |
|---|---|---|
| SERVICE_NAME | (For the task mode only) queue's name for task processing | `my-stt` |
| SERVICE_BROKER | (For the task mode only) URL of the message broker | `redis://my-broker:6379` |
| BROKER_PASS | (For the task mode only) broker password | `my-password` \| (empty) |

Shared parameters with [HTTP service mode](#file-transcription-parameters):
| PARAMETER | DESCRIPTION | EXAMPLE |
|---|---|---|
| [CONCURRENCY](#concurrency-environment-variable) | Maximum number of parallel requests plus one. For example CONCURRENCY=0 means 1 worker, CONCURRENCY=1 means 2 workers, etc. | `2` |
| [LONG_FILE_THRESHOLD](#long_file-environment-variables) | A file longer than that will be split into smaller chunks to avoid Out of Memory issues. This value depends on your VRAM/RAM amount . Default is 480s (8mins) | `480` \| `240`
| LONG_FILE_CHUNK_LEN | For long form transcription, size of the chunks into which the audio is splitted. This value depends on your VRAM/RAM amount. Default is 240s (4mins) | `240` \| `120`
| LONG_FILE_CHUNK_CONTEXT_LEN | For long form transcription, size of the context added at the beginning and end of each chunk. Default is 10s (10s before and after the chunk) | `10` \| `5`

### Usage

See [HTTP usage](#usages).

#### Through the message broker

STT-Worker accepts requests with the following arguments:
```file_path: str, with_metadata: bool```

* <ins>file_path</ins>: Is the location of the file within the shared_folder. /.../SHARED_FOLDER/{file_path}
* <ins>with_metadata</ins>: If True, words timestamps and confidence will be computed and returned. If false, the fields will be empty.

#### Return format
On a successfull transcription the returned object is a json object structured as follow:
```json
{
    "text" : "This is the transcription as text",
    "words": [
        {
        "word" : "This",
        "start": 0.0,
        "end": 0.124,
        "conf": 0.82341
        },
        ...
    ],
    "confidence-score": 0.879
}
```

* The <ins>text</ins> field contains the raw transcription.
* The <ins>word</ins> field contains each word with their time stamp and individual confidence. (Empty if with_metadata=False)
* The <ins>confidence</ins> field contains the overall confidence for the transcription. (0.0 if with_metadata=False)

## Websocket - Streaming

In WebSocket mode, only the streaming transcription service is deployed. Default API values are defined in [.envdefault](https://github.com/linto-ai/linto-stt/blob/master/nemo/.envdefault), which can serve as a template for your own configuration.


The SERVICE_MODE value in the .env should be set to ```websocket```. 

### Usage

The data exchange process follows these steps:
1. Client send a json {"config": {"sample_rate":16000}}.
2. Client send audio chunk (go to 3- ) or {"eof" : 1} (go to 5-).
3. Server send either a partial result {"partial" : "this is a "} or a final result {"text": "this is a transcription"}.
4. Back to 2-
5. Server send a final result and close the connection.

### Streaming parameters

See [Common parameters](#parameters) for other parameters.
| PARAMETER | DESCRIPTION | EXAMPLE |
|---|---|---|
| STREAMING_PORT | The listening port for ingoing WS connections. Default is 80 | `80` |
| STREAMING_MIN_CHUNK_SIZE | The minimal size of the buffer (in seconds) before transcribing.  Used to lower the hardware usage (low value=high usage, high value=low usage). Default is 0.5 | `0.5` \| `26` \| ... |
| STREAMING_BUFFER_TRIMMING_SEC | The maximum targeted length of the buffer (in seconds). It tries to cut after a transcription has been made (bigger value=higher hardware usage). Default is 8 | `8` \| `10` \| ... |
| [STREAMING_PAUSE_FOR_FINAL](#streaming_pause_for_final-environment-variable) | The minimum duration of silence (in seconds) needed to be able to output a final. Default is 1.5 | `0.5` \| `2` \| ... |
| STREAMING_TIMEOUT_FOR_SILENCE | If a VAD is applied externally, this parameter will allow the server to find the silence (silences are used to output `final`). The `packet duration` is determined from the first packet. If a packet is not received during `packet duration * STREAMING_TIMEOUT_FOR_SILENCE` it considers that a silence (lasting the packet duration) is present. If specified, value should be between 1 and 2 (2 times the duration of a packet). Default is deactivated | `1.8` \| ... |
| STREAMING_MAX_WORDS_IN_BUFFER | How much words can stay in the buffer. It means how much words can be changed. Default is 4 | `4` \| `2` \| ... |
| STREAMING_MAX_PARTIAL_ACTUALIZATION_PER_SECOND | The maximum of messages that can be sent by the server to the client in one second. Default is 4, put 0 to deactivate it | `3` \| ... |
| [PUNCTUATION_MODEL](#punctuation_model-environment-variable) | Path to a recasepunc model, for recovering punctuation and upper letter in streaming. Use it if your model doesn't output punctuation and upper case letters.  | /opt/PUNCT |

#### STREAMING_PAUSE_FOR_FINAL environment variable

The `STREAMING_PAUSE_FOR_FINAL` value will depend on your type of speech. On prepared speech for example, you can probably lower it whereas on real discussions you can leave it as default or increase it. Without punctuations, 1.5 seconds is a good value. With punctuations, you can lower it to 1 second because a final will be outputted only when a punctuation is detected.

#### PUNCTUATION_MODEL environment variable

If you use a model that outputs lower-case text without punctuations, you can specify a recasepunc model (which must be in version 0.4 at least).
Some recasepunc models trained on [Common Crawl](http://data.statmt.org/cc-100/) are available on [recasepunc](https://github.com/benob/recasepunc/releases/) for the following the languages:
* French
  * [fr.24000](https://github.com/benob/recasepunc/releases/download/0.4/fr.24000)
* English
  * [en.22000](https://github.com/benob/recasepunc/releases/download/0.4/en.22000)

After downloading a recasepunc model, you can mount it as a volume and specify its location within the Docker container using the `PUNCTUATION_MODEL` environment variable.
```
-v <PATH>/models/lm/fr.24000:/opt/models/lm/fr.24000
```

### Example with lowest latency

todo example config with latency expected, used resources, etc.

#### English
Here is a config for low latency streaming in English that you can use as a starting point:
```
SERVICE_MODE=websocket
STREAMING_PORT=80
DEVICE=cuda

MODEL=nvidia/stt_en_fastconformer_hybrid_medium_streaming_80ms_pc
ARCHITECTURE=hybrid_bpe_ctc
STREAMING_MIN_CHUNK_SIZE=0.5
STREAMING_BUFFER_TRIMMING_SEC=5
STREAMING_PAUSE_FOR_FINAL=1.0
STREAMING_TIMEOUT_FOR_SILENCE=
STREAMING_MAX_WORDS_IN_BUFFER=6
STREAMING_MAX_PARTIAL_ACTUALIZATION_PER_SECOND=4
```

With this config:
- Include punctuation: yes
- Latency: around 1s (depends on your hardware)
- VRAM: around 2GB

Same config with `DEVICE=cpu` and `NUM_THREADS=16`:
- Latency: around 2s (depends on your hardware)

#### French

```
SERVICE_MODE=websocket
STREAMING_PORT=80
DEVICE=cuda

MODEL=linagora/linto_stt_fr_fastconformer
ARCHITECTURE=hybrid_bpe_ctc
STREAMING_MIN_CHUNK_SIZE=0.5
STREAMING_BUFFER_TRIMMING_SEC=8
STREAMING_PAUSE_FOR_FINAL=1.0
STREAMING_TIMEOUT_FOR_SILENCE=
STREAMING_MAX_WORDS_IN_BUFFER=5
STREAMING_MAX_PARTIAL_ACTUALIZATION_PER_SECOND=4
```

With this config:
- Include punctuation: no ([see](#punctuation_model-environment-variable) and the [example](#docker-options))
- Latency: around 1.4s (depends on your hardware)
- VRAM: around 2.5GB

Same config with `DEVICE=cpu` and `NUM_THREADS=16`:
- Latency: around 2.4s (depends on your hardware)

### Example with high latency

Here is a config for high latency streaming (for better accuracy) in English that you can use as a starting point:

```
SERVICE_MODE=websocket
STREAMING_PORT=80
DEVICE=cuda

MODEL=nvidia/parakeet-tdt-0.6b-v2
ARCHITECTURE=rnnt_bpe
STREAMING_MIN_CHUNK_SIZE=1
STREAMING_BUFFER_TRIMMING_SEC=15
STREAMING_PAUSE_FOR_FINAL=1.0
STREAMING_TIMEOUT_FOR_SILENCE=
STREAMING_MAX_WORDS_IN_BUFFER=10
STREAMING_MAX_PARTIAL_ACTUALIZATION_PER_SECOND=3
```

With this config:
- Include punctuation: yes
- Latency: around 2.5s (depends on your hardware)
- VRAM: around 4.5GB

## License
This project is licensed under AGPLv3 (see LICENSE).

## Acknowledgments

* [NeMo](https://github.com/NVIDIA/NeMo)
* [SpeechBrain](https://github.com/speechbrain/speechbrain)
* [TorchAudio](https://github.com/pytorch/audio)
* [Whisper_Streaming](https://github.com/ufal/whisper_streaming)