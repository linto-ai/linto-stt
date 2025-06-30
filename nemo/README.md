# LinTO-STT-NeMo

LinTO-STT-NeMo is an API for Automatic Speech Recognition (ASR) based on [NeMo toolkit](https://github.com/NVIDIA/NeMo).

LinTO-STT-NeMo can either be used as a standalone transcription service or deployed within a micro-services infrastructure using a message broker connector. 

It can be used to do offline or real-time transcriptions.

You can try the LinTO-STT NeMo API, powered by the [LinTO French Fast Conformer model](https://huggingface.co/linagora/linto_stt_fr_fastconformer), directly in your browser via LinTO Studio.

## Quick Start

### Prerequisites

- Install [docker](https://www.docker.com/products/docker-desktop/) and ensure it is up and running.

- For GPU capabilities, it is also needed to install
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

- At least 15GB of disk space is required to build the docker image and the models.

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

An API for transcribing audio files, accessible via standard HTTP requests.

Run the API to transcribe in english using in:
```sh
docker run -p 8080:80 --name linto-stt-nemo --env-file nemo/.envdefault_offline lintoai/linto-stt-nemo
```

or transcribe in french using:
```sh
docker run -p 8080:80 --name linto-stt-nemo --env-file nemo/.envdefault_offline -e MODEL=linagora/linto_stt_fr_fastconformer -e ARCHITECTURE=hybrid_bpe lintoai/linto-stt-nemo
```

If you have a GPU, you can add `--gpus all` to the command.

Once the API is running, you can test it using:
```sh
curl -X POST "http://localhost:8080/transcribe" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@test/bonjour.wav;type=audio/wav"
```

### Run streaming transcription API

The real-time transcription (streaming) API is accessible via a WebSocket connection ([see](#websocket---streaming)).

Run the API to transcribe in english in real-time using:
```sh
docker run -p 8080:80 --name linto-stt-nemo --env-file nemo/.envdefault_streaming lintoai/linto-stt-nemo
```

or transcribe in french using:
```sh
docker run --rm -it -p 8080:80 --name linto-stt-nemo --env-file nemo/.envdefault_streaming -e MODEL=linagora/linto_stt_fr_fastconformer lintoai/linto-stt-nemo
```
This french model does not include punctuation. See [PUNCTUATION_MODEL environment variable](#punctuation_model-environment-variable) if you want to add punctuation.

If you have a GPU, you can add `-e DEVICE=cuda --gpus all` to the command. 


Once the API is running, you can test it using:
```sh
python test/test_streaming.py -v --audio_file test/bonjour.wav
```

# Scenarios

### SERVING_MODE
![Serving Modes](https://i.ibb.co/qrtv3Z6/platform-stt.png)

STT service can be used in three ways:
* Through an [HTTP API](#http---offline) using the **http**'s mode.
* Through a [message broker](#celery-task---offline) using the **task**'s mode.
* Through a [websocket server](#websocket---streaming) using the **websocket**'s mode.

Mode is specified using the .env value or environment variable ```SERVING_MODE```.
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

For example:
```sh
docker run -p 8080:80 --name linto-stt-nemo --env-file nemo/.envdefault_streaming -e MODEL=/opt/models/linto_stt_fr_fastconformer.nemo -e ARCHITECTURE=hybrid_bpe -e DEVICE=cuda -e PUNCTUATION_MODEL=/opt/models/fr.24000 --gpus all -v ~/models:/opt/models lintoai/linto-stt-nemo
```

Will launch a Websocket server using the GPU and the French model `linagora/linto_stt_fr_fastconformer`, and mount the folder `~/models` to `/opt/models`. So you must have the model `linto_stt_fr_fastconformer.nemo` in `~/models` and the punctuation model `fr.24000` in `~/models`. See [PUNCTUATION_MODEL environment variable](#punctuation_model-environment-variable) for more details about punctuation models.


<!-- **Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| `HOST_SERVING_PORT` | Host serving port | 8080 |
| `<CACHE_PATH>` | Path to a folder to download wav2vec alignment models when relevant | /home/username/.cache |
| `<ASR_PATH>` | Path to the NeMo model on the host machine mounted to /opt/model.nemo | /my/path/to/models/stt_fr.nemo | -->

### Parameters

| PARAMETER | DESCRIPTION | EXAMPLE |
|---|---|---|
| MODEL | (Required) Path to a NeMo model or HuggingFace identifier. | `nvidia/parakeet-ctc-1.1b` \| `linagora/linto_stt_fr_fastconformer` \| \<ASR_PATH\> \| ... |
| ARCHITECTURE | (Required) The architecture of the model used. Suported (and tested) architectures are CTC, Hybrid and RNNT models. | `hybrid_bpe` \| `ctc_bpe` \| `rnnt_bpe` \| ... |
| DEVICE | Device to use for the model (by default, GPU/CUDA is used if it is available, CPU otherwise) | `cpu` \| `cuda` |
| NUM_THREADS | Number of threads (maximum) to speed up the transcription, when running on CPU. | `1` \| `4` \| ... |
| CUDA_VISIBLE_DEVICES | GPU device index to use, when running on GPU/CUDA. We also recommend to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` on multi-GPU machines | `0` \| `1` \| `2` \| ... |
| USER_ID | Enable to use the user of the host  | blabla |
| GROUP_ID | blabla. If specified, the cache folder will be in `/home/appuser` | blabla |
| VAD | Voice Activity Detection method. VAD is used to detect the presence of human speech in an audio stream. Use "false" to disable. Default is auditok VAD. | `true` \| `false` \| `1` \| `0` \| `auditok` \| `silero`
| VAD_DILATATION | How much (in sec) to enlarge each speech segment detected by the VAD. Default is auditok 0.5 | `0.1` \| `0.5` \| ...
| VAD_MIN_SPEECH_DURATION | Minimum duration (in sec) of a speech segment. Default is 0.1 | `0.1` \| `0.5` \| ...
| VAD_MIN_SILENCE_DURATION | Minimum duration (in sec) of a silence segment. Default is 0.1 | `0.1` \| `0.5` \| ...

#### MODEL environment variable

The model will be (downloaded from huggingface if required and) loaded in memory when the server starts.
If you want to preload the model,
you may want to download one of NeMo models:
| Model | Language | Description | Architecture | [WER (lower is better)](https://en.wikipedia.org/wiki/Word_error_rate) on Common Voice| [RTFx (higher is better)](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) (on a RTX 4090 laptop) | RTFX (on CPU) | VRAM/RAM (GB) |
|---|---|---|---|---|---|---|---|
| [LinTO French Large Fast Conformer (by LINAGORA)](https://huggingface.co/linagora/linto_stt_fr_fastconformer) | fr | Robust hybrid french model | `ARCHITECTURE=hybrid_bpe_rnnt` | 8.96 | 128 | 51 | 0.76 |
| [LinTO French Large Fast Conformer (by LINAGORA)](https://huggingface.co/linagora/linto_stt_fr_fastconformer) | fr | Same model as above but using the CTC decoder | `ARCHITECTURE=hybrid_bpe_ctc` | 10.53 | 985 | 66 | 0.76|
| [French Large Fast Conformer (by NVIDIA)](https://huggingface.co/nvidia/stt_fr_fastconformer_hybrid_large_pc) | fr | Includes uppercase letters and punctuation | `ARCHITECTURE=hybrid_bpe_rnnt` | 10.04| 128 | 51 |0.76|
| [English Large Fast Conformer (by NVIDIA)](https://huggingface.co/nvidia/stt_en_fastconformer_transducer_large) | en| English model | `ARCHITECTURE=rnnt_bpe` | 7.5 | ?? | ?? | 0.76 |
| [English XXL Fast Conformer (by NVIDIA)](https://huggingface.co/nvidia/parakeet-ctc-1.1b) | en | Very large english model | `ARCHITECTURE=ctc_bpe` | 6.53 | ?? | ?? | 4.4 |


   <!-- * [LinTO French Large Fast Conformer (by LINAGORA)](https://huggingface.co/linagora/linto_stt_fr_fastconformer): Most robust French model. Use `ARCHITECTURE=hybrid_bpe`.
   * [French Large Fast Conformer (by NVIDIA)](https://huggingface.co/nvidia/stt_fr_fastconformer_hybrid_large_pc): , but is less precise than the LINAGORA model. Use `ARCHITECTURE=hybrid_bpe`
   * [English Large Fast Conformer (by NVIDIA)](https://huggingface.co/nvidia/stt_en_fastconformer_transducer_large). Use `ARCHITECTURE=rnnt_bpe`
   * [English XXL Fast Conformer (by NVIDIA)](https://huggingface.co/nvidia/parakeet-ctc-1.1b). Use `ARCHITECTURE=ctc_bpe`
   * More stt models are available in [NVIDIA](https://huggingface.co/nvidia) huggingface -->

More stt models are available in [NVIDIA](https://huggingface.co/nvidia) huggingface.

<!-- NeMo models from Hugging Face, as for instance https://huggingface.co/nvidia/parakeet-ctc-1.1b (you can either download the model or use the Hugging Face identifier `nvidia/parakeet-ctc-1.1b`). -->
Hybrid models like [LinTO French Large Fast Conformer (by LINAGORA)](https://huggingface.co/linagora/linto_stt_fr_fastconformer) can do both `ctc` and `rnnt` decoding methods, so you can choose which one you want to use by adding `ctc` to get `hybrid_bpe_ctc` for example. `ctc` is less accurate but it runs faster `rnnt` (see table above).

## HTTP serving mode - File transcription

The HTTP serving mode deploys a HTTP server and a swagger-ui to allow transcription request on a dedicated route. A [```.envdefault_offline```](.envdefault_offline) is already available with an english model selected. You can send files to the server.
The SERVICE_MODE value in the .env should be set to ```http```.

### File transcription Parameters

See [Common parameters](#parameters) for other parameters.
| PARAMETER | DESCRIPTION | EXAMPLE |
|---|---|---|
| CONCURRENCY | Maximum number of parallel requests plus one. For example CONCURRENCY=0 means 1 worker, CONCURRENCY=1 means 2 workers, etc. | `2` |
| LONG_FILE_THRESHOLD | For offline decoding, a file longer than that will be split into smaller parts (long form transcription). This value depends on your VRAM/RAM amount. Default is 480s (8mins) | `480` \| `240`
| LONG_FILE_CHUNK_LEN | For long form transcription, size of the parts into which the audio is splitted. This value depends on your VRAM/RAM amount. Default is 240s (4mins) | `240` \| `120`
| LONG_FILE_CHUNK_CONTEXT_LEN | For long form transcription, size of the context added at the begining and end of each chunk. Default is 10s (10s before and after the chunk) | `10` \| `5`

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

Return the transcripted text using "text/plain" or a json object when using "application/json" structure as followed:
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
The STT only entry point in task mode are tasks posted on a message broker. Supported message broker are RabbitMQ, Redis, Amazon SQS.
On addition, as to prevent large audio from transiting through the message broker, STT-Worker use a shared storage folder (SHARED_FOLDER).

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
| CONCURRENCY | Maximum number of parallel requests plus one. For example CONCURRENCY=0 means 1 worker, CONCURRENCY=1 means 2 workers, etc. | `2` |
| LONG_FILE_THRESHOLD | For offline decoding, a file longer than that will be split into smaller parts (long form transcription). This value depends on your VRAM/RAM amount. Default is 480s (8mins) | `480` \| `240`
| LONG_FILE_CHUNK_LEN | For long form transcription, size of the parts into which the audio is splitted. This value depends on your VRAM/RAM amount. Default is 240s (4mins) | `240` \| `120`
| LONG_FILE_CHUNK_CONTEXT_LEN | For long form transcription, size of the context added at the begining and end of each chunk. Default is 10s (10s before and after the chunk) | `10` \| `5`

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

Websocket server's mode deploy a streaming transcription service only. A [```.envdefault_streaming```](.envdefault_streaming) is already available with an english model selected.

The SERVICE_MODE value in the .env should be set to ```websocket```. 

### Usage

The exchanges are structured as followed:
1. Client send a json {"config": {"sample_rate":16000}}.
2. Client send audio chunk (go to 3- ) or {"eof" : 1} (go to 5-).
3. Server send either a partial result {"partial" : "this is a "} or a final result {"text": "this is a transcription"}.
4. Back to 2-
5. Server send a final result and close the connexion.

<!-- We recommend to use a VAD on the server side to improve the "final" quality. -->

### Streaming parameters

See [Common parameters](#parameters) for other parameters.
| PARAMETER | DESCRIPTION | EXAMPLE |
|---|---|---|
| STREAMING_PORT | (For the websocket mode) the listening port for ingoing WS connexions. Default is 80 | `80` |
| STREAMING_MIN_CHUNK_SIZE | The minimal size of the buffer (in seconds) before transcribing. Used to lower the hardware usage (low value=high usage, high value=low usage). Default is 0.5 | `0.5` \| `26` \| ... |
| STREAMING_BUFFER_TRIMMING_SEC | The maximum targeted length of the buffer (in seconds). It tries to cut after a transcription has been made (bigger value=higher hardware usage). Default is 8 | `8` \| `10` \| ... |
| STREAMING_PAUSE_FOR_FINAL | The minimum duration of silence (in seconds) needed to be able to output a final. Default is 1.5 | `0.5` \| `2` \| ... |
| STREAMING_TIMEOUT_FOR_SILENCE | If a VAD is applied externally, this parameter will allow the server to find the silence (silences are used to output `final`). The `packet duration` is determined from the first packet. If a packet is not received during `packet duration * STREAMING_TIMEOUT_FOR_SILENCE` it considers that a silence (lasting the packet duration) is present. If specified, value should be between 1 and 2 (2 times the duration of a packet). Default is deactivated | `1.8` \| ... |
| STREAMING_MAX_WORDS_IN_BUFFER | How much words can stay in the buffer. It means how much words can be changed. Default is 4 | `4` \| `2` \| ... |
| STREAMING_MAX_PARTIAL_ACTUALIZATION_PER_SECOND | How much time per seconds you want the server to send a message to the client. Default is 4 | `3` \| ... |
| PUNCTUATION_MODEL | Path to a recasepunc model, for recovering punctuation and upper letter in streaming. Use it if your model doesn't output punctuation and upper case letters.  | /opt/PUNCT |

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

The `STREAMING_PAUSE_FOR_FINAL` value will depend on your type of speech. On prepared speech for example, you can probably lower it whereas on real discussions you can leave it as default or increase it. 

## License
This project is developped under the AGPLv3 License (see LICENSE).

## Acknowlegment.

* [NeMo](https://github.com/NVIDIA/NeMo)
* [SpeechBrain](https://github.com/speechbrain/speechbrain)
* [TorchAudio](https://github.com/pytorch/audio)
* [Whisper_Streaming](https://github.com/ufal/whisper_streaming)