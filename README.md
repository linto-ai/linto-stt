# LINTO-PLATFORM-STT
LinTO-platform-stt is the transcription service within the [LinTO stack](https://github.com/linto-ai/linto-platform-stack).

The STT-worker is configured with an acoustic model and a language model to perform Speech-To-Text tasks with high efficiency.

LinTO-platform-stt can either be used as a standalone transcription service or deployed within a micro-services infrastructure using a message broker connector.

## Pre-requisites

### Hardware
To run the transcription models you'll need:
* At least 7Go of disk space to build the docker image.
* 500MB-3GB-7GB of RAM depending on the model used (small-medium-large).
* One CPU per worker. Inference time scales on CPU performances. 

### Model
The transcription service relies on 2 models:
* An acoustic model.
* A language model (or decoding graph).

We provide some models on [dl.linto.ai](https://dl.linto.ai/downloads/model-distribution/).

### Docker
The transcription service requires docker up and running.

### (micro-service) Service broker and shared folder
The STT only entry point in job mode are tasks posted on a message broker. Supported message broker are RabbitMQ, Redis, Amazon SQS.
On addition, as to prevent large audio from transiting through the message broker, STT-Worker use a shared storage folder.

## Deploy linto-platform-stt
linto-platform-stt can be deployed three ways:
* As a standalone transcription service through an HTTP API.
* As a micro-service connected to a message broker.

**1- First step is to build the image:**

```bash
git clone https://github.com/linto-ai/linto-platform-stt.git
cd linto-platform-stt
docker build . -t linto-platform-stt:latest
```

**2- Download the models**

Have the acoustic and language model ready at AM_PATH and LM_PATH.

### HTTP API

```bash
docker run --rm \
-p HOST_SERVING_PORT:80 \
-v AM_PATH:/opt/models/AM \
-v LM_PATH:/opt/models/LM \
--env SERVICE_NAME=stt \
--env LANGUAGE=en_US \
--env SERVICE_MODE=http \
--env CONCURRENCY=10 \
linto-platform-stt:latest
```

This will run a container providing an http API binded on the host HOST_SERVING_PORT port.

**Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| HOST_SERVING_PORT | Host serving port | 80 |
| AM_PATH | Path to the acoustic model | /my/path/to/models/AM_fr-FR_v2.2.0 |
| LM_PATH | Path to the language model | /my/path/to/models/AM_fr-FR_v2.2.0 |
| LANGUAGE | Language code as a BCP-47 code  | en-US, fr_FR, ... |
| CONCURRENCY | Number of worker (1 worker = 1 cpu) | 4 |

### Micro-service within LinTO-Platform stack
>LinTO-platform-stt can be deployed within the linto-platform-stack through the use of linto-platform-services-manager. Used this way, the container spawn celery worker waiting for transcription task on a message broker.
>LinTO-platform-stt in task mode is not intended to be launch manually.
>However, if you intent to connect it to your custom message's broker here are the parameters:

You need a message broker up and running at MY_SERVICE_BROKER.

```bash
docker run --rm \
-v AM_PATH:/opt/models/AM \
-v LM_PATH:/opt/models/LM \
-v SHARED_AUDIO_FOLDER:/opt/audio \
--env SERVICES_BROKER=MY_SERVICE_BROKER \
--env BROKER_PASS=MY_BROKER_PASS \
--env SERVICE_NAME=stt \
--env LANGUAGE=en_US \
--env SERVICE_MODE=task \
--env CONCURRENCY=10 \
linstt:dev
```

**Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| AM_PATH | Path to the acoustic model | /my/path/to/models/AM_fr-FR_v2.2.0 |
| LM_PATH | Path to the language model | /my/path/to/models/AM_fr-FR_v2.2.0 |
| SERVICES_BROKER | Service broker uri | redis://my_redis_broker:6379 |
| BROKER_PASS | Service broker password (Leave empty if there is no password) | my_password |
| SERVICE_NAME* | Transcription service name | my_stt |
| LANGUAGE | Transcription language | en-US |
| CONCURRENCY | Number of worker (1 worker = 1 cpu) | [ 1 -> numberOfCPU] |

(* SERVICE NAME needs to be the same as the linto-platform-transcription-service if used.)


## Usages

### HTTP API

#### /healthcheck

Returns the state of the API

Method: GET

Returns "1" if healthcheck passes.

#### /transcribe

Transcription API

* Method: POST
* Response content: text/plain or application/json
* File: An Wave f ile 16b 16Khz

Return the transcripted text using "text/plain" or a json object when using "application/json" structure as followed:
```json
{
  "text" : "This is the transcription",
  "words" : [
    {"word":"This", "start": 0.123, "end": 0.453, "conf": 0.9},
    ...
  ]
  "confidence-score": 0.879
}
```

#### /docs
The /docs route offers a OpenAPI/swagger interface. 

### Through the message broker

STT-Worker accepts requests with the following arguments:
```file_path: str, with_metadata: bool```

* <ins>file_path</ins>: Is the location of the file within the shared_folder. /.../SHARED_FOLDER/{file_path}
* <ins>with_metadata</ins>: If True, words timestamps and confidence will be computed and returned. If false, the fields will be empty.

#### Return format
On a successfull transcription the returned object is a json object structured as follow:
```json
{
    "text" : "this is the transcription as text",
    "words": [
        {
        "word" : "this",
        "start": 0.0,
        "end": 0.124,
        "conf": 1.0
        },
        ...
    ],
    "confidence-score": ""
}
```

* The <ins>text</ins> field contains the raw transcription.
* The <ins>word</ins> field contains each word with their time stamp and individual confidence. (Empty if with_metadata=False)
* The <ins>confidence</ins> field contains the overall confidence for the transcription. (0.0 if with_metadata=False)

## Test
### Curl
You can test you http API using curl:
```bash 
curl -X POST "http://YOUR_SERVICE:YOUR_PORT/transcribe" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=@YOUR_FILE;type=audio/x-wav"
```

## License
This project is developped under the AGPLv3 License (see LICENSE).

## Acknowlegment.

* [Vosk, speech recognition toolkit](https://alphacephei.com/vosk/).
* [Kaldi Speech Recognition Toolkit](https://github.com/kaldi-asr/kaldi)
