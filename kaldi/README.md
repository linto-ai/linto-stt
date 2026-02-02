# LinTO-STT-Kaldi

LinTO-STT-Kaldi is an API for Automatic Speech Recognition (ASR) based on models trained with [Kaldi](https://github.com/kaldi-asr/kaldi).

LinTO-STT-Kaldi can either be used as a standalone transcription service or deployed within a micro-services infrastructure using a message broker connector.

It can be used to do offline or real-time transcriptions.

You can try LinTO-STT directly in your browser via [LinTO Studio](https://studio.linto.app/).

## Pre-requisites

### Hardware
To run the transcription models you'll need:
* At least 7Go of disk space to build the docker image.
* Up to 7GB of RAM depending on the model used.
* One CPU per worker. Inference time scales on CPU performances. 

### Model
If not done alreadt, download and unzip model folders into a directory accessible from the docker container.

LinTO-STT-Kaldi accepts two kinds of ASR models:
* LinTO Acoustic and Languages models.
* Vosk models (all in one).

We provide home-cured models (v2) on [dl.linto.ai](https://doc.linto.ai/docs/developpers/apis/ASR/models).
Or you can also use Vosk models available [here](https://alphacephei.com/vosk/models).


### Docker
The transcription service requires docker up and running.

### (micro-service) Service broker and shared folder
The STT only entry point in task mode are tasks posted on a message broker. Supported message broker are RabbitMQ, Redis, Amazon SQS.
On addition, as to prevent large audio from transiting through the message broker, STT-Worker use a shared storage folder (SHARED_FOLDER).

## Deploy LinTO-STT-Kaldi

**1- First step is to build or pull the image:**

```bash
git clone https://github.com/linto-ai/linto-stt.git
cd linto-stt
docker build . -f kaldi/Dockerfile -t linto-stt-kaldi:latest
```
or

```bash
docker pull lintoai/linto-stt-kaldi
```

**2- Download the models**

Have the acoustic and language model ready at AM_PATH and LM_PATH if you are using LinTO models. If you are using a Vosk model, have it ready at MODEL.

**3- Fill the .env file**

An example of .env file is provided in [kaldi/.envdefault](https://github.com/linto-ai/linto-stt/blob/master/kaldi/.envdefault).

| PARAMETER | DESCRIPTION | EXEMPLE |
|---|---|---|
| SERVICE_MODE | STT serving mode see [Serving mode](#serving-mode) | http\|task\|websocket |
| MODEL_TYPE | Type of STT model used. | lin\|vosk |
| ENABLE_STREAMING | Using http serving mode, enable the /streaming websocket route | true\|false |
| SERVICE_NAME | Using the task mode, set the queue's name for task processing | my-stt |
| SERVICE_BROKER | Using the task mode, URL of the message broker | redis://my-broker:6379 |
| BROKER_PASS | Using the task mode, broker password | my-password |
| STREAMING_PORT | Using the websocket mode, the listening port for ingoing WS connexions.  | 80 |
| CONCURRENCY | Maximum number of parallel requests | >1 |
| PUNCTUATION_MODEL | Path to a recasepunc model, for recovering punctuation and upper letter in streaming | /opt/PUNCT |


### Serving mode 
![Serving Modes](https://i.ibb.co/qrtv3Z6/platform-stt.png)

STT can be used three ways:
* Through an [HTTP API](#http-server) using the **http**'s mode.
* Through a [message broker](#celery-task) using the **task**'s mode.
* Through a [websocket server](#websocket-server) **websocket**'s mode.

Mode is specified using the .env value or environment variable ```SERVING_MODE```.
```bash
SERVICE_MODE=http
```
### HTTP Server
The HTTP serving mode deploys a HTTP server and a swagger-ui to allow transcription request on a dedicated route.

The SERVICE_MODE value in the .env should be set to ```http```.

```bash
docker run --rm \
-p HOST_SERVING_PORT:80 \
-v AM_PATH:/opt/AM \
-v LM_PATH:/opt/LM \
--env-file .env \
linto-stt-kaldi:latest
```

If you have a recasepunc model do recover punctuation marks, you can add the following option:
```bash
-v <</path/to/recasepunc/model/folder>>:/opt/PUNCT
--env PUNCTUATION_MODEL=/opt/PUNCT
```

This will run a container providing an [HTTP API](#http-api) binded on the host HOST_SERVING_PORT port.

**Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| HOST_SERVING_PORT | Host serving port | 80 |
| AM_PATH | Path to the acoustic model on the host machine mounted to /opt/AM | /my/path/to/models/AM_fr-FR_v2.2.0 |
| LM_PATH | Path to the language model on the host machine mounted to /opt/LM | /my/path/to/models/fr-FR_big-v2.2.0 |
| MODEL_PATH | Path to the model (using MODEL_TYPE=vosk) mounted to /opt/models | /my/path/to/models/vosk-model |

### Celery task
The TASK serving mode connect a celery worker to a message broker.

The SERVICE_MODE value in the .env should be set to ```task```.

You need a message broker up and running at MY_SERVICE_BROKER.

```bash
docker run --rm \
-v AM_PATH:/opt/AM \
-v LM_PATH:/opt/LM \
-v SHARED_AUDIO_FOLDER:/opt/audio \
--env-file .env \
linto-stt-kaldi:latest
```

**Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| AM_PATH | Path to the acoustic model on the host machine mounted to /opt/AM | /my/path/to/models/AM_fr-FR_v2.2.0 |
| LM_PATH | Path to the language model on the host machine mounted to /opt/LM | /my/path/to/models/fr-FR_big-v2.2.0 |
| MODEL_PATH | Path to the model (using MODEL_TYPE=vosk) mounted to /opt/models | /my/path/to/models/vosk-model |
| SHARED_AUDIO_FOLDER | Shared audio folder mounted to /opt/audio | /my/path/to/models/vosk-model |


### Websocket Server (streaming)
Websocket server's mode deploy a streaming transcription service only.

The SERVICE_MODE value in the .env should be set to ```websocket```.

The exchanges are structured as followed:
1. Client send a json {"config": {"sample_rate":16000, "language":"en"}}. Language is optional, if not specified it will use the language from the env.
2. Client send audio chunk (go to 3- ) or {"eof" : 1} (go to 5-).
3. Server send either a partial result {"partial" : "this is a "} or a final result {"text": "this is a transcription"}.
4. Back to 2-
5. Server send a final result and close the connexion.

If you want text with upper case letters and punctuation, you can specify a recasepunc model (which must be in version 0.4 at least).
Some recasepunc models trained on [Common Crawl](http://data.statmt.org/cc-100/) are available on [recasepunc](https://github.com/benob/recasepunc/releases/) for the following the languages:
* French
  * [fr.24000](https://github.com/benob/recasepunc/releases/download/0.4/fr.24000)
* English
  * [en.22000](https://github.com/benob/recasepunc/releases/download/0.4/en.22000)
* Italian
  * [it.23000](https://github.com/benob/recasepunc/releases/download/0.4/it.23000)
* Chinese
  * [zh-Hant.17000](https://github.com/benob/recasepunc/releases/download/0.4/zh-Hant.17000)

After downloading a recasepunc model, you can mount it as a volume and specify its location within the Docker container using the `PUNCTUATION_MODEL` environment variable.

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
* File: An Wave file 16b 16Khz

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

#### /streaming
The /streaming route is accessible if the ENABLE_STREAMING environment variable is set to true.

The route accepts websocket connexions. Exchanges are structured as followed:
1. Client send a json {"config": {"sample_rate":16000}}.
2. Client send audio chunk (go to 3- ) or {"eof" : 1} (go to 5-).
3. Server send either a partial result {"partial" : "this is a "} or a final result {"text": "this is a transcription"}.
4. Back to 2-
5. Server send a final result and close the connexion.

> Connexion will be closed and the worker will be freed if no chunk are received for 10s. 

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


## Tests

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
