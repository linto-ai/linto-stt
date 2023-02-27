# LINTO-PLATFORM-STT
LinTO-platform-stt is the transcription service within the [LinTO stack](https://github.com/linto-ai/linto-platform-stack).

LinTO-platform-stt can either be used as a standalone transcription service or deployed within a micro-services infrastructure using a message broker connector.

## Pre-requisites

### Hardware
To run the transcription models you'll need:
* At least 8Go of disk space to build the docker image.
* Up to 7GB of RAM depending on the model used.
* One CPU per worker. Inference time scales on CPU performances. 

### Model
LinTO-Platform-STT works with two models:
* A Whisper model to perform Automatic Speech Recognition, which must be in the PyTorch format.
* A wav2vec model to perform word alignment, which can be in the format of SpeechBrain, HuggingFace's Transformers or TorchAudio

The wav2vec model can be specified either
* with a string corresponding to a `torchaudio` pipeline (e.g. "WAV2VEC2_ASR_BASE_960H") or
* with a string corresponding to a HuggingFace repository of a wav2vec model (e.g. "jonatasgrosman/wav2vec2-large-xlsr-53-english"), or
* with a path corresponding to a folder with a SpeechBrain model

Default models are provided for the following languages:
* French (fr)
* English (en)
* Spanish (es)
* German (de)
* Dutch (nl)
* Japanese (ja)
* Chinese (zh)

### Docker
The transcription service requires docker up and running.

### (micro-service) Service broker and shared folder
The STT only entry point in task mode are tasks posted on a message broker. Supported message broker are RabbitMQ, Redis, Amazon SQS.
On addition, as to prevent large audio from transiting through the message broker, STT-Worker use a shared storage folder (SHARED_FOLDER).

## Deploy linto-platform-stt

**1- First step is to build or pull the image:**

```bash
git clone https://github.com/linto-ai/linto-platform-stt.git
cd linto-platform-stt
docker build . -t linto-platform-stt:latest
```
or

```bash
docker pull lintoai/linto-platform-stt
```

**2- Download the models**

Have the Whisper model file ready at ASR_PATH.

If you already used Whisper in the past, you may have models in ~/.cache/whisper.

You can download mutli-lingual Whisper models with the following links:
* tiny: "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt
* base: https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt
* small: https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt
* medium: https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt
* large-v1: https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt
* large-v2: https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt

Whisper models specialized for English can also be found here:
* tiny.en: "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt
* base.en: https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt
* small.en: https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt
* medium.en: https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt

If may also want to download a specific wav2vec model for word alignment.

**3- Fill the .env**

```bash
cp .envdefault .env
```

| PARAMETER | DESCRIPTION | EXEMPLE |
|---|---|---|
| SERVICE_MODE | STT serving mode see [Serving mode](#serving-mode) | `http` \| `task` |
| MODEL | Path to the Whisper model, or type of Whisper model used. | \<ASR_PATH\> \| `medium` \| `large-v1` \| ... |
| ALIGNMENT_MODEL | (Optional) Path to the wav2vec model for word alignment, or name of HuggingFace repository or torchaudio pipeline | \<WAV2VEC_PATH\> \| `WAV2VEC2_ASR_BASE_960H` \| `jonatasgrosman/wav2vec2-large-xlsr-53-english` \| ... |
| LANGUAGE | (Optional) Language to recognize | `*` \| `fr` \| `fr-FR` \| `French` \| `en` \| `en-US` \| `English` \| ... |
| SERVICE_NAME | Using the task mode, set the queue's name for task processing | `my-stt` |
| SERVICE_BROKER | Using the task mode, URL of the message broker | `redis://my-broker:6379` |
| BROKER_PASS | Using the task mode, broker password | `my-password` |
| CONCURRENCY | Maximum number of parallel requests | `3` |

If `*` is used for the `LANGUAGE` environment variable, or if `LANGUAGE` is not defined,
automatic language detection will be performed by Whisper.

The language can be a code of two or three letters. The list of languages supported by Whisper are:
```
af(afrikaans), am(amharic), ar(arabic), as(assamese), az(azerbaijani),
ba(bashkir), be(belarusian), bg(bulgarian), bn(bengali), bo(tibetan), br(breton), bs(bosnian),
ca(catalan), cs(czech), cy(welsh), da(danish), de(german), el(greek), en(english), es(spanish),
et(estonian), eu(basque), fa(persian), fi(finnish), fo(faroese), fr(french), gl(galician),
gu(gujarati), ha(hausa), haw(hawaiian), he(hebrew), hi(hindi), hr(croatian), ht(haitian creole),
hu(hungarian), hy(armenian), id(indonesian), is(icelandic), it(italian), ja(japanese),
jw(javanese), ka(georgian), kk(kazakh), km(khmer), kn(kannada), ko(korean), la(latin),
lb(luxembourgish), ln(lingala), lo(lao), lt(lithuanian), lv(latvian), mg(malagasy), mi(maori),
mk(macedonian), ml(malayalam), mn(mongolian), mr(marathi), ms(malay), mt(maltese), my(myanmar),
ne(nepali), nl(dutch), nn(nynorsk), no(norwegian), oc(occitan), pa(punjabi), pl(polish),
ps(pashto), pt(portuguese), ro(romanian), ru(russian), sa(sanskrit), sd(sindhi), si(sinhala),
sk(slovak), sl(slovenian), sn(shona), so(somali), sq(albanian), sr(serbian), su(sundanese),
sv(swedish), sw(swahili), ta(tamil), te(telugu), tg(tajik), th(thai), tk(turkmen), tl(tagalog),
tr(turkish), tt(tatar), uk(ukrainian), ur(urdu), uz(uzbek), vi(vietnamese), yi(yiddish),
yo(yoruba), zh(chinese)
```

### Serving mode 
![Serving Modes](https://i.ibb.co/qrtv3Z6/platform-stt.png)

STT can be used in two ways:
* Through an [HTTP API](#http-server) using the **http**'s mode.
* Through a [message broker](#micro-service-within-linto-platform-stack) using the **task**'s mode.

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
-v ASR_PATH:/opt/model.pt \
--env-file .env \
linto-platform-stt:latest
```

This will run a container providing an [HTTP API](#http-api) binded on the host HOST_SERVING_PORT port.

You may also want to mount your cache folder CACHE_PATH (e.g. "~/.cache") ```-v CACHE_PATH:/root/.cache```
in order to avoid downloading models each time.

Also if you want to specifiy a custom alignment model already downloaded in a folder WAV2VEC_PATH,
you can add option ```-v WAV2VEC_PATH:/opt/wav2vec``` and environment variable ```ALIGNMENT_MODEL=/opt/wav2vec```.

**Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| HOST_SERVING_PORT | Host serving port | 8080 |
| ASR_PATH | Path to the Whisper model on the host machine mounted to /opt/model.pt | /my/path/to/models/medium.pt |
| CACHE_PATH | (Optional) Path to a folder to download wav2vec alignment models when relevant | /home/username/.cache |
| WAV2VEC_PATH | (Optional) Path to a folder to a custom wav2vec alignment model |  /my/path/to/models/wav2vec |

### Micro-service within LinTO-Platform stack
The HTTP serving mode connect a celery worker to a message broker.

The SERVICE_MODE value in the .env should be set to ```task```.

>LinTO-platform-stt can be deployed within the linto-platform-stack through the use of linto-platform-services-manager. Used this way, the container spawn celery worker waiting for transcription task on a message broker.
>LinTO-platform-stt in task mode is not intended to be launch manually.
>However, if you intent to connect it to your custom message's broker here are the parameters:

You need a message broker up and running at MY_SERVICE_BROKER.

```bash
docker run --rm \
-v ASR_PATH:/opt/model.pt \
-v SHARED_AUDIO_FOLDER:/opt/audio \
--env-file .env \
linto-platform-stt:latest
```

You may also want to mount your cache folder CACHE_PATH (e.g. "~/.cache") ```-v CACHE_PATH:/root/.cache```
in order to avoid downloading models each time.

Also if you want to specifiy a custom alignment model already downloaded in a folder WAV2VEC_PATH,
you can add option ```-v WAV2VEC_PATH:/opt/wav2vec``` and environment variable ```ALIGNMENT_MODEL=/opt/wav2vec```.

**Parameters:**
| Variables | Description | Example |
|:-|:-|:-|
| SHARED_AUDIO_FOLDER | Shared audio folder mounted to /opt/audio | /my/path/to/models/vosk-model |
| ASR_PATH | Path to the Whisper model on the host machine mounted to /opt/model.pt | /my/path/to/models/medium.pt |
| CACHE_PATH | (Optional) Path to a folder to download wav2vec alignment models when relevant | /home/username/.cache |
| WAV2VEC_PATH | (Optional) Path to a folder to a custom wav2vec alignment model |  /my/path/to/models/wav2vec |


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


## Test
### Curl
You can test you http API using curl:
```bash 
curl -X POST "http://YOUR_SERVICE:YOUR_PORT/transcribe" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=@YOUR_FILE;type=audio/x-wav"
```

## License
This project is developped under the AGPLv3 License (see LICENSE).

## Acknowlegment.

* [OpenAI Whisper](https://github.com/openai/whisper)
* [SpeechBrain](https://github.com/speechbrain/speechbrain).
* [TorchAudio](https://github.com/pytorch/audio)
* [HuggingFace Transformers](https://github.com/huggingface/transformers)