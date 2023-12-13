# LinTO-STT-Whisper

LinTO-STT-Whisper is the transcription service within the [LinTO stack](https://github.com/linto-ai/linto-platform-stack)
based on Speech-To-Text (STT) [Whisper models](https://openai.com/research/whisper).

LinTO-STT-Whisper can either be used as a standalone transcription service or deployed within a micro-services infrastructure using a message broker connector.

## Pre-requisites

### Hardware
To run the transcription models you'll need:
* At least 8Go of disk space to build the docker image.
* Up to 7GB of RAM depending on the model used.
* One CPU per worker. Inference time scales on CPU performances. 

### Model(s)

LinTO-STT-Whisper works with a Whisper model to perform Automatic Speech Recognition.
If not downloaded already, the model will be downloaded when calling the first transcription,
and can occupy several GB of disk space.

#### Optional alignment model (deprecated)

LinTO-STT-Whisper has also the option to work with a wav2vec model to perform word alignment.
The wav2vec model can be specified either
* (TorchAudio) with a string corresponding to a `torchaudio` pipeline (e.g. "WAV2VEC2_ASR_BASE_960H") or
* (HuggingFace's Transformers) with a string corresponding to a HuggingFace repository of a wav2vec model (e.g. "jonatasgrosman/wav2vec2-large-xlsr-53-english"), or
* (SpeechBrain) with a path corresponding to a folder with a SpeechBrain model

Default wav2vec models are provided for French (fr), English (en), Spanish (es), German (de), Dutch (nl), Japanese (ja), Chinese (zh).

But we advise not to use a companion wav2vec alignment model.
This is not needed neither tested anymore.

### Docker
The transcription service requires docker up and running.

### (micro-service) Service broker and shared folder
The STT only entry point in task mode are tasks posted on a message broker. Supported message broker are RabbitMQ, Redis, Amazon SQS.
On addition, as to prevent large audio from transiting through the message broker, STT-Worker use a shared storage folder (SHARED_FOLDER).

## Deploy LinTO-STT-Whisper

### 1- First step is to build or pull the image

```bash
git clone https://github.com/linto-ai/linto-stt.git
cd linto-stt
docker build . -f whisper/Dockerfile.ctranslate2 -t linto-stt-whisper:latest
```
or

```bash
docker pull lintoai/linto-stt-whisper
```

### 2- Fill the .env

```bash
cp whisper/.envdefault whisper/.env
```

| PARAMETER | DESCRIPTION | EXEMPLE |
|---|---|---|
| SERVICE_MODE | STT serving mode see [Serving mode](#serving-mode) | `http` \| `task` |
| MODEL | Path to a Whisper model, type of Whisper model used, or HuggingFace identifier of a Whisper model. | \<ASR_PATH\> \| `large-v3` \| `distil-whisper/distil-large-v2` \| ... |
| LANGUAGE | (Optional) Language to recognize | `*` \| `fr` \| `fr-FR` \| `French` \| `en` \| `en-US` \| `English` \| ... |
| PROMPT | (Optional) Prompt to use for the Whisper model | `some free text to encourage a certain transcription style (disfluencies, no punctuation, ...)` |
| ALIGNMENT_MODEL | (Optional) Path to the wav2vec model for word alignment, or name of HuggingFace repository or torchaudio pipeline | \<WAV2VEC_PATH\> \| `WAV2VEC2_ASR_BASE_960H` \| `jonatasgrosman/wav2vec2-large-xlsr-53-english` \| ... |
| CONCURRENCY | Maximum number of parallel requests | `3` |
| SERVICE_NAME | (For the task mode) queue's name for task processing | `my-stt` |
| SERVICE_BROKER | (For the task mode) URL of the message broker | `redis://my-broker:6379` |
| BROKER_PASS | (For the task mode only) broker password | `my-password` |

#### MODEL environment variable

**Warning:**
The model will be (downloaded if required and) loaded in memory when calling the first transcription.
When using a Whisper model from Hugging Face (transformers) along with ctranslate2 (faster_whisper),
it will also download torch library to make the conversion from torch to ctranslate2.

If you want to preload the model (and later specify a path `ASR_PATH` as `MODEL`),
you may want to download one of OpenAI Whisper models:
* Mutli-lingual Whisper models can be downloaded with the following links:
    * [tiny](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)
    * [base](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt)
    * [small](https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt)
    * [medium](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt)
    * [large-v1](https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt)
    * [large-v2](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt)
    * [large-v3](https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt)
* Whisper models specialized for English can also be found here:
    * [tiny.en](https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt)
    * [base.en](https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt)
    * [small.en](https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt)
    * [medium.en](https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt)

If you already used Whisper in the past locally using [OpenAI-Whipser](https://github.com/openai/whisper), models can be found under ~/.cache/whisper.

The same apply for Whisper models from Hugging Face (transformers), as for instance https://huggingface.co/distil-whisper/distil-large-v2
(you can either download the model or use the Hugging Face identifier `distil-whisper/distil-large-v2`).

#### LANGUAGE

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
and also `yue(cantonese)` since large-v3.

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
--env-file whisper/.env \
linto-stt-whisper:latest
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

You need a message broker up and running at MY_SERVICE_BROKER.

```bash
docker run --rm \
-v ASR_PATH:/opt/model.pt \
-v SHARED_AUDIO_FOLDER:/opt/audio \
--env-file whisper/.env \
linto-stt-whisper:latest
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

* [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)
* [OpenAI Whisper](https://github.com/openai/whisper)
* [Ctranslate2](https://github.com/OpenNMT/CTranslate2)
* [SpeechBrain](https://github.com/speechbrain/speechbrain)
* [TorchAudio](https://github.com/pytorch/audio)
* [HuggingFace Transformers](https://github.com/huggingface/transformers)