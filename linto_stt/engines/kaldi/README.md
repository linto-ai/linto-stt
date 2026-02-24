# LinTO-STT-Kaldi

ASR API based on [Kaldi](https://github.com/kaldi-asr/kaldi) and [Vosk](https://alphacephei.com/vosk/). Supports offline and real-time transcription.

> See the [main README](https://github.com/linto-ai/linto-stt/blob/master/README.md) for API docs, Docker options, and serving modes.
> See [ENV.md](https://github.com/linto-ai/linto-stt/blob/master/ENV.md) for all environment variables.

## Quick Start

### Prerequisites

- [Docker](https://www.Docker.com/products/Docker-desktop/)
- At least 7GB disk space for the image
- Up to 7GB RAM depending on model

### Models

LinTO-STT-Kaldi accepts two kinds of models:

**LinTO models** (acoustic + language model separately):

- Download from [dl.linto.ai](https://doc.linto.ai/docs/developpers/apis/ASR/models)
- Set `MODEL_TYPE=lin`
- Mount volumes: `-v <AM_PATH>:/opt/AM -v <LM_PATH>:/opt/LM`

**Vosk models** (all-in-one):

- Download from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)
- Set `MODEL_TYPE=vosk`
- Mount volume: `-v <MODEL_PATH>:/opt/model`

### Pull or Build

```sh
docker pull lintoai/linto-stt-kaldi
```

or

```sh
docker build -t linto-stt-kaldi:latest --build-arg STT_ENGINE=kaldi .
```

### Run

With LinTO models:

```sh
docker run --rm -p 8080:80 \
  -e SERVICE_MODE=http \
  -e MODEL_TYPE=lin \
  -v /path/to/AM:/opt/AM \
  -v /path/to/LM:/opt/LM \
  lintoai/linto-stt-kaldi
```

With a Vosk model:

```sh
docker run --rm -p 8080:80 \
  -e SERVICE_MODE=http \
  -e MODEL_TYPE=vosk \
  -v /path/to/vosk-model:/opt/model \
  lintoai/linto-stt-kaldi
```

For streaming, use `-e SERVICE_MODE=websocket`. For Celery task mode, use `-e SERVICE_MODE=task` with `-v /shared/audio:/opt/audio`.

For punctuation recovery, add a [recasepunc model](../../../README.md#punctuation-model-recasepunc):

```sh
-v /path/to/fr.24000:/opt/PUNCT -e PUNCTUATION_MODEL=/opt/PUNCT
```

## License

AGPLv3 (see LICENSE).

## Acknowledgments

- [Vosk, speech recognition toolkit](https://alphacephei.com/vosk/)
- [Kaldi Speech Recognition Toolkit](https://github.com/kaldi-asr/kaldi)
