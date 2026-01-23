# LinTO-STT

```sh
apt install python3-pyaudio portaudio19-dev
```

```sh
uv sync --extra [nemo]
```

## Run

### HTTP / Websocket

`uv run main.py -m [http|websocket] -b [kaldi|whisper|nemo|kyutai] -p [listening_port] -i [listening_ip]`

> kyutai only support streaming

<!-- // todo: describe protocol -->

### Celery

`uv run main.py -m task -b [kaldi|whisper|nemo]`

## Kaldi configuration

With a kaldi model

```
todo
```

With a vosk model. [Available model on vosk website](https://alphacephei.com/vosk/models)

```
MODEL_PATH=<path/to/vosk/model>
MODEL_TYPE=vosk
```

[See more](./linto_stt/backends/kaldi/README.md)

## Whisper configuration

[See more](./linto_stt/backends/whisper/README.md)

## Nemo configuration

Basic use with the french LINAGORA model

```
# .env file
ARCHITECTURE=hybrid_bpe_rnnt
MODEL=linagora/linto_stt_fr_fastconformer
```

[See more](./linto_stt/backends/nemo/README.md)

## Kyutai configuration

Kyutai need a moshi server running see https://github.com/kyutai-labs/delayed-streams-modeling

```
# .env file
KYUTAI_URL=ws://localhost:9002
```
