# LinTO-STT

```sh
apt install python3-pyaudio portaudio19-dev
```

```sh
uv sync --extra [nemo]
```

## Run

### HTTP / Websocket

`uv run main.py -m [http|websocket] -b [nemo|kyutai] -p [listening_port] -i [listening_ip]`

> kyutai only support streaming

<!-- // todo: describe protocol -->

### Celery

`uv run main.py -m task -b nemo`

## Nemo configuration

To use the french LINAGORA model

```
# .env file
ARCHITECTURE=hybrid_bpe_rnnt
MODEL=linagora/linto_stt_fr_fastconformer
```

## Kyutai configuration

Kyutai need a moshi server running see https://github.com/kyutai-labs/delayed-streams-modeling

```
# .env file
KYUTAI_URL=ws://localhost:9002
```
