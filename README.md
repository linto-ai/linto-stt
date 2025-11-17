# LinTO-STT

```sh
apt install python3-pyaudio portaudio19-dev
```

```sh
uv sync --all-extras
uv run main.py -m [http|websocket] -b nemo -p [listening_port] -i [listening_ip]
```

## Nemo configuration

To use the french LINAGORA model

```
# .env file
ARCHITECTURE=hybrid_bpe_rnnt
MODEL=linagora/linto_stt_fr_fastconformer
```
