# Docker (cuda)

```bash
docker build -t moshi-stt:cuda --target runtime .
docker build -t moshi-stt:cpu --target runtime-cpu .
```

## Run
docker run --rm --gpus all -p 8080:8080 \                                                              
           -e RUST_LOG=info              \
           -it moshi-stt:cuda

# Local Rust

```bash
sudo apt update
sudo apt install -y build-essential pkg-config clang cmake libssl-dev git curl wget    
sudo apt install -y mold
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup default stable
```

## Run 

After `git submodule update --init --recursive` inside kyutai/delayed-streams-modeling
```bash
moshi-server worker --config configs/config-stt-en_fr-hf.toml
```

investigate - official doc : moshi-backend --features cuda --config $(moshi-backend default-config) standalone


# Client

See local scripts in delayed-streams-modeling. They run with uv.

```
curl -LsSf https://astral.sh/uv/install.sh | sh
# Clone kyutai stuff as submodules
git submodule update --init --recursive
```
inside delayed-streams-modeling folder :
```bash
uv run scripts/stt_from_mic_rust_server.py # Don't run on WSL. No ALSA for pyaudio
uv run scripts/stt_from_file_rust_server.py audio/bria.mp3
```

# Moshi WebClient - Moshi protocol (not working)

Shall help you consume your running moshi server... but fails

Inside moshi/client folder
```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \                                                                                                                                                       
            -days 365 -nodes -subj "/CN=localhost"
npm i
npm run dev
```
shall work with https://localhost:5173/?worker_addr=localhost:8080/api/asr-streaming but currently serving on ws not wss. Mixed content + Mic permissions with non-ssl address don't work.

# Quick and dirty WebClient - LinTO Protocol (working)
```
python3 -m http.server --directory webclient 8000
```

Open your browser (tested both implementations with Chrome) ; use the links below while setting GET parameter to the address of the LinTO ASR Websocket server you want to connect to.

http://localhost:8000/audioprocessor.html?server=ws://localhost:8001/streaming

or

http://localhost:8000/worklet.html?server=ws://localhost:8001/streaming

Replace the `?server=ws://localhost:8001/streaming` with the actual address you want to test. (the /streaming part remains the same)

# LinTO wrapper - Purpose of this Kyutai branch

The `kyutai/stt/processing` package shall provide a lightweight wrapper that exposes the
standard LinTO streaming API and forwards the audio stream to a running (dockerized ?) Kyutai
server.

## Quick start

Install `uv` if not already available and install the wrapper requirements:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.9 .venv && source .venv/bin/activate  
uv pip install -r kyutai/requirements.wrapper.txt  
```

Then run the websocket server from the repository root. By default it reaches the moshi server on KYUTAI_URL=`ws://localhost:8080` and listens on default `8001`

```bash
LOG_TRANSCRIPTS=true LOG_LEVEL=INFO FINAL_TRANSCRIPT_DELAY=1.5 STREAMING_PORT=8002 PYTHONPATH=kyutai KYUTAI_URL=ws://localhost:8080 uv run python websocket/websocketserver.py
```

__note__ :  LOG_TRANSCRIPTS to log "finals" (simulated after a semantic utterance, ending with a '.?!...' and followed by a FINAL_TRANSCRIPT_DELAY number of seconds without any new transcription.)

## LinTO wrapper - Docker image

```bash
docker build -f kyutai/Dockerfile.wrapper -t linto-stt-kyutai-wrapper .
docker run --rm -p 8001:8001 \
  -e SERVICE_MODE=websocket \
  -e KYUTAI_URL=ws://host.docker.internal:8080 \
  linto-stt-kyutai-wrapper
```

The container exposes the same `/streaming` endpoint as other LinTO backends and
forwards requests to a Kyutai server running locally or on the network. See [protocol details](PROTOCOL.md)
