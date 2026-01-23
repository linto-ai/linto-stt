# LinTO wrapper - Purpose of this Kyutai branch

The `kyutai/stt/processing` package shall provide a lightweight wrapper that exposes the
standard LinTO streaming API and forwards the audio stream to a running (dockerized ?) Kyutai moshi-server worker
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

### Semantic VAD Configuration

The wrapper now supports **Semantic Voice Activity Detection (VAD)** from the Moshi 1B model's extra heads. This provides intelligent utterance boundary detection beyond simple timer-based heuristics.

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_SEMANTIC_VAD` | `true` | Enable/disable semantic VAD processing |
| `VAD_THRESHOLD` | `0.5` | Probability threshold for end-of-utterance detection (0.0-1.0) |
| `VAD_HISTORY_SIZE` | `3` | Number of consecutive high VAD signals needed for confirmation |
| `VAD_DELAY` | `0.3` | Delay (seconds) after VAD trigger before sending final transcript |
| `VAD_REQUIRE_PUNCTUATION` | `true` | Require both VAD signal AND punctuation for finalization |
| `LOG_VAD` | `false` | Log VAD probabilities for debugging |
| `LOG_TRANSCRIPTS` | `false` | Log final transcripts with finalization reason |
| `FINAL_TRANSCRIPT_DELAY` | `1.5` | Fallback timer delay (seconds) when VAD is not triggered |

**Examples:**

```bash
# Default: Use VAD with punctuation confirmation
PYTHONPATH=kyutai uv run python websocket/websocketserver.py

# More sensitive VAD (lower threshold)
USE_SEMANTIC_VAD=true VAD_THRESHOLD=0.3 PYTHONPATH=kyutai uv run python websocket/websocketserver.py

# VAD without requiring punctuation (more aggressive)
VAD_REQUIRE_PUNCTUATION=false PYTHONPATH=kyutai uv run python websocket/websocketserver.py

# Debug VAD behavior
LOG_VAD=true LOG_TRANSCRIPTS=true LOG_LEVEL=DEBUG PYTHONPATH=kyutai uv run python websocket/websocketserver.py

# Disable VAD (fallback to original timer-only behavior)
USE_SEMANTIC_VAD=false PYTHONPATH=kyutai uv run python websocket/websocketserver.py
```

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

# Moshi-server worker - Docker image

> **Note:** Depending on your NVIDIA GPU architecture, you may need to specify the `CUDARC_COMPUTE` build argument. For example, for a Turing architecture (e.g., RTX 2080), you would build with:
> ```bash
> docker build --build-arg CUDARC_COMPUTE=75 -t moshi-stt:cuda --target runtime .
> ```
> You can find the compute capability of your GPU on the [NVIDIA CUDA GPUs - Compute Capability](https://developer.nvidia.com/cuda-gpus) page. The default is `89` (Ada Lovelace).

```bash
docker build -t moshi-stt:cuda --target runtime .
docker build -t moshi-stt:cpu --target runtime-cpu .
```

```bash
docker run --rm --gpus all -p 8080:8080 \                                                              
           -e RUST_LOG=info              \
           -it moshi-stt:cuda
```
# Docker Compose

A `docker-compose.yml` file is provided to easily run the entire stack locally. This includes the Moshi server, the LinTO wrapper, and a web client for testing.

## Local Usage

The setup uses Docker Compose profiles to select between the CPU and CUDA environments. Navigate to the `kyutai` directory and use one of the following commands:

To build and run the **CPU** version:
```bash
docker-compose --profile cpu up --build
```

To build and run the **CUDA** version, first set the `MOSHI_SERVER` environment variable, then run the compose command. You can also set the `CUDARC_COMPUTE` build argument if you need to target a specific CUDA architecture (default is 89 - Ada Lovelace):
```bash
export MOSHI_SERVER=moshi-server-cuda
CUDARC_COMPUTE=75 docker compose --profile cuda up --build
```

As the Kyutai's model is downloaded, wait a bit until you see logs like 

```bash
moshi-stt-server-cuda  | 2025-07-08T22:28:57.767774Z  INFO moshi_server: /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/moshi-server-0.6.3/src/main.rs:529: listening on http://0.0.0.0:8080
moshi-stt-server-cuda  | 2025-07-08T22:28:57.767895Z  INFO moshi_server::batched_asr: /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/moshi-server-0.6.3/src/batched_asr.rs:226: warming-up the asr
moshi-stt-server-cuda  | 2025-07-08T22:28:58.231816Z  INFO moshi_server::batched_asr: /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/moshi-server-0.6.3/src/batched_asr.rs:228: starting asr loop 64
```

In either case, the web client will be available at `http://localhost:8088/?server=ws://localhost:8001/streaming`. Allow microphone usage and transcribe.

# Dockerless : Rust install

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


## Test

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


# Quick and dirty WebClient - Using LinTO Client / Protocol
```
python3 -m http.server --directory webclient 8000
```

Open your browser (tested both implementations with Chrome) ; use the links below while setting GET parameter to the address of the LinTO ASR Websocket server you want to connect to.

http://localhost:8000/audioprocessor.html?server=ws://localhost:8001/streaming

or

http://localhost:8000/worklet.html?server=ws://localhost:8001/streaming

Replace the `?server=ws://localhost:8001/streaming` with the actual address you want to test. (the /streaming part remains the same)