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

moshi-backend --features cuda --config $(moshi-backend default-config) standalone


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

# WebClient (not working)

inside moshi folder
```bash
npm i
npm run dev
```
shall work with https://localhost:5173/?worker_addr=localhost:8080 but currently serving on ws not wss. Mixed content + Mic permissions with non-ssl address don't work.
