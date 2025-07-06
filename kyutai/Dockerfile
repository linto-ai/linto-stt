# ────────────────────────────── 1. BUILDER ──────────────────────────────
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder
# Version Cuda 12.4 --> see
# thread 'main' panicked at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cudarc-0.16.2/build.rs:104:14:
# 16.47   Unsupported cuda toolkit version: `12.9`

# 86 = Ampere, 89 = Ada/Lovelace, 90 = Hopper (RTX 4090 mobile 89)
ARG CUDARC_COMPUTE=89

# Rust stable via rustup
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git pkg-config clang cmake ca-certificates \
        python3 python3-minimal python3-dev libssl-dev \
    && curl -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH=/root/.cargo/bin:$PATH \
    CARGO_TERM_COLOR=always \
    CUDARC_COMPUTE=${CUDARC_COMPUTE} \
#  évite l’appel à nvidia-smi
    CUDA_COMPUTE_CAP=${CUDARC_COMPUTE}

# Compile la toute dernière moshi-server (+cuda) en statique
RUN cargo install --root /usr/local \
                  --features cuda --locked moshi-server



# ────────────────────────────── 2. RUNTIME ─────────────────────────────
FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl3 ca-certificates libexpat1 && \
    rm -rf /var/lib/apt/lists/*
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/data/hf_cache \
    RUST_LOG=info \
    MOSHI_PORT=8080
COPY --from=builder /usr/local/bin/moshi-server /usr/local/bin/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0 /usr/lib/x86_64-linux-gnu/
COPY delayed-streams-modeling/configs/config-stt-en_fr-hf.toml /etc/moshi/config.toml
VOLUME /data/hf_cache
VOLUME /root/tmp/tts-logs
EXPOSE 8080
ENTRYPOINT ["moshi-server","worker","--config","/etc/moshi/config.toml"]



# ────────────────────────────── 3. BUILDER CPU ──────────────────────────
FROM rust:1.82-slim AS builder-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git pkg-config clang cmake ca-certificates \
        python3 python3-dev libssl-dev build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN cargo install --root /usr/local \
                  --no-default-features --locked moshi-server



# ────────────────────────────── 4. RUNTIME CPU ─────────────────────────
FROM debian:bookworm-slim AS runtime-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl3 ca-certificates libexpat1 libpython3.11-stdlib && \
    rm -rf /var/lib/apt/lists/*
ENV HF_HOME=/data/hf_cache \
    RUST_LOG=info \
    MOSHI_PORT=8080
COPY --from=builder-cpu /usr/local/bin/moshi-server /usr/local/bin/
COPY --from=builder-cpu /usr/lib/x86_64-linux-gnu/libpython3.*so* /usr/lib/x86_64-linux-gnu/
COPY delayed-streams-modeling/configs/config-stt-en_fr-hf.toml /etc/moshi/config.toml
VOLUME /data/hf_cache
VOLUME /root/tmp/tts-logs
EXPOSE 8080
ENTRYPOINT ["moshi-server","worker","--config","/etc/moshi/config.toml"]