################################################################################
# Builder stage: compiles native extensions (PyAudio), installs all Python deps
################################################################################
FROM python:3.12-slim AS builder

ARG STT_ENGINE=nemo
ARG EXTRA_DEPS=""
ARG GPU=""

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Build-time system dependencies
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git \
  && if [ "$STT_ENGINE" = "nemo" ] || [ "$STT_ENGINE" = "whisper" ] || [ "$STT_ENGINE" = "whisper-torch" ] || [ "$STT_ENGINE" = "hf" ]; then \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    libsndfile1; \
  fi \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Install Python dependencies (cached as long as pyproject.toml/uv.lock don't change)
COPY pyproject.toml uv.lock ./
# GPU builds pull multi-hundred-megabyte CUDA wheels; uv's default 30 s request
# timeout makes such builds fail on a slow or busy connection.
ENV UV_HTTP_TIMEOUT=600
# The hf engine's default backend is vLLM, which needs a GPU. A GPU
# build (GPU set) installs the `hf-vllm` extra (a superset that also covers the
# in-process `transformers` backend); a CPU build installs the lean `hf` extra
# (no vLLM), so such an image must run with BACKEND=transformers. Other
# engines install their own extra.
RUN if [ "$STT_ENGINE" = "hf" ]; then \
        if [ -n "$GPU" ]; then EXTRA="hf-vllm"; else EXTRA="hf"; fi; \
    else EXTRA="$STT_ENGINE"; fi && \
    uv export --extra "$EXTRA" --no-emit-project --no-hashes --frozen > requirements.txt && \
    uv pip install --system --no-cache -r requirements.txt

# Optional: install recasepunc (torch CPU + transformers)
RUN if [ "$EXTRA_DEPS" = "recasepunc" ]; then \
        uv pip install --system --no-cache \
            "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu && \
        uv pip install --system --no-cache "transformers>=4.30.0"; \
    fi

# The hf GPU image: vLLM pulls optional packages that ASR never uses (DeepSeek
# MLA attention kernels, TileLang, an nvcc toolchain, the video codec); drop
# them to save ~1 GB. CUTLASS, cuDNN frontend and OpenCV are kept (FlashInfer
# paths on recent GPUs, Mistral/Voxtral tokenizers).
RUN if [ "$STT_ENGINE" = "hf" ] && [ -n "$GPU" ]; then \
        uv pip uninstall --system \
            tokenspeed-mla tokenspeed-triton tilelang quack-kernels humming-kernels \
            nvidia-cuda-nvcc nvidia-cuda-nvdisasm nvidia-cuda-crt nvidia-nvvm nvidia-cuda-cccl \
            pynvvideocodec; \
    fi

# Optional: CUDA 12 runtime libraries for ctranslate2 (faster-whisper) on GPU.
# Only the whisper engine needs them: torch-based engines ship their own CUDA
# libraries, and nvidia-cudnn-cu12 would even overwrite torch's cuDNN (both
# unpack into the same directory).
RUN if [ -n "$GPU" ] && [ "$STT_ENGINE" = "whisper" ]; then \
        uv pip install --system --no-cache \
            "nvidia-cublas-cu12>=12.4,<13" \
            "nvidia-cudnn-cu12>=9,<10"; \
    fi

################################################################################
# Runtime stage: minimal image with only runtime dependencies
################################################################################
FROM python:3.12-slim

LABEL maintainer="contact@linto.ai"

ARG STT_ENGINE=nemo
ARG GPU=""
ENV STT_ENGINE=${STT_ENGINE}

# Runtime-only system dependencies
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  curl \
  gosu \
  netcat-traditional \
  passwd \
  && if [ "$STT_ENGINE" = "nemo" ] || [ "$STT_ENGINE" = "whisper" ] || [ "$STT_ENGINE" = "whisper-torch" ] || [ "$STT_ENGINE" = "hf" ]; then \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libportaudio2 \
    libsndfile1; \
  fi \
  && if [ "$STT_ENGINE" = "nemo" ] || [ "$STT_ENGINE" = "hf" ]; then \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg; \
  fi \
  && if [ "$STT_ENGINE" = "hf" ] && [ -n "$GPU" ]; then \
    # Triton compiles small host-side launcher stubs at runtime and needs a C
    # compiler with the C library headers (vLLM fails to start without them,
    # even in eager mode); C++ is not needed, including for torch.compile and
    # CUDA graphs.
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    gcc libc6-dev; \
  fi \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Hardening: no package installers at runtime (uv only served the build; pip is
# kept for the whisper engine, which can pip-install a converter on demand), and
# no setuid/setgid binaries (su, mount, passwd...): the entrypoint starts as root
# and drops privileges with gosu, which does not rely on a setuid bit.
RUN rm -f /usr/local/bin/uv /usr/local/bin/uvx && \
    if [ "$STT_ENGINE" != "whisper" ] && [ "$STT_ENGINE" != "whisper-torch" ]; then \
        python -m pip uninstall -y -q pip && \
        rm -f /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.12; \
    fi && \
    find / -xdev -type f -perm /6000 -exec chmod ug-s {} +

# Configure CUDA library paths if GPU is enabled
RUN if [ -n "$GPU" ]; then \
        find /usr/local/lib -path '*/nvidia/*/lib' -type d \
            > /etc/ld.so.conf.d/nvidia.conf && \
        ldconfig; \
    fi

WORKDIR /usr/src/app

# Copy application files
COPY linto_stt ./linto_stt
COPY tests/bonjour.wav ./tests/bonjour.wav
COPY docker-entrypoint.sh wait-for-it.sh ./
RUN chmod +x docker-entrypoint.sh wait-for-it.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
