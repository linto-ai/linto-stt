################################################################################
# Builder stage — compiles native extensions (PyAudio), installs all Python deps
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
  && if [ "$STT_ENGINE" = "nemo" ] || [ "$STT_ENGINE" = "whisper" ] || [ "$STT_ENGINE" = "whisper-torch" ]; then \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    libsndfile1; \
  fi \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Install Python dependencies (cached as long as pyproject.toml/uv.lock don't change)
COPY pyproject.toml uv.lock ./
RUN uv export --extra "$STT_ENGINE" --no-emit-project --no-hashes --frozen > requirements.txt && \
    uv pip install --system --no-cache -r requirements.txt

# Optional: install recasepunc (torch CPU + transformers)
RUN if [ "$EXTRA_DEPS" = "recasepunc" ]; then \
        uv pip install --system --no-cache \
            "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu && \
        uv pip install --system --no-cache "transformers>=4.30.0"; \
    fi

# Optional: install CUDA runtime libraries for GPU (cuBLAS + cuDNN for ctranslate2)
RUN if [ -n "$GPU" ]; then \
        uv pip install --system --no-cache \
            "nvidia-cublas-cu12>=12.4,<13" \
            "nvidia-cudnn-cu12>=9,<10"; \
    fi

# Copy source and install the project itself (no deps, fast)
COPY linto_stt ./linto_stt
COPY tests/bonjour.wav ./tests/bonjour.wav
RUN uv pip install --system --no-cache --no-deps .

################################################################################
# Runtime stage — minimal image with only runtime dependencies
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
  && if [ "$STT_ENGINE" = "nemo" ] || [ "$STT_ENGINE" = "whisper" ] || [ "$STT_ENGINE" = "whisper-torch" ]; then \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libportaudio2 \
    libsndfile1; \
  fi \
  && if [ "$STT_ENGINE" = "nemo" ]; then \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg; \
  fi \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

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
