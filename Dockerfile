FROM python:3.12-slim
LABEL maintainer="contact@linto.ai"

ARG STT_ENGINE=nemo
ENV STT_ENGINE=${STT_ENGINE}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Common system dependencies (all engines)
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git \
  curl \
  gosu \
  netcat-traditional \
  passwd \
  && if [ "$STT_ENGINE" = "nemo" ] || [ "$STT_ENGINE" = "whisper" ] || [ "$STT_ENGINE" = "whisper-torch" ]; then \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    libsndfile1; \
  fi \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Install dependencies first (cached as long as pyproject.toml/uv.lock don't change)
COPY pyproject.toml uv.lock /usr/src/app/
RUN uv export --extra "$STT_ENGINE" --no-emit-project --frozen > requirements.txt && \
    uv pip install --system --no-cache -r requirements.txt

# Copy source and test data
COPY linto_stt /usr/src/app/linto_stt
COPY tests/bonjour.wav /usr/src/app/tests/bonjour.wav

# Install the project itself (no deps, fast)
RUN uv pip install --system --no-cache --no-deps .

COPY docker-entrypoint.sh /usr/src/app/docker-entrypoint.sh
COPY wait-for-it.sh /usr/src/app/wait-for-it.sh
RUN chmod +x docker-entrypoint.sh wait-for-it.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
