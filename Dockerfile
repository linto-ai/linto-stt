FROM python:3.12-trixie
LABEL maintainer="contact@linto.ai"

ARG SERVICE_NAME=nemo

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Common system dependencies
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ffmpeg \
  git \
  curl \
  gosu \
  netcat-traditional \
  libsndfile1 sox \
  libfreetype6 \
  swig \
  portaudio19-dev \
  libavutil-dev \
  libavdevice-dev && \
  rm -rf /var/lib/apt/lists/*;

WORKDIR /usr/src/app

# Install dependencies first (cached as long as pyproject.toml/uv.lock don't change)
COPY pyproject.toml uv.lock /usr/src/app/
RUN uv export --extra "$SERVICE_NAME" --no-emit-project --frozen > requirements.txt && \
    uv pip install --system --no-cache -r requirements.txt

# Copy source and test data
COPY linto_stt /usr/src/app/linto_stt
COPY test/bonjour.wav /usr/src/app/test/bonjour.wav

# Install the project itself (no deps, fast)
RUN uv pip install --system --no-cache --no-deps .

COPY docker-entrypoint.sh /usr/src/app/docker-entrypoint.sh
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
