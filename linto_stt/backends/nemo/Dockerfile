# FROM nvcr.io/nvidia/pytorch:24.02-py3
FROM python:3.12
LABEL maintainer="contact@linto.ai"

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
  libsndfile1 sox \
  libfreetype6 \
  swig \
  ffmpeg \
  gosu \
  netcat-traditional \
  libavdevice-dev \
  portaudio19-dev && \
  rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /usr/src/app

# Copy project files for dependency installation
COPY pyproject.toml uv.lock ./

# Install dependencies with nemo extra
RUN uv sync --extra nemo --no-dev

# Copy application code
COPY linto_stt ./linto_stt
COPY main.py ./
COPY healthcheck.sh ./
COPY test/bonjour.wav ./test/bonjour.wav

ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/linto_stt/backends/nemo/stt"

HEALTHCHECK CMD ./healthcheck.sh

ENTRYPOINT ["uv", "run", "main.py", "-b", "nemo"]
