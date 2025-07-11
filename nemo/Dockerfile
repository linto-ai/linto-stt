# FROM nvcr.io/nvidia/pytorch:24.02-py3
FROM python:3.10
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
  libavdevice-dev && \
  rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install Cython
COPY nemo/requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

COPY punctuation/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

WORKDIR /usr/src/app

COPY celery_app /usr/src/app/celery_app
COPY http_server /usr/src/app/http_server
COPY websocket /usr/src/app/websocket
COPY document /usr/src/app/document
COPY nemo/stt /usr/src/app/stt
COPY punctuation ./punctuation
COPY nemo/docker-entrypoint.sh wait-for-it.sh healthcheck.sh ./
COPY test/bonjour.wav /usr/src/app/test/bonjour.wav

ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/stt"

# HEALTHCHECK CMD ./healthcheck.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
