FROM python:3.9
LABEL maintainer="jlouradour@linagora.com"

ARG KALDI_MKL

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        nano \
        bzip2 \
        unzip \
        xz-utils \
        sox \
        ffmpeg \
        g++ \
        make \
        cmake \
        git \
        zlib1g-dev \
        automake \
        autoconf \
        libtool \
        pkg-config \
        ca-certificates

RUN rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt ./
RUN pip install --force-reinstall --no-cache-dir -r requirements.txt

# Download alignment model
COPY load_alignment_model.py ./
RUN python3 load_alignment_model.py

# Cleaning
RUN rm requirements.txt load_alignment_model.py

WORKDIR /usr/src/app

COPY stt /usr/src/app/stt
COPY celery_app /usr/src/app/celery_app
COPY http_server /usr/src/app/http_server
COPY websocket /usr/src/app/websocket
COPY document /usr/src/app/document
COPY docker-entrypoint.sh wait-for-it.sh healthcheck.sh ./

ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/stt"

HEALTHCHECK CMD ./healthcheck.sh

ENTRYPOINT ["./docker-entrypoint.sh"]