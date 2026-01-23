FROM python:3.12
LABEL maintainer="contact@linto.ai"

ARG KALDI_MKL

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        nano \
        bzip2 \
        unzip \
        xz-utils \
        sox \
        g++ \
        make \
        cmake \
        curl \
        netcat-traditional \
        git \
        zlib1g-dev \
        automake \
        autoconf \
        libtool \
        pkg-config \
        ca-certificates \
        gosu \
    && rm -rf /var/lib/apt/lists/*

# Build vosk-kaldi
RUN git clone -b vosk --single-branch https://github.com/alphacep/kaldi /opt/kaldi \
    && cd /opt/kaldi/tools \
    && sed -i 's:status=0:exit 0:g' extras/check_dependencies.sh \
    && sed -i 's:--enable-ngram-fsts:--enable-ngram-fsts --disable-bin:g' Makefile \
    && make -j $(( $(nproc) < 8 ? $(nproc) : 8 )) openfst cub \
    && if [ "x$KALDI_MKL" != "x1" ] ; then \
          extras/install_openblas_clapack.sh; \
       else \
          extras/install_mkl.sh; \
       fi \
    && cd /opt/kaldi/src \
    && if [ "x$KALDI_MKL" != "x1" ] ; then \
          ./configure --mathlib=OPENBLAS_CLAPACK --shared; \
       else \
          ./configure --mathlib=MKL --shared; \
       fi \
    && sed -i 's:-msse -msse2:-msse -msse2:g' kaldi.mk \
    && sed -i 's: -O1 : -O3 :g' kaldi.mk \
    && make -j $(( $(nproc) < 8 ? $(nproc) : 8 )) online2 lm rnnlm

# Install Custom Vosk API
RUN git clone --depth 1 https://github.com/alphacep/vosk-api /opt/vosk-api && cd /opt/vosk-api/python && \
    cd /opt/vosk-api/src \
    && KALDI_MKL=$KALDI_MKL KALDI_ROOT=/opt/kaldi make -j $(( $(nproc) < 8 ? $(nproc) : 8 )) \
    && cd /opt/vosk-api/python \
    && python3 ./setup.py install

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /usr/src/app

# Copy project files for dependency installation
COPY pyproject.toml uv.lock ./

# Install dependencies with kaldi extra
RUN uv sync --extra kaldi --no-dev

# Copy application code
COPY linto_stt ./linto_stt
COPY main.py ./
COPY healthcheck.sh ./
COPY test/bonjour.wav ./test/bonjour.wav

ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/linto_stt/backends/kaldi/stt"

HEALTHCHECK CMD ./healthcheck.sh

ENTRYPOINT ["uv", "run", "main.py", "-b", "kaldi"]
