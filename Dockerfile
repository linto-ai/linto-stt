<<<<<<< HEAD
FROM ubuntu:18.04
LABEL maintainer="irebai@linagora.com"

RUN apt-get update &&\
    apt-get install -y \
    python2.7   \
    python3     \
    python3-pip \
    git  \
    swig \
    nano \
    sox  \
    automake wget unzip build-essential libtool zlib1g-dev locales libatlas-base-dev ca-certificates gfortran subversion &&\
    apt-get clean

## Build kaldi and Clean installation (intel, openfst, src/*)
RUN git clone --depth 1 https://github.com/kaldi-asr/kaldi.git /opt/kaldi && \
    cd /opt/kaldi/tools && \
    ./extras/install_mkl.sh && \
    make -j $(nproc) && \
    cd /opt/kaldi/src && \
    ./configure --shared && \
    make depend -j $(nproc) && \
    make -j $(nproc) && \
    mkdir -p /opt/kaldi/src_ && \
    mv       /opt/kaldi/src/base \
             /opt/kaldi/src/chain \
             /opt/kaldi/src/cudamatrix \
             /opt/kaldi/src/decoder \
             /opt/kaldi/src/feat \
             /opt/kaldi/src/fstext \
             /opt/kaldi/src/gmm \
             /opt/kaldi/src/hmm \
             /opt/kaldi/src/ivector \
             /opt/kaldi/src/kws \
             /opt/kaldi/src/lat \
             /opt/kaldi/src/lm \
             /opt/kaldi/src/matrix \
             /opt/kaldi/src/nnet \
             /opt/kaldi/src/nnet2 \
             /opt/kaldi/src/nnet3 \
             /opt/kaldi/src/online2 \
             /opt/kaldi/src/rnnlm \
             /opt/kaldi/src/sgmm2 \
             /opt/kaldi/src/transform \
             /opt/kaldi/src/tree \
             /opt/kaldi/src/util \
             /opt/kaldi/src/itf \
             /opt/kaldi/src/lib /opt/kaldi/src_ && \
    cd /opt/kaldi && rm -r src && mv src_ src && rm src/*/*.cc && rm src/*/*.o && rm src/*/*.so && \
    cd /opt/intel/mkl/lib && rm -f intel64/*.a intel64_lin/*.a && \
    cd /opt/kaldi/tools && mkdir openfst_ && mv openfst-*/lib openfst-*/include openfst-*/bin openfst_ && rm openfst_/lib/*.so* openfst_/lib/*.la && \
    rm -r openfst-*/* && mv openfst_/* openfst-*/ && rm -r openfst_

# Install python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# build VOSK KALDI
RUN git clone --depth 1 https://github.com/irebai/vosk-api.git /opt/vosk-api && cd /opt/vosk-api/python && \
    export KALDI_ROOT=/opt/kaldi && \
    export KALDI_MKL=1 && \
    python3 setup.py install --user --single-version-externally-managed --root=/

# Define the main folder
WORKDIR /usr/src/speech-to-text

COPY tools.py run.py docker-entrypoint.sh wait-for-it.sh ./

EXPOSE 80

# Entrypoint handles the passed arguments
ENTRYPOINT ["./docker-entrypoint.sh"]
=======
FROM python:3.9
LABEL maintainer="irebai@linagora.com, rbaraglia@linagora.com"

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
        git \
        zlib1g-dev \
        automake \
        autoconf \
        libtool \
        pkg-config \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build vosk-kaldi
RUN git clone -b vosk --single-branch https://github.com/alphacep/kaldi /opt/kaldi \
    && cd /opt/kaldi/tools \
    && sed -i 's:status=0:exit 0:g' extras/check_dependencies.sh \
    && sed -i 's:--enable-ngram-fsts:--enable-ngram-fsts --disable-bin:g' Makefile \
    && make -j $(nproc) openfst cub \
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
    && make -j $(nproc) online2 lm rnnlm

# Install python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install Custom Vosk API
RUN git clone --depth 1 https://github.com/alphacep/vosk-api /opt/vosk-api && cd /opt/vosk-api/python && \
    cd /opt/vosk-api/src \
    && KALDI_MKL=$KALDI_MKL KALDI_ROOT=/opt/kaldi make -j $(nproc) \
    && cd /opt/vosk-api/python \
    && python3 ./setup.py install

WORKDIR /usr/src/app

COPY stt /usr/src/app/stt
COPY celery_app /usr/src/app/celery_app
COPY http_server /usr/src/app/http_server
COPY websocket /usr/src/app/websocket
COPY document /usr/src/app/document
COPY docker-entrypoint.sh wait-for-it.sh healthcheck.sh ./
COPY lin_to_vosk.py /usr/src/app/lin_to_vosk.py

RUN mkdir -p /var/log/supervisor/

ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/stt"

HEALTHCHECK CMD ./healthcheck.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
>>>>>>> next
