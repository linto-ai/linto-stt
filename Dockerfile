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