FROM debian:9
MAINTAINER Ilyes REBAI <irebai@linagora.com>

# Install all our dependencies and set some required build changes
RUN apt-get update &&\
    apt-get install -y \
    python2.7 \
    python3   \
    python-dev \
    python3-dev \
    python-pip \
    python3-pip \
    autoconf \
    automake \
    unzip \
    bc \
    bzip2 \
    default-jre \
    g++ \
    git \
    gzip \
    libatlas3-base \
    libtool-bin \
    make \
    sox \
    libav-tools \
    subversion \
    vorbis-tools \
    wget \
    zlib1g-dev &&\
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    ln -s /usr/bin/python2.7 /usr/bin/python ; ln -s -f bash /bin/sh

ENV BASE_DIR /opt/speech-to-text


# Build kaldi
## Install main libraries
RUN cd /opt && git clone https://github.com/kaldi-asr/kaldi.git && \
    cd /opt/kaldi/tools && make -j$(nproc)

#Install MKL package
RUN cd /opt/kaldi/tools && \
    extras/install_mkl.sh

## Install main functions
RUN cd /opt/kaldi/src && \
    sed -i -e ':a;N;$!ba;s:\\\n::g' Makefile && \
    sed -i -e 's:^SUBDIRS = .*$:SUBDIRS = base matrix util feat tree gmm transform fstext hmm lm decoder lat cudamatrix nnet bin nnet2 nnet3 chain ivector online2:g' -e 's:^MEMTESTDIRS = .*$:MEMTESTDIRS = :g' Makefile && \
    ./configure --shared && make depend -j$(nproc) && make -j$(nproc) && rm */*{.a,.o}

RUN apt install -y libatlas-dev
RUN pip2 install flask configparser requests

RUN echo "/opt/kaldi/src/lib/" > /etc/ld.so.conf.d/kaldi.conf && \
    echo "/opt/kaldi/tools/openfst/lib/" >> /etc/ld.so.conf.d/kaldi.conf && \
    ldconfig
    
COPY Makefile /opt
RUN cd /opt && ./Makefile

WORKDIR $BASE_DIR
COPY . .

RUN ln -s /opt/kaldi-nnet3-latgen-faster /opt/kaldi-nnet2-latgen-faster  /usr/bin/
RUN mkdir -p /opt/tmp

CMD ./run.py
