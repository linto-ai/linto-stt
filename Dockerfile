# Dockerfile for building PyKaldi image from Ubuntu 16.04 image
FROM ubuntu:18.04
LABEL maintainer="irebai@linagora.com"

# Install necessary system packages
RUN apt-get update \
    && apt-get install -y \
    python3 \
    python3-pip \
    python2.7 \
    autoconf \
    automake \
    cmake \
    make \
    curl \
    g++ \
    git \
    graphviz \
    libatlas3-base \
    libtool \
    pkg-config \
    sox \
    subversion \
    bzip2 \
    unzip \
    wget \
    zlib1g-dev \
    ca-certificates \
    gfortran \
    patch \
    ffmpeg \
    nano && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Install necessary Python packages (pykaldi dependencies)
RUN pip install --upgrade pip \
    numpy \
    setuptools \
    pyparsing \
    ninja

## Install Protobuf, CLIF, Kaldi and PyKaldi and Clean installation
RUN git clone --depth 1 https://github.com/pykaldi/pykaldi.git /pykaldi \
    && cd /pykaldi/tools \
    && sed -i "s/make \-j4/make -j $(nproc)/g" ./install_kaldi.sh \
    && sed -i "s/\-j 2/-j $(nproc)/g" ./install_clif.sh \
    && sed -i "s/make \-j4/make -j $(nproc)/g" ./install_protobuf.sh \
    && ./check_dependencies.sh \
    && ./install_protobuf.sh \
    && ./install_clif.sh \
    && ./install_kaldi.sh \
    && cd /pykaldi \
    && python setup.py install \
    && rm -rf   /pykaldi/CMakeLists.txt \
                /pykaldi/LICENSE \
                /pykaldi/README.md \
                /pykaldi/setup.cfg \
                /pykaldi/setup.py \
                /pykaldi/docker \
                /pykaldi/docs \
                /pykaldi/extras \
                /pykaldi/pykaldi.egg-info \
                /pykaldi/tests \
                /pykaldi/build/CMakeCache.txt \
                /pykaldi/build/bdist.linux-x86_64 \
                /pykaldi/build/build.ninja \
                /pykaldi/build/cmake_install.cmake \
                /pykaldi/build/docs \
                /pykaldi/build/kaldi \
                /pykaldi/build/lib \
                /pykaldi/build/rules.ninja \
                /pykaldi/tools/check_dependencies.sh \
                /pykaldi/tools/clif* \
                /pykaldi/tools/find_python_library.py \
                /pykaldi/tools/install_* \
                /pykaldi/tools/protobuf \
                /pykaldi/tools/use_namespace.sh \
                /pykaldi/tools/kaldi/COPYING \
                /pykaldi/tools/kaldi/INSTALL \
                /pykaldi/tools/kaldi/README.md \
                /pykaldi/tools/kaldi/egs \
                /pykaldi/tools/kaldi/misc \
                /pykaldi/tools/kaldi/scripts \
                /pykaldi/tools/kaldi/windows \
    && mkdir -p /pykaldi/tools/kaldi/src_/lib \
    && mv  /pykaldi/tools/kaldi/src/base/libkaldi-base.so \
            /pykaldi/tools/kaldi/src/chain/libkaldi-chain.so \
            /pykaldi/tools/kaldi/src/cudamatrix/libkaldi-cudamatrix.so \
            /pykaldi/tools/kaldi/src/decoder/libkaldi-decoder.so \
            /pykaldi/tools/kaldi/src/feat/libkaldi-feat.so \
            /pykaldi/tools/kaldi/src/fstext/libkaldi-fstext.so \
            /pykaldi/tools/kaldi/src/gmm/libkaldi-gmm.so \
            /pykaldi/tools/kaldi/src/hmm/libkaldi-hmm.so \
            /pykaldi/tools/kaldi/src/ivector/libkaldi-ivector.so \
            /pykaldi/tools/kaldi/src/kws/libkaldi-kws.so \
            /pykaldi/tools/kaldi/src/lat/libkaldi-lat.so \
            /pykaldi/tools/kaldi/src/lm/libkaldi-lm.so \
            /pykaldi/tools/kaldi/src/matrix/libkaldi-matrix.so \
            /pykaldi/tools/kaldi/src/nnet/libkaldi-nnet.so \
            /pykaldi/tools/kaldi/src/nnet2/libkaldi-nnet2.so \
            /pykaldi/tools/kaldi/src/nnet3/libkaldi-nnet3.so \
            /pykaldi/tools/kaldi/src/online2/libkaldi-online2.so \
            /pykaldi/tools/kaldi/src/rnnlm/libkaldi-rnnlm.so \
            /pykaldi/tools/kaldi/src/sgmm2/libkaldi-sgmm2.so \
            /pykaldi/tools/kaldi/src/transform/libkaldi-transform.so \
            /pykaldi/tools/kaldi/src/tree/libkaldi-tree.so \
            /pykaldi/tools/kaldi/src/util/libkaldi-util.so \
            /pykaldi/tools/kaldi/src_/lib \
        && rm -rf /pykaldi/tools/kaldi/src && mv /pykaldi/tools/kaldi/src_ /pykaldi/tools/kaldi/src \
        && cd /pykaldi/tools/kaldi/tools && mkdir openfsttmp && mv openfst-*/lib openfst-*/include openfst-*/bin openfsttmp && rm openfsttmp/lib/*.a openfsttmp/lib/*.la && \
                rm -r openfst-*/* && mv openfsttmp/* openfst-*/ && rm -r openfsttmp

# Define the main folder
WORKDIR /usr/src/speech-to-text

# Install main service packages
RUN pip3 install flask flask-cors flask-swagger-ui configparser pyyaml logger librosa webrtcvad scipy sklearn gevent
RUN apt-get install -y libsox-fmt-all && pip3 install git+https://github.com/rabitt/pysox.git \
    && git clone https://github.com/irebai/pyBK.git /pykaldi/tools/pyBK \
    && cp /pykaldi/tools/pyBK/diarizationFunctions.py .

# Set environment variables
ENV PATH /pykaldi/tools/kaldi/egs/wsj/s5/utils/:$PATH

COPY tools.py .
COPY run.py .

EXPOSE 80

CMD python3 ./run.py