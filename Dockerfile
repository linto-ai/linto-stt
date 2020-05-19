FROM debian:9
LABEL maintainer="irebai@linagora.com"

# Install all our dependencies and set some required build changes
RUN apt-get update &&\
    apt-get install -y \
    python2.7 \
    python3   \
    python-dev \
    python3-dev \
    python-pip \
    python3-pip \
    g++ make automake autoconf bzip2 unzip wget sox libtool git subversion zlib1g-dev ca-certificates gfortran patch ffmpeg nano && \
    apt-get clean

## Build kaldi and Clean installation (intel, openfst, src/*)
RUN git clone --depth 1 https://github.com/kaldi-asr/kaldi.git /opt/kaldi && \
    cd /opt/kaldi && \
    cd /opt/kaldi/tools && \
    ./extras/install_mkl.sh && \
    make -j $(nproc) && \
    cd /opt/kaldi/src && \
    ./configure --shared && \
    make depend -j $(nproc) && \
    make -j $(nproc) && \
    mkdir -p /opt/kaldi/src_/lib /opt/kaldi/src_/bin && \
    mv /opt/kaldi/src/base/libkaldi-base.so \
       /opt/kaldi/src/chain/libkaldi-chain.so \
       /opt/kaldi/src/cudamatrix/libkaldi-cudamatrix.so \
       /opt/kaldi/src/decoder/libkaldi-decoder.so \
       /opt/kaldi/src/feat/libkaldi-feat.so \
       /opt/kaldi/src/fstext/libkaldi-fstext.so \
       /opt/kaldi/src/gmm/libkaldi-gmm.so \
       /opt/kaldi/src/hmm/libkaldi-hmm.so \
       /opt/kaldi/src/ivector/libkaldi-ivector.so \
       /opt/kaldi/src/kws/libkaldi-kws.so \
       /opt/kaldi/src/lat/libkaldi-lat.so \
       /opt/kaldi/src/lm/libkaldi-lm.so \
       /opt/kaldi/src/matrix/libkaldi-matrix.so \
       /opt/kaldi/src/nnet/libkaldi-nnet.so \
       /opt/kaldi/src/nnet2/libkaldi-nnet2.so \
       /opt/kaldi/src/nnet3/libkaldi-nnet3.so \
       /opt/kaldi/src/online2/libkaldi-online2.so \
       /opt/kaldi/src/rnnlm/libkaldi-rnnlm.so \
       /opt/kaldi/src/sgmm2/libkaldi-sgmm2.so \
       /opt/kaldi/src/transform/libkaldi-transform.so \
       /opt/kaldi/src/tree/libkaldi-tree.so \
       /opt/kaldi/src/util/libkaldi-util.so \
       /opt/kaldi/src_/lib && \
    mv /opt/kaldi/src/online2bin/online2-wav-nnet2-latgen-faster \
       /opt/kaldi/src/online2bin/online2-wav-nnet3-latgen-faster \
       /opt/kaldi/src/latbin/lattice-1best \
       /opt/kaldi/src/latbin/lattice-align-words \
       /opt/kaldi/src/latbin/nbest-to-ctm /opt/kaldi/src_/bin && \
    rm -rf /opt/kaldi/src && mv /opt/kaldi/src_ /opt/kaldi/src && \
    cd /opt/kaldi/src && rm -f lmbin/*.cc lmbin/*.o lmbin/Makefile fstbin/*.cc fstbin/*.o fstbin/Makefile bin/*.cc bin/*.o bin/Makefile && \
    cd /opt/intel/mkl/lib && rm -f intel64/*.a intel64_lin/*.a && \
    cd /opt/kaldi/tools && mkdir openfsttmp && mv openfst-*/lib openfst-*/include openfst-*/bin openfsttmp && rm openfsttmp/lib/*.a openfsttmp/lib/*.la && \
    rm -r openfst-*/* && mv openfsttmp/* openfst-*/ && rm -r openfsttmp

## Install python packages
RUN pip3 install flask flask-cors flask-swagger-ui configparser pyyaml

## Create symbolik links
RUN cd /opt/kaldi/src/bin && \
    ln -s online2-wav-nnet2-latgen-faster kaldi-nnet2-latgen-faster && \
    ln -s online2-wav-nnet3-latgen-faster kaldi-nnet3-latgen-faster && \
    ln -s lattice-1best kaldi-lattice-1best && \
    ln -s lattice-align-words kaldi-lattice-align-words && \
    ln -s nbest-to-ctm kaldi-nbest-to-ctm

# Set environment variables
ENV PATH /opt/kaldi/src/bin:/opt/kaldi/egs/wsj/s5/utils/:$PATH

WORKDIR /usr/src/speech-to-text
COPY run.py .

EXPOSE 80

CMD python3 ./run.py
