echo "Compile nnet2_decoder"
g++ -std=c++11 -L/usr/lib/atlas-base/atlas -L/opt/kaldi/tools/openfst/lib -L/opt/kaldi/src/lib -lblas -lkaldi-decoder -lkaldi-lat -lkaldi-fstext -lkaldi-hmm -lkaldi-transform -lkaldi-gmm -lkaldi-tree -lkaldi-util -lkaldi-matrix -lkaldi-base -lkaldi-nnet3 -lkaldi-online2 -lfst -lkaldi-cudamatrix -lkaldi-ivector -I /opt/kaldi/src -I /opt/kaldi/tools/openfst/include /opt/kaldi/src/online2bin/online2-wav-nnet2-latgen-faster.cc -o kaldi-nnet2-latgen-faster /opt/kaldi/src/lib/libkaldi-feat.so /opt/kaldi/src/lib/libkaldi-nnet2.so -lrt -lm -lpthread

echo "Compile nnet3_decoder"
g++ -std=c++11 -L/usr/lib/atlas-base/atlas -L/opt/kaldi/tools/openfst/lib -L/opt/kaldi/src/lib -lblas -lkaldi-decoder -lkaldi-lat -lkaldi-fstext -lkaldi-hmm -lkaldi-transform -lkaldi-gmm -lkaldi-tree -lkaldi-util -lkaldi-matrix -lkaldi-base -lkaldi-nnet3 -lkaldi-online2 -lfst -lkaldi-cudamatrix -lkaldi-ivector -I /opt/kaldi/src -I /opt/kaldi/tools/openfst/include /opt/kaldi/src/online2bin/online2-wav-nnet3-latgen-faster.cc -o kaldi-nnet3-latgen-faster /opt/kaldi/src/lib/libkaldi-feat.so -lrt -lm -lpthread

