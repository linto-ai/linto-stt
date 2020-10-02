## Kaldi ASR decoder
from kaldi.asr import NnetLatticeFasterOnlineRecognizer
from kaldi.decoder import (LatticeFasterDecoderOptions,
                           LatticeFasterOnlineDecoder)
from kaldi.nnet3 import NnetSimpleLoopedComputationOptions
from kaldi.online2 import (OnlineEndpointConfig,
                           OnlineIvectorExtractorAdaptationState,
                           OnlineNnetFeaturePipelineConfig,
                           OnlineNnetFeaturePipelineInfo,
                           OnlineNnetFeaturePipeline,
                           OnlineSilenceWeighting)
from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialWaveReader
from kaldi.matrix import Matrix, Vector
##############

## word to CTM
from kaldi.lat.align import (WordBoundaryInfoNewOpts,
                            WordBoundaryInfo,
                            word_align_lattice)
from kaldi.lat.functions import (compact_lattice_to_word_alignment,
                                 compact_lattice_shortest_path)
from kaldi.asr import NnetRecognizer
import kaldi.fstext as _fst
##############

## Speaker Diarization
from diarizationFunctions import *
import numpy as np
import librosa
from kaldi.ivector import (compute_vad_energy,
                           VadEnergyOptions)
from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.util.options import ParseOptions
##############

## other packages
import configparser, sys, os, re, sox, time, logging
from concurrent.futures import ThreadPoolExecutor
import scipy.io.wavfile
##############

class ASR:
    def __init__(self, AM_PATH, LM_PATH, CONFIG_FILES_PATH):
        self.log = logging.getLogger('__stt-standelone-worker__.ASR')
        self.AM_PATH = AM_PATH
        self.LM_PATH = LM_PATH
        self.CONFIG_FILES_PATH = CONFIG_FILES_PATH
    
    def run(self):
        def loadConfig(self):
            #get decoder parameters from "decode.cfg"
            decoder_settings = configparser.ConfigParser()
            decoder_settings.read(self.AM_PATH+'/decode.cfg')
            self.DECODER_SYS = decoder_settings.get('decoder_params', 'decoder')
            self.AM_FILE_PATH = decoder_settings.get('decoder_params', 'ampath')
            self.DECODER_MINACT = int(decoder_settings.get('decoder_params', 'min_active'))
            self.DECODER_MAXACT = int(decoder_settings.get('decoder_params', 'max_active'))
            self.DECODER_BEAM = float(decoder_settings.get('decoder_params', 'beam'))
            self.DECODER_LATBEAM = float(decoder_settings.get('decoder_params', 'lattice_beam'))
            self.DECODER_ACWT = float(decoder_settings.get('decoder_params', 'acwt'))
            self.DECODER_FSF = int(decoder_settings.get('decoder_params', 'frame_subsampling_factor'))

            #Prepare "online.conf"
            self.AM_PATH=self.AM_PATH+"/"+self.AM_FILE_PATH
            with open(self.AM_PATH+"/conf/online.conf") as f:
                values = f.readlines()
                with open(self.CONFIG_FILES_PATH+"/online.conf", 'w') as f:
                    for i in values:
                        f.write(i)
                    f.write("--ivector-extraction-config="+self.CONFIG_FILES_PATH+"/ivector_extractor.conf\n")
                    f.write("--mfcc-config="+self.AM_PATH+"/conf/mfcc.conf")

            #Prepare "ivector_extractor.conf"
            with open(self.AM_PATH+"/conf/ivector_extractor.conf") as f:
                values = f.readlines()
                with open(self.CONFIG_FILES_PATH+"/ivector_extractor.conf", 'w') as f:
                    for i in values:
                        f.write(i)
                    f.write("--splice-config="+self.AM_PATH+"/conf/splice.conf\n")
                    f.write("--cmvn-config="+self.AM_PATH+"/conf/online_cmvn.conf\n")
                    f.write("--lda-matrix="+self.AM_PATH+"/ivector_extractor/final.mat\n")
                    f.write("--global-cmvn-stats="+self.AM_PATH+"/ivector_extractor/global_cmvn.stats\n")
                    f.write("--diag-ubm="+self.AM_PATH+"/ivector_extractor/final.dubm\n")
                    f.write("--ivector-extractor="+self.AM_PATH+"/ivector_extractor/final.ie")
            
            #Prepare "word_boundary.int" if not exist
            if not os.path.exists(self.LM_PATH+"/word_boundary.int"):
                if os.path.exists(self.AM_PATH+"phones.txt"):
                    with open(self.AM_PATH+"phones.txt") as f:
                        phones = f.readlines()

                    with open(self.LM_PATH+"/word_boundary.int", "w") as f:
                        for phone in phones:
                            phone = phone.strip()
                            phone = re.sub('^<eps> .*','', phone)
                            phone = re.sub('^#\d+ .*','', phone)
                            if phone != '':
                                id = phone.split(' ')[1]
                                if '_I ' in phone:
                                    f.write(id+" internal\n")
                                elif '_B ' in phone:
                                    f.write(id+" begin\n")
                                elif '_E ' in phone:
                                    f.write(id+" end\n")
                                elif '_S ' in phone:
                                    f.write(id+" singleton\n")
                                else:
                                    f.write(id+" nonword\n")

                else:
                    raise ValueError('Neither word_boundary.int nor phones.txt exists!!!')
        
        try:
            # Define online feature pipeline
            self.log.info("Load decoder config")
            loadConfig(self)
            feat_opts = OnlineNnetFeaturePipelineConfig()
            self.endpoint_opts = OnlineEndpointConfig()
            po = ParseOptions("")
            feat_opts.register(po)
            self.endpoint_opts.register(po)
            po.read_config_file(self.CONFIG_FILES_PATH+"/online.conf")
            self.feat_info = OnlineNnetFeaturePipelineInfo.from_config(feat_opts)
            
            # Set metadata parameters
            self.samp_freq = self.feat_info.mfcc_opts.frame_opts.samp_freq
            self.frame_shift = self.feat_info.mfcc_opts.frame_opts.frame_shift_ms / 1000

            # Construct recognizer
            self.log.info("Load Decoder model")
            decoder_opts = LatticeFasterDecoderOptions()
            decoder_opts.beam = self.DECODER_BEAM
            decoder_opts.max_active = self.DECODER_MAXACT
            decoder_opts.min_active = self.DECODER_MINACT
            decoder_opts.lattice_beam = self.DECODER_LATBEAM
            self.decodable_opts = NnetSimpleLoopedComputationOptions()
            self.decodable_opts.acoustic_scale = self.DECODER_ACWT
            self.decodable_opts.frame_subsampling_factor = self.DECODER_FSF
            self.decodable_opts.frames_per_chunk = 150
            
            # Load Acoustic and graph models and other files
            self.transition_model, self.acoustic_model = NnetRecognizer.read_model(self.AM_PATH+"/final.mdl")
            graph = _fst.read_fst_kaldi(self.LM_PATH+"/HCLG.fst")
            self.decoder_graph = LatticeFasterOnlineDecoder(graph, decoder_opts)
            self.symbols = _fst.SymbolTable.read_text(self.LM_PATH+"/words.txt")
            self.info = WordBoundaryInfo.from_file(WordBoundaryInfoNewOpts(),self.LM_PATH+"/word_boundary.int")
            del graph, decoder_opts
        except Exception as e:
            self.log.error(e)
            raise ValueError("AM and LM loading failed!!! (see logs for more details)")

    def get_sample_rate(self):
        return self.samp_freq

    def get_frames(self,feat_pipeline):
        rows = feat_pipeline.num_frames_ready()
        cols = feat_pipeline.dim()
        frames = Matrix(rows,cols)
        feat_pipeline.get_frames(range(rows),frames)
        return frames[:,:self.feat_info.mfcc_opts.num_ceps], frames[:,self.feat_info.mfcc_opts.num_ceps:]
        # return feats + ivectors
        
    def compute_feat(self,audio):
        try:
            feat_pipeline = OnlineNnetFeaturePipeline(self.feat_info)
            feat_pipeline.accept_waveform(audio.sr, audio.getDataKaldyVector())
            feat_pipeline.input_finished()
        except Exception as e:
            self.log.error(e)
            raise ValueError("Feature extraction failed!!!")
        else:
            return feat_pipeline
        
    def decoder(self,feats):
        try:
            start_time = time.time()
            self.log.info("Start Decoding: %s" % (start_time))
            asr = NnetLatticeFasterOnlineRecognizer(self.transition_model, self.acoustic_model, self.decoder_graph,
                                                    self.symbols, decodable_opts= self.decodable_opts, endpoint_opts=self.endpoint_opts)
            asr.set_input_pipeline(feats)
            decode = asr.decode()
            self.log.info("Decode time in seconds: %s" % (time.time() - start_time))
        except Exception as e:
            self.log.error(e)
            raise ValueError("Decoder failed to transcribe the input audio!!!")
        else:
            return decode
        
    def wordTimestamp(self,decode):
        try:
            _fst.utils.scale_compact_lattice([[1.0, 0],[0, float(self.DECODER_ACWT)]], decode['lattice'])
            bestPath = compact_lattice_shortest_path(decode['lattice'])
            _fst.utils.scale_compact_lattice([[1.0, 0],[0, 1.0/float(self.DECODER_ACWT)]], bestPath)
            bestLattice = word_align_lattice(bestPath, self.transition_model, self.info, 0)
            alignment = compact_lattice_to_word_alignment(bestLattice[1])
            words = _fst.indices_to_symbols(self.symbols, alignment[0])
        except Exception as e:
            self.log.error(e)
            raise ValueError("Decoder failed to create the word timestamps!!!")
        else:
            return {
                "words":words,
                "start":alignment[1],
                "dur":alignment[2]
            }

class SpeakerDiarization:
    def __init__(self):
        self.log = logging.getLogger('__stt-standelone-worker__.SPKDiarization')

       ### MFCC FEATURES PARAMETERS
        self.frame_length_s=0.025
        self.frame_shift_s=0.01
        self.num_bins=40
        self.num_ceps=40
        self.low_freq=40
        self.high_freq=-200
        #####

        ### VAD PARAMETERS
        self.vad_ops = VadEnergyOptions()
        self.vad_ops.vad_energy_mean_scale = 0.9
        self.vad_ops.vad_energy_threshold = 5
        #vad_ops.vad_frames_context = 2
        #vad_ops.vad_proportion_threshold = 0.12
        #####

        ### Segment
        self.seg_length = 100 # Window size in frames
        self.seg_increment = 100 # Window increment after and before window in frames
        self.seg_rate = 100 # Window shifting in frames
        #####

        ### KBM
        self.minimumNumberOfInitialGaussians = 1024 # Minimum number of Gaussians in the initial pool
        self.maximumKBMWindowRate = 50 # Maximum window rate for Gaussian computation
        self.windowLength = 200 # Window length for computing Gaussians
        self.kbmSize = 320 # Number of final Gaussian components in the KBM
        self.useRelativeKBMsize = 1 # If set to 1, the KBM size is set as a proportion, given by "relKBMsize", of the pool size
        self.relKBMsize = 0.3 # Relative KBM size if "useRelativeKBMsize = 1" (value between 0 and 1).
        ######

        ### BINARY_KEY
        self.topGaussiansPerFrame = 5 # Number of top selected components per frame
        self.bitsPerSegmentFactor = 0.2 # Percentage of bits set to 1 in the binary keys
        ######

        ### CLUSTERING
        self.N_init = 16 # Number of initial clusters
        self.linkage = 0 # Set to one to perform linkage clustering instead of clustering/reassignment
        self.linkageCriterion = 'average' # Linkage criterion used if linkage==1 ('average', 'single', 'complete')
        self.metric = 'cosine' # Similarity metric: 'cosine' for cumulative vectors, and 'jaccard' for binary keys
        ######

        ### CLUSTERING_SELECTION
        self.metric_clusteringSelection = 'cosine' # Distance metric used in the selection of the output clustering solution ('jaccard','cosine')
        self.bestClusteringCriterion = 'elbow' # Method employed for number of clusters selection. Can be either 'elbow' for an elbow criterion based on within-class sum of squares (WCSS) or 'spectral' for spectral clustering
        self.sigma = 1 # Spectral clustering parameters, employed if bestClusteringCriterion == spectral
        self.percentile = 40
        self.maxNrSpeakers = 16 # If known, max nr of speakers in a sesssion in the database. This is to limit the effect of changes in very small meaningless eigenvalues values generating huge eigengaps
        ######

        ### RESEGMENTATION
        self.resegmentation = 1 # Set to 1 to perform re-segmentation
        self.modelSize = 6 # Number of GMM components
        self.nbIter = 10 # Number of expectation-maximization (EM) iterations
        self.smoothWin = 100 # Size of the likelihood smoothing window in nb of frames
        ######
    
    def set_maxNrSpeakers(self,nbr):
        self.maxNrSpeakers = nbr
    
    def compute_feat_Librosa(self,audio):
        try:
            self.log.info("Start feature extraction: %s" % (time.time()))
            if audio.sr == 16000:
                self.low_freq=20
                self.high_freq=7600
            data = audio.data/32768
            frame_length_inSample = self.frame_length_s * audio.sr
            hop = int(self.frame_shift_s * audio.sr)
            NFFT = int(2**np.ceil(np.log2(frame_length_inSample)))
            mfccNumpy = librosa.feature.mfcc(y=data,
                                             sr=audio.sr,
                                             dct_type=2,
                                             n_mfcc=self.num_ceps,
                                             n_mels=self.num_bins,
                                             n_fft=NFFT,
                                             hop_length=hop,
                                             fmin=self.low_freq,
                                             fmax=self.high_freq).T
        except Exception as e:
            self.log.error(e)
            raise ValueError("Speaker diarization failed when extracting features!!!")
        else:
            return mfccNumpy

    def compute_feat_KALDI(self,audio):
        try:
            self.log.info("Start feature extraction: %s" % (time.time()))
            po = ParseOptions("")
            mfcc_opts = MfccOptions()
            mfcc_opts.use_energy = False
            mfcc_opts.frame_opts.samp_freq = audio.sr
            mfcc_opts.frame_opts.frame_length_ms = self.frame_length_s*1000
            mfcc_opts.frame_opts.frame_shift_ms = self.frame_shift_s*1000
            mfcc_opts.frame_opts.allow_downsample = False
            mfcc_opts.mel_opts.num_bins = self.num_bins
            mfcc_opts.mel_opts.low_freq = self.low_freq
            mfcc_opts.mel_opts.high_freq = self.high_freq
            mfcc_opts.num_ceps = self.num_ceps
            mfcc_opts.register(po)
            
            # Create MFCC object and obtain sample frequency
            mfccObj = Mfcc(mfcc_opts)
            mfccKaldi = mfccObj.compute_features(audio.getDataKaldyVector(), audio.sr, 1.0)
        except Exception as e:
            self.log.error(e)
            raise ValueError("Speaker diarization failed while extracting features!!!")
        else:
            return mfccKaldi
        
    def computeVAD_WEBRTC(self, audio):
        try:
            self.log.info("Start VAD: %s" % (time.time()))
            data = audio.data/32768
            hop = 30
            va_framed = py_webrtcvad(data, fs=audio.sr, fs_vad=audio.sr, hoplength=hop, vad_mode=0)
            segments = get_py_webrtcvad_segments(va_framed,audio.sr)
            maskSAD = np.zeros([1,nFeatures])
            for seg in segments:
                start=int(np.round(seg[0]/frame_shift_s))
                end=int(np.round(seg[1]/frame_shift_s))
                maskSAD[0][start:end]=1
        except Exception as e:
            self.log.error(e)
            raise ValueError("Speaker diarization failed while voice activity detection!!!")
        else:
            return maskSAD
    
    def computeVAD_KALDI(self, audio, feats=None):
        try:
            self.log.info("Start VAD: %s" % (time.time()))
            vadStream = compute_vad_energy(self.vad_ops,feats)
            vad = Vector(vadStream)
            VAD = vad.numpy()
                        
            ### segmentation
            occurence=[]
            value=[]
            occurence.append(1)
            value.append(VAD[0])

            # compute the speech and non-speech frames
            for i in range(1,len(VAD)):
                if value[-1] == VAD[i]:
                    occurence[-1]+=1
                else:
                    occurence.append(1)
                    value.append(VAD[i])

            # filter the speech and non-speech segments that are below 30 frames
            i = 0
            while(i < len(occurence)):
                if i != 0 and (occurence[i] < 30 or value[i-1] == value[i]):
                    occurence[i-1] += occurence[i]
                    del value[i]
                    del occurence[i]
                else:
                    i+=1

            # split if and only if the silence is above 50 frames
            i = 0
            while(i < len(occurence)):
                if i != 0 and ((occurence[i] < 30 and value[i] == 0.0) or value[i-1] == value[i]):
                    occurence[i-1] += occurence[i]
                    del value[i]
                    del occurence[i]
                else:
                    i+=1
            
            # compute VAD mask
            maskSAD = np.zeros(len(VAD))
            start=0
            for i in range(len(occurence)):
                if value[i] == 1.0:
                    end=start+occurence[i]
                    maskSAD[start:end] = 1
                    start=end
                else:
                    start += occurence[i]

            maskSAD = np.expand_dims(maskSAD, axis=0)
        except ValueError as v:
            self.log.error(v)
        except Exception as e:
            self.log.error(e)
            raise ValueError("Speaker diarization failed while voice activity detection!!!")
        else:
            return maskSAD

    def run(self, audio, feats=None):
        try:
            def getSegments(frameshift, finalSegmentTable, finalClusteringTable, dur):
                numberOfSpeechFeatures = finalSegmentTable[-1,2].astype(int)+1
                solutionVector = np.zeros([1,numberOfSpeechFeatures])
                for i in np.arange(np.size(finalSegmentTable,0)):
                    solutionVector[0,np.arange(finalSegmentTable[i,1],finalSegmentTable[i,2]+1).astype(int)]=finalClusteringTable[i]
                seg = np.empty([0,3]) 
                solutionDiff = np.diff(solutionVector)[0]
                first = 0
                for i in np.arange(0,np.size(solutionDiff,0)):
                    if solutionDiff[i]:
                        last = i+1
                        seg1 = (first)*frameshift
                        seg2 = (last-first)*frameshift
                        seg3 = solutionVector[0,last-1]
                        if seg.shape[0] != 0 and seg3 == seg[-1][2]:
                            seg[-1][1] += seg2
                        elif seg3 and seg2 > 0.3: # and seg2 > 0.1
                            seg = np.vstack((seg,[seg1,seg2,seg3]))
                        first = i+1
                last = np.size(solutionVector,1)
                seg1 = (first-1)*frameshift
                seg2 = (last-first+1)*frameshift
                seg3 = solutionVector[0,last-1]
                if seg3 == seg[-1][2]:
                    seg[-1][1] += seg2
                elif seg3 and seg2 > 0.3: # and seg2 > 0.1
                    seg = np.vstack((seg,[seg1,seg2,seg3]))
                seg = np.vstack((seg,[dur,-1,-1]))
                seg[0][0]=0.0
                return seg
        

            start_time = time.time()
            self.log.info("Start Speaker Diarization: %s" % (start_time))
            if self.maxNrSpeakers == 1 or audio.dur < 3:
                self.log.info("Speaker Diarization time in seconds: %s" % (time.time() - start_time))
                return [[0, audio.dur, 1],
                        [audio.dur, -1, -1]]
            if feats == None:
                feats = self.compute_feat_KALDI(audio)
            nFeatures = feats.shape[0]
            maskSAD = self.computeVAD_KALDI(audio,feats)
            maskUEM = np.ones([1,nFeatures])

            mask = np.logical_and(maskUEM,maskSAD)    
            mask = mask[0][0:nFeatures]
            nSpeechFeatures=np.sum(mask)
            speechMapping = np.zeros(nFeatures)
            #you need to start the mapping from 1 and end it in the actual number of features independently of the indexing style
            #so that we don't lose features on the way
            speechMapping[np.nonzero(mask)] = np.arange(1,nSpeechFeatures+1)
            data=feats[np.where(mask==1)]
            del feats

            segmentTable=getSegmentTable(mask,speechMapping,self.seg_length,self.seg_increment,self.seg_rate)
            numberOfSegments=np.size(segmentTable,0)
            #create the KBM
            #set the window rate in order to obtain "minimumNumberOfInitialGaussians" gaussians
            if np.floor((nSpeechFeatures-self.windowLength)/self.minimumNumberOfInitialGaussians) < self.maximumKBMWindowRate:
                windowRate = int(np.floor((np.size(data,0)-self.windowLength)/self.minimumNumberOfInitialGaussians))
            else:
                windowRate = int(self.maximumKBMWindowRate)
            
            if windowRate == 0:
                raise ValueError('The audio is to short in order to perform the speaker diarization!!!')
            
            poolSize = np.floor((nSpeechFeatures-self.windowLength)/windowRate)
            if  self.useRelativeKBMsize:
                kbmSize = int(np.floor(poolSize*self.relKBMsize))
            else:
                kbmSize = int(self.kbmSize)
            
            #Training pool of',int(poolSize),'gaussians with a rate of',int(windowRate),'frames'
            kbm, gmPool = trainKBM(data,self.windowLength,windowRate,kbmSize)
            
            #'Selected',kbmSize,'gaussians from the pool'
            Vg = getVgMatrix(data,gmPool,kbm,self.topGaussiansPerFrame)
            
            #'Computing binary keys for all segments... '
            segmentBKTable, segmentCVTable = getSegmentBKs(segmentTable, kbmSize, Vg, self.bitsPerSegmentFactor, speechMapping)
            
            #'Performing initial clustering... '
            initialClustering = np.digitize(np.arange(numberOfSegments),np.arange(0,numberOfSegments,numberOfSegments/self.N_init))
            
            
            #'Performing agglomerative clustering... '
            if self.linkage:
                finalClusteringTable, k = performClusteringLinkage(segmentBKTable, segmentCVTable, self.N_init, self.linkageCriterion, self.metric)
            else:
                finalClusteringTable, k = performClustering(speechMapping, segmentTable, segmentBKTable, segmentCVTable, Vg, self.bitsPerSegmentFactor, kbmSize, self.N_init, initialClustering, self.metric)

            #'Selecting best clustering...'
            if self.bestClusteringCriterion == 'elbow':
                bestClusteringID = getBestClustering(self.metric_clusteringSelection, segmentBKTable, segmentCVTable, finalClusteringTable, k, self.maxNrSpeakers)
            elif self.bestClusteringCriterion == 'spectral':
                bestClusteringID = getSpectralClustering(self.metric_clusteringSelection,finalClusteringTable,self.N_init,segmentBKTable,segmentCVTable,k,self.sigma,self.percentile,self.maxNrSpeakers)+1
                
            if self.resegmentation and np.size(np.unique(finalClusteringTable[:,bestClusteringID.astype(int)-1]),0)>1:
                finalClusteringTableResegmentation,finalSegmentTable = performResegmentation(data,speechMapping, mask,finalClusteringTable[:,bestClusteringID.astype(int)-1],segmentTable,self.modelSize,self.nbIter,self.smoothWin,nSpeechFeatures)
                seg = getSegments(self.frame_shift_s,finalSegmentTable, np.squeeze(finalClusteringTableResegmentation), audio.dur)
            else:
                seg = getSegmentationFile(self.frame_shift_s,segmentTable, finalClusteringTable[:,bestClusteringID.astype(int)-1])
            self.log.info("Speaker Diarization time in seconds: %s" % (time.time() - start_time))
        except ValueError as v:
            self.log.info(v)
            return [[0, audio.dur, 1],
                    [audio.dur, -1, -1]]
        except Exception as e:
            self.log.error(e)
            raise ValueError("Speaker Diarization failed!!!")
        else:
            return seg
        
class SttStandelone:
    def __init__(self,metadata=False,spkDiarization=False):
        self.log = logging.getLogger('__stt-standelone-worker__.SttStandelone')
        self.metadata = metadata
        self.spkDiarization = spkDiarization
        self.timestamp = True if self.metadata or self.spkDiarization else False
        
    def run(self,audio,asr,spk):
        feats = asr.compute_feat(audio)
        mfcc, ivector = asr.get_frames(feats)
        if self.spkDiarization:
            with ThreadPoolExecutor(max_workers=2) as executor:
                thrd1 = executor.submit(asr.decoder, feats)
                thrd2 = executor.submit(spk.run, audio, mfcc)
                decode = thrd1.result()
                spkSeg = thrd2.result()
        else:
            decode = asr.decoder(feats)
            spkSeg = []
        
        if self.timestamp:
            timestamps = asr.wordTimestamp(decode)
            output = self.getOutput(timestamps,asr.frame_shift, asr.decodable_opts.frame_subsampling_factor,spkSeg)
            if self.metadata:
                return output
            else:
                return {"text":output["text"]}
        else:
            return decode["text"]

    def getOutput(self,timestamps,frame_shift, frame_subsampling, spkSeg = []):
        output = {}
        if len(spkSeg) == 0:
            text = ""
            output["words"] = []
            for i in range(len(timestamps["words"])):
                if timestamps["words"][i] != "<eps>":
                    meta = {}
                    meta["word"] = timestamps["words"][i]
                    meta["btime"] = round(timestamps["start"][i] * frame_shift * frame_subsampling,2)
                    meta["etime"] = round((timestamps["start"][i]+timestamps["dur"][i]) * frame_shift * frame_subsampling, 2)
                    output["words"].append(meta)
                    text += " "+meta["word"]
            output["text"] = text
        else:
            output["speakers"] = []
            output["text"] = []
            j = 0
            newSpk = 1
            for i in range(len(timestamps["words"])):
                if timestamps["words"][i] != "<eps>":
                    if newSpk:
                        speaker = {}
                        speaker["speaker_id"] = "spk_"+str(int(spkSeg[j][2]))
                        speaker["words"] = []
                        txtSpk = speaker["speaker_id"]+":"
                        newSpk = 0
                    word = {}
                    word["word"] = timestamps["words"][i]
                    word["btime"] = round(timestamps["start"][i] * frame_shift * frame_subsampling,2)
                    word["etime"] = round((timestamps["start"][i]+timestamps["dur"][i]) * frame_shift * frame_subsampling, 2)
                    speaker["words"].append(word)
                    txtSpk += " "+word["word"]
                    if word["etime"] > spkSeg[j+1][0]:
                        speaker["btime"] = speaker["words"][0]["btime"]
                        speaker["etime"] = speaker["words"][-1]["etime"]
                        output["speakers"].append(speaker)
                        output["text"].append(txtSpk)
                        newSpk = 1
                        j += 1
            #add the last speaker to the output speakers
            speaker["btime"] = speaker["words"][0]["btime"]
            speaker["etime"] = speaker["words"][-1]["etime"]
            output["speakers"].append(speaker)
            output["text"].append(txtSpk)
        return output
        
        
class Audio:
    def __init__(self,sr):
        self.log = logging.getLogger('__stt-standelone-worker__.Audio')
        self.bit = 16
        self.channels = 1
        self.sr = sr
    
    def set_logger(self,log):
        self.log = log
    
    def read_audio(self, audio):
        try:
            data, sr = librosa.load(audio,sr=None)
            if sr != self.sr:
                self.log.info('Resample audio file: '+str(sr)+'Hz -> '+str(self.sr)+'Hz')
                data = librosa.resample(data, sr, self.sr)
            data = (data * 32767).astype(np.int16)
            self.data = data
            self.dur = len(self.data) / self.sr
        except Exception as e:
            self.log.error(e)
            raise ValueError("The uploaded file format is not supported!!!")
    
    def getDataKaldyVector(self):
        return Vector(self.data)