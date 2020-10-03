# Kaldi ASR decoder
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

# word to CTM
from kaldi.lat.align import (WordBoundaryInfoNewOpts,
                             WordBoundaryInfo,
                             word_align_lattice)
from kaldi.lat.functions import (compact_lattice_to_word_alignment,
                                 compact_lattice_shortest_path)
from kaldi.asr import NnetRecognizer
import kaldi.fstext as _fst
##############

# Speaker Diarization
from diarizationFunctions import *
import numpy as np
import librosa
from kaldi.ivector import (compute_vad_energy,
                           VadEnergyOptions)
from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.util.options import ParseOptions
##############

# other packages
import configparser, sys, os, re, time, logging, yaml
from flask_swagger_ui import get_swaggerui_blueprint
##############


class ASR:
    def __init__(self, AM_PATH, LM_PATH, CONFIG_FILES_PATH):
        self.log = logging.getLogger('__stt-standelone-worker__.ASR')
        self.AM_PATH = AM_PATH
        self.LM_PATH = LM_PATH
        self.CONFIG_FILES_PATH = CONFIG_FILES_PATH
        self.LoadModels()
        
    def LoadModels(self):
        try:
            # Define online feature pipeline
            po = ParseOptions("")

            decoder_opts = LatticeFasterDecoderOptions()
            self.endpoint_opts = OnlineEndpointConfig()
            self.decodable_opts = NnetSimpleLoopedComputationOptions()
            feat_opts = OnlineNnetFeaturePipelineConfig()


            decoder_opts.register(po)
            self.endpoint_opts.register(po)
            self.decodable_opts.register(po)
            feat_opts.register(po)

            po.read_config_file(self.CONFIG_FILES_PATH+"/online.conf")
            self.feat_info = OnlineNnetFeaturePipelineInfo.from_config(
                feat_opts)

            # Set metadata parameters
            self.samp_freq = self.feat_info.mfcc_opts.frame_opts.samp_freq
            self.frame_shift = self.feat_info.mfcc_opts.frame_opts.frame_shift_ms / 1000
            self.acwt = self.decodable_opts.acoustic_scale

            # Load Acoustic and graph models and other files
            self.transition_model, self.acoustic_model = NnetRecognizer.read_model(
                self.AM_PATH+"/final.mdl")
            graph = _fst.read_fst_kaldi(self.LM_PATH+"/HCLG.fst")
            self.decoder_graph = LatticeFasterOnlineDecoder(
                graph, decoder_opts)
            self.symbols = _fst.SymbolTable.read_text(
                self.LM_PATH+"/words.txt")
            self.info = WordBoundaryInfo.from_file(
                WordBoundaryInfoNewOpts(), self.LM_PATH+"/word_boundary.int")

            
            self.asr = NnetLatticeFasterOnlineRecognizer(self.transition_model, self.acoustic_model, self.decoder_graph,
                                                    self.symbols, decodable_opts=self.decodable_opts, endpoint_opts=self.endpoint_opts)
            del graph, decoder_opts
        except Exception as e:
            self.log.error(e)
            raise ValueError(
                "AM and LM loading failed!!! (see logs for more details)")

    def get_sample_rate(self):
        return self.samp_freq

    def get_frames(self, feat_pipeline):
        rows = feat_pipeline.num_frames_ready()
        cols = feat_pipeline.dim()
        frames = Matrix(rows, cols)
        feat_pipeline.get_frames(range(rows), frames)
        return frames[:, :self.feat_info.mfcc_opts.num_ceps], frames[:, self.feat_info.mfcc_opts.num_ceps:]
        # return feats + ivectors

    def compute_feat(self, wav):
        try:
            feat_pipeline = OnlineNnetFeaturePipeline(self.feat_info)
            feat_pipeline.accept_waveform(self.samp_freq, wav)
            feat_pipeline.input_finished()
        except Exception as e:
            self.log.error(e)
            raise ValueError("Feature extraction failed!!!")
        else:
            return feat_pipeline

    def decoder(self, feats):
        try:
            start_time = time.time()
            self.log.info("Start Decoding: %s" % (start_time))
            self.asr.set_input_pipeline(feats)
            decode = self.asr.decode()
            self.log.info("Decode time in seconds: %s" %
                          (time.time() - start_time))
        except Exception as e:
            self.log.error(e)
            raise ValueError("Decoder failed to transcribe the input audio!!!")
        else:
            return decode

    def wordTimestamp(self, text, lattice, frame_shift, frame_subsampling):
        try:
            _fst.utils.scale_compact_lattice(
                [[1.0, 0], [0, float(self.acwt)]], lattice)
            bestPath = compact_lattice_shortest_path(lattice)
            _fst.utils.scale_compact_lattice(
                [[1.0, 0], [0, 1.0/float(self.acwt)]], bestPath)
            bestLattice = word_align_lattice(
                bestPath, self.transition_model, self.info, 0)
            alignment = compact_lattice_to_word_alignment(bestLattice[1])
            words = _fst.indices_to_symbols(self.symbols, alignment[0])
            start = alignment[1]
            dur   = alignment[2]

            output = {}
            output["words"] = []
            for i in range(len(words)):
                meta = {}
                meta["word"] = words[i]
                meta["start"] = round(start[i] * frame_shift * frame_subsampling, 2)
                meta["end"] = round((start[i]+dur[i]) * frame_shift * frame_subsampling, 2)
                output["words"].append(meta)
                text += " "+meta["word"]
            output["text"] = text

        except Exception as e:
            self.log.error(e)
            raise ValueError("Decoder failed to create the word timestamps!!!")
        else:
            return output


class SpeakerDiarization:
    def __init__(self, sample_rate):
        self.log = logging.getLogger(
            '__stt-standelone-worker__.SPKDiarization')

        # MFCC FEATURES PARAMETERS
        self.sr = sample_rate
        self.frame_length_s = 0.025
        self.frame_shift_s = 0.01
        self.num_bins = 40
        self.num_ceps = 40
        self.low_freq = 40
        self.high_freq = -200
        if self.sr == 16000:
            self.low_freq = 20
            self.high_freq = 7600
        #####

        # VAD PARAMETERS
        self.vad_ops = VadEnergyOptions()
        self.vad_ops.vad_energy_mean_scale = 0.9
        self.vad_ops.vad_energy_threshold = 5
        #vad_ops.vad_frames_context = 2
        #vad_ops.vad_proportion_threshold = 0.12
        #####

        # Segment
        self.seg_length = 100  # Window size in frames
        self.seg_increment = 100  # Window increment after and before window in frames
        self.seg_rate = 100  # Window shifting in frames
        #####

        # KBM
        # Minimum number of Gaussians in the initial pool
        self.minimumNumberOfInitialGaussians = 1024
        self.maximumKBMWindowRate = 50  # Maximum window rate for Gaussian computation
        self.windowLength = 200  # Window length for computing Gaussians
        self.kbmSize = 320  # Number of final Gaussian components in the KBM
        # If set to 1, the KBM size is set as a proportion, given by "relKBMsize", of the pool size
        self.useRelativeKBMsize = 1
        # Relative KBM size if "useRelativeKBMsize = 1" (value between 0 and 1).
        self.relKBMsize = 0.3
        ######

        # BINARY_KEY
        self.topGaussiansPerFrame = 5  # Number of top selected components per frame
        self.bitsPerSegmentFactor = 0.2  # Percentage of bits set to 1 in the binary keys
        ######

        # CLUSTERING
        self.N_init = 16  # Number of initial clusters
        # Set to one to perform linkage clustering instead of clustering/reassignment
        self.linkage = 0
        # Linkage criterion used if linkage==1 ('average', 'single', 'complete')
        self.linkageCriterion = 'average'
        # Similarity metric: 'cosine' for cumulative vectors, and 'jaccard' for binary keys
        self.metric = 'cosine'
        ######

        # CLUSTERING_SELECTION
        # Distance metric used in the selection of the output clustering solution ('jaccard','cosine')
        self.metric_clusteringSelection = 'cosine'
        # Method employed for number of clusters selection. Can be either 'elbow' for an elbow criterion based on within-class sum of squares (WCSS) or 'spectral' for spectral clustering
        self.bestClusteringCriterion = 'elbow'
        self.sigma = 1  # Spectral clustering parameters, employed if bestClusteringCriterion == spectral
        self.percentile = 40
        self.maxNrSpeakers = 10  # If known, max nr of speakers in a sesssion in the database. This is to limit the effect of changes in very small meaningless eigenvalues values generating huge eigengaps
        ######

        # RESEGMENTATION
        self.resegmentation = 1  # Set to 1 to perform re-segmentation
        self.modelSize = 6  # Number of GMM components
        self.nbIter = 10  # Number of expectation-maximization (EM) iterations
        self.smoothWin = 100  # Size of the likelihood smoothing window in nb of frames
        ######

    def compute_feat_KALDI(self, wav):
        try:
            po = ParseOptions("")
            mfcc_opts = MfccOptions()
            mfcc_opts.use_energy = False
            mfcc_opts.frame_opts.samp_freq = self.sr
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
            mfccKaldi = mfccObj.compute_features(wav, self.sr, 1.0)
        except Exception as e:
            self.log.error(e)
            raise ValueError(
                "Speaker diarization failed while extracting features!!!")
        else:
            return mfccKaldi

    def computeVAD_KALDI(self, feats):
        try:
            vadStream = compute_vad_energy(self.vad_ops, feats)
            vad = Vector(vadStream)
            VAD = vad.numpy()

            #  segmentation
            occurence = []
            value = []
            occurence.append(1)
            value.append(VAD[0])

            # compute the speech and non-speech frames
            for i in range(1, len(VAD)):
                if value[-1] == VAD[i]:
                    occurence[-1] += 1
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
                    i += 1

            # split if and only if the silence is above 50 frames
            i = 0
            while(i < len(occurence)):
                if i != 0 and ((occurence[i] < 30 and value[i] == 0.0) or value[i-1] == value[i]):
                    occurence[i-1] += occurence[i]
                    del value[i]
                    del occurence[i]
                else:
                    i += 1

            # compute VAD mask
            maskSAD = np.zeros(len(VAD))
            start = 0
            for i in range(len(occurence)):
                if value[i] == 1.0:
                    end = start+occurence[i]
                    maskSAD[start:end] = 1
                    start = end
                else:
                    start += occurence[i]

            maskSAD = np.expand_dims(maskSAD, axis=0)
        except ValueError as v:
            self.log.error(v)
        except Exception as e:
            self.log.error(e)
            raise ValueError(
                "Speaker diarization failed while voice activity detection!!!")
        else:
            return maskSAD

    def run(self, wav, dur, feats=None):
        try:
            def getSegments(frameshift, finalSegmentTable, finalClusteringTable, dur):
                numberOfSpeechFeatures = finalSegmentTable[-1, 2].astype(int)+1
                solutionVector = np.zeros([1, numberOfSpeechFeatures])
                for i in np.arange(np.size(finalSegmentTable, 0)):
                    solutionVector[0, np.arange(
                        finalSegmentTable[i, 1], finalSegmentTable[i, 2]+1).astype(int)] = finalClusteringTable[i]
                seg = np.empty([0, 3])
                solutionDiff = np.diff(solutionVector)[0]
                first = 0
                for i in np.arange(0, np.size(solutionDiff, 0)):
                    if solutionDiff[i]:
                        last = i+1
                        seg1 = (first)*frameshift
                        seg2 = (last-first)*frameshift
                        seg3 = solutionVector[0, last-1]
                        if seg.shape[0] != 0 and seg3 == seg[-1][2]:
                            seg[-1][1] += seg2
                        elif seg3 and seg2 > 0.3:  # and seg2 > 0.1
                            seg = np.vstack((seg, [seg1, seg2, seg3]))
                        first = i+1
                last = np.size(solutionVector, 1)
                seg1 = (first-1)*frameshift
                seg2 = (last-first+1)*frameshift
                seg3 = solutionVector[0, last-1]
                if seg3 == seg[-1][2]:
                    seg[-1][1] += seg2
                elif seg3 and seg2 > 0.3:  # and seg2 > 0.1
                    seg = np.vstack((seg, [seg1, seg2, seg3]))
                seg = np.vstack((seg, [dur, -1, -1]))
                seg[0][0] = 0.0
                return seg


            start_time = time.time()
            self.log.info("Start Speaker Diarization: %s" % (start_time))
            if self.maxNrSpeakers == 1 or dur < 5:
                self.log.info("Speaker Diarization time in seconds: %s" %
                              (time.time() - start_time))
                return [[0, dur, 1],
                        [dur, -1, -1]]
            if feats == None:
                feats = self.compute_feat_KALDI(wav)
            nFeatures = feats.shape[0]
            maskSAD = self.computeVAD_KALDI(feats)
            maskUEM = np.ones([1, nFeatures])

            mask = np.logical_and(maskUEM, maskSAD)
            mask = mask[0][0:nFeatures]
            nSpeechFeatures = np.sum(mask)
            speechMapping = np.zeros(nFeatures)
            # you need to start the mapping from 1 and end it in the actual number of features independently of the indexing style
            # so that we don't lose features on the way
            speechMapping[np.nonzero(mask)] = np.arange(1, nSpeechFeatures+1)
            data = feats[np.where(mask == 1)]
            del feats

            segmentTable = getSegmentTable(
                mask, speechMapping, self.seg_length, self.seg_increment, self.seg_rate)
            numberOfSegments = np.size(segmentTable, 0)
            # create the KBM
            # set the window rate in order to obtain "minimumNumberOfInitialGaussians" gaussians
            if np.floor((nSpeechFeatures-self.windowLength)/self.minimumNumberOfInitialGaussians) < self.maximumKBMWindowRate:
                windowRate = int(np.floor(
                    (np.size(data, 0)-self.windowLength)/self.minimumNumberOfInitialGaussians))
            else:
                windowRate = int(self.maximumKBMWindowRate)

            if windowRate == 0:
                raise ValueError(
                    'The audio is to short in order to perform the speaker diarization!!!')

            poolSize = np.floor((nSpeechFeatures-self.windowLength)/windowRate)
            if self.useRelativeKBMsize:
                kbmSize = int(np.floor(poolSize*self.relKBMsize))
            else:
                kbmSize = int(self.kbmSize)

            # Training pool of',int(poolSize),'gaussians with a rate of',int(windowRate),'frames'
            kbm, gmPool = trainKBM(
                data, self.windowLength, windowRate, kbmSize)

            #'Selected',kbmSize,'gaussians from the pool'
            Vg = getVgMatrix(data, gmPool, kbm, self.topGaussiansPerFrame)

            #'Computing binary keys for all segments... '
            segmentBKTable, segmentCVTable = getSegmentBKs(
                segmentTable, kbmSize, Vg, self.bitsPerSegmentFactor, speechMapping)

            #'Performing initial clustering... '
            initialClustering = np.digitize(np.arange(numberOfSegments), np.arange(
                0, numberOfSegments, numberOfSegments/self.N_init))

            #'Performing agglomerative clustering... '
            if self.linkage:
                finalClusteringTable, k = performClusteringLinkage(
                    segmentBKTable, segmentCVTable, self.N_init, self.linkageCriterion, self.metric)
            else:
                finalClusteringTable, k = performClustering(
                    speechMapping, segmentTable, segmentBKTable, segmentCVTable, Vg, self.bitsPerSegmentFactor, kbmSize, self.N_init, initialClustering, self.metric)

            #'Selecting best clustering...'
            if self.bestClusteringCriterion == 'elbow':
                bestClusteringID = getBestClustering(
                    self.metric_clusteringSelection, segmentBKTable, segmentCVTable, finalClusteringTable, k, self.maxNrSpeakers)
            elif self.bestClusteringCriterion == 'spectral':
                bestClusteringID = getSpectralClustering(self.metric_clusteringSelection, finalClusteringTable,
                                                         self.N_init, segmentBKTable, segmentCVTable, k, self.sigma, self.percentile, self.maxNrSpeakers)+1

            if self.resegmentation and np.size(np.unique(finalClusteringTable[:, bestClusteringID.astype(int)-1]), 0) > 1:
                finalClusteringTableResegmentation, finalSegmentTable = performResegmentation(data, speechMapping, mask, finalClusteringTable[:, bestClusteringID.astype(
                    int)-1], segmentTable, self.modelSize, self.nbIter, self.smoothWin, nSpeechFeatures)
                seg = getSegments(self.frame_shift_s, finalSegmentTable, np.squeeze(finalClusteringTableResegmentation), dur)
            else:
                seg = getSegmentationFile(
                    self.frame_shift_s, segmentTable, finalClusteringTable[:, bestClusteringID.astype(int)-1])
            self.log.info("Speaker Diarization time in seconds: %s" %
                          (time.time() - start_time))
        except ValueError as v:
            self.log.info(v)
            return [[0, dur, 1],
                    [dur, -1, -1]]
        except Exception as e:
            self.log.error(e)
            raise ValueError("Speaker Diarization failed!!!")
        else:
            return seg


class SttStandelone:
    def __init__(self):
        self.log = logging.getLogger("__stt-standelone-worker-streaming__")
        logging.basicConfig(level=logging.INFO)

        # Main parameters
        self.AM_PATH = '/opt/models/AM'
        self.LM_PATH = '/opt/models/LM'
        self.TEMP_FILE_PATH = '/opt/tmp'
        self.CONFIG_FILES_PATH = '/opt/config'
        self.SAVE_AUDIO = False
        self.SERVICE_PORT = 80
        self.SWAGGER_URL = '/api-doc'
        self.SWAGGER_PATH = None

        if not os.path.isdir(self.TEMP_FILE_PATH):
            os.mkdir(self.TEMP_FILE_PATH)
        if not os.path.isdir(self.CONFIG_FILES_PATH):
            os.mkdir(self.CONFIG_FILES_PATH)

        # Environment parameters
        if 'SERVICE_PORT' in os.environ:
            self.SERVICE_PORT = os.environ['SERVICE_PORT']
        if 'SAVE_AUDIO' in os.environ:
            self.SAVE_AUDIO = os.environ['SAVE_AUDIO']
        if 'SWAGGER_PATH' in os.environ:
            self.SWAGGER_PATH = os.environ['SWAGGER_PATH']

        self.loadConfig()

    def loadConfig(self):
        # get decoder parameters from "decode.cfg"
        decoder_settings = configparser.ConfigParser()
        if not os.path.exists(self.AM_PATH+'/decode.cfg'):
            return False
        decoder_settings.read(self.AM_PATH+'/decode.cfg')

        # Prepare "online.conf"
        self.AM_PATH = self.AM_PATH+"/" + \
            decoder_settings.get('decoder_params', 'ampath')
        with open(self.AM_PATH+"/conf/online.conf") as f:
            values = f.readlines()
            with open(self.CONFIG_FILES_PATH+"/online.conf", 'w') as f:
                for i in values:
                    f.write(i)
                f.write("--ivector-extraction-config=" +
                        self.CONFIG_FILES_PATH+"/ivector_extractor.conf\n")
                f.write("--mfcc-config="+self.AM_PATH+"/conf/mfcc.conf\n")
                f.write(
                    "--beam="+decoder_settings.get('decoder_params', 'beam')+"\n")
                f.write(
                    "--lattice-beam="+decoder_settings.get('decoder_params', 'lattice_beam')+"\n")
                f.write("--acoustic-scale=" +
                        decoder_settings.get('decoder_params', 'acwt')+"\n")
                f.write(
                    "--min-active="+decoder_settings.get('decoder_params', 'min_active')+"\n")
                f.write(
                    "--max-active="+decoder_settings.get('decoder_params', 'max_active')+"\n")
                f.write("--frame-subsampling-factor="+decoder_settings.get(
                    'decoder_params', 'frame_subsampling_factor')+"\n")

        # Prepare "ivector_extractor.conf"
        with open(self.AM_PATH+"/conf/ivector_extractor.conf") as f:
            values = f.readlines()
            with open(self.CONFIG_FILES_PATH+"/ivector_extractor.conf", 'w') as f:
                for i in values:
                    f.write(i)
                f.write("--splice-config="+self.AM_PATH+"/conf/splice.conf\n")
                f.write("--cmvn-config="+self.AM_PATH +
                        "/conf/online_cmvn.conf\n")
                f.write("--lda-matrix="+self.AM_PATH +
                        "/ivector_extractor/final.mat\n")
                f.write("--global-cmvn-stats="+self.AM_PATH +
                        "/ivector_extractor/global_cmvn.stats\n")
                f.write("--diag-ubm="+self.AM_PATH +
                        "/ivector_extractor/final.dubm\n")
                f.write("--ivector-extractor="+self.AM_PATH +
                        "/ivector_extractor/final.ie")

        # Prepare "word_boundary.int" if not exist
        if not os.path.exists(self.LM_PATH+"/word_boundary.int") and os.path.exists(self.AM_PATH+"phones.txt"):
            with open(self.AM_PATH+"phones.txt") as f:
                phones = f.readlines()

            with open(self.LM_PATH+"/word_boundary.int", "w") as f:
                for phone in phones:
                    phone = phone.strip()
                    phone = re.sub('^<eps> .*', '', phone)
                    phone = re.sub('^#\d+ .*', '', phone)
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

    def swaggerUI(self, app):
        ### swagger specific ###
        swagger_yml = yaml.load(
            open(self.SWAGGER_PATH, 'r'), Loader=yaml.Loader)
        swaggerui = get_swaggerui_blueprint(
            # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
            self.SWAGGER_URL,
            self.SWAGGER_PATH,
            config={  # Swagger UI config overrides
                'app_name': "STT API Documentation",
                'spec': swagger_yml
            }
        )
        app.register_blueprint(swaggerui, url_prefix=self.SWAGGER_URL)
        ### end swagger specific ###

    def read_audio(self, file, sample_rate):
        file_path = self.TEMP_FILE_PATH+file.filename.lower()
        file.save(file_path)
        try:
            data, sr = librosa.load(file_path, sr=None)
            if sr != sample_rate:
                self.log.info('Resample audio file: '+str(sr) +
                              'Hz -> '+str(sample_rate)+'Hz')
                data = librosa.resample(data, sr, sample_rate)
            data = (data * 32767).astype(np.int16)
            self.dur = len(data) / sample_rate
            self.data = Vector(data)

            if not self.SAVE_AUDIO:
                os.remove(file_path)
        except Exception as e:
            self.log.error(e)
            raise ValueError("The uploaded file format is not supported!!!")

    def run(self, asr, metadata):
        feats = asr.compute_feat(self.data)
        mfcc, ivector = asr.get_frames(feats)
        decode = asr.decoder(feats)
        if metadata:
            spk = SpeakerDiarization(asr.get_sample_rate())
            spkSeg = spk.run(self.data, self.dur, mfcc)
            data = asr.wordTimestamp(decode["text"], decode['lattice'], asr.frame_shift, asr.decodable_opts.frame_subsampling_factor)
            output = self.process_output(data, spkSeg)
            return output
        else:
            return self.parse_text(decode["text"])


    # return a json object including word-data, speaker-data
    def process_output(self, data, spkrs):
        speakers = []
        text = []
        i = 0
        text_ = ""
        words=[]
        for word in data['words']:
            if i+1 < len(spkrs) and word["end"] < spkrs[i+1][0]:
                text_ += word["word"]  + " "
                words.append(word)
            else:
                speaker = {}
                speaker["start"]=words[0]["start"]
                speaker["end"]=words[len(words)-1]["end"]
                speaker["speaker_id"]='spk'+str(int(spkrs[i][2]))
                speaker["words"]=words

                text.append('spk'+str(int(spkrs[i][2]))+' : '+ self.parse_text(text_))
                speakers.append(speaker)

                words=[word]
                text_=word["word"] + " "
                i+=1

        speaker = {}
        speaker["start"]=words[0]["start"]
        speaker["end"]=words[len(words)-1]["end"]
        speaker["speaker_id"]='spk'+str(int(spkrs[i][2]))
        speaker["words"]=words

        text.append('spk'+str(int(spkrs[i][2]))+' : '+ self.parse_text(text_))
        speakers.append(speaker)

        return {'speakers': speakers, 'text': text}

    # remove extra symbols
    def parse_text(self, text):
        text = re.sub(r"<unk>", "", text) # remove <unk> symbol
        text = re.sub(r"#nonterm:[^ ]* ", "", text) # remove entity's mark
        text = re.sub(r"<eps>", "", text) # remove <eps>
        text = re.sub(r"' ", "'", text) # remove space after quote '
        text = re.sub(r" +", " ", text) # remove multiple spaces
        text = text.strip()
        return text
