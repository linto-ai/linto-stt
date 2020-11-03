#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  ASR
from vosk import Model, KaldiRecognizer
##############

# Speaker Diarization
from pyBK.diarizationFunctions import *
import librosa
import time
import webrtcvad
##############

# other packages
import configparser
import librosa
import logging
import os
import re
import uuid
import json
import yaml
import numpy as np
from scipy.io import wavfile
from flask_swagger_ui import get_swaggerui_blueprint
##############


class WorkerStreaming:
    def __init__(self):
        # Set logger config
        self.log = logging.getLogger("__stt-standelone-worker-streaming__")
        logging.basicConfig(level=logging.INFO)

        # Main parameters
        self.AM_PATH = '/opt/models/AM'
        self.LM_PATH = '/opt/models/LM'
        self.TEMP_FILE_PATH = '/opt/tmp'
        self.CONFIG_FILES_PATH = '/opt/config'
        self.SAVE_AUDIO = False
        self.SERVICE_PORT = 80
        self.NBR_THREADS = 100
        self.SWAGGER_URL = '/api-doc'
        self.SWAGGER_PATH = ''
        self.ONLINE = False

        if not os.path.isdir(self.CONFIG_FILES_PATH):
            os.mkdir(self.CONFIG_FILES_PATH)

        if not os.path.isdir(self.TEMP_FILE_PATH):
            os.mkdir(self.TEMP_FILE_PATH)

        # Environment parameters
        if 'NBR_THREADS' in os.environ:
            if int(os.environ['NBR_THREADS']) > 0:
                self.NBR_THREADS = int(os.environ['NBR_THREADS'])
            else:
                self.log.warning(
                    "You must to provide a positif number of threads 'NBR_THREADS'")
        if 'SWAGGER_PATH' in os.environ:
            self.SWAGGER_PATH = os.environ['SWAGGER_PATH']

        # start loading ASR configuration
        self.log.info("Create the new config files")
        self.loadConfig()

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

    def getAudio(self, file):
        filename = str(uuid.uuid4())
        self.file_path = self.TEMP_FILE_PATH+"/"+filename
        file.save(self.file_path)
        try:
            self.rate, self.data = wavfile.read(self.file_path)
            # if stereo file, convert to mono by computing the mean of the channels
            if len(self.data.shape) == 2 and self.data.shape[1] == 2:
                self.data = np.mean(self.data, axis=1, dtype=np.int16)
        except Exception as e:
            self.log.error(e)
            raise ValueError("The uploaded file format is not supported!!!")

    def clean(self):
        if not self.SAVE_AUDIO:
            os.remove(self.file_path)

    # re-create config files
    def loadConfig(self):
        # load decoder parameters from "decode.cfg"
        decoder_settings = configparser.ConfigParser()
        if os.path.exists(self.AM_PATH+'/decode.cfg') == False:
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
        if not os.path.exists(self.LM_PATH+"/word_boundary.int") and os.path.exists(self.AM_PATH+"/phones.txt"):
            self.log.info("Create word_boundary.int based on phones.txt")
            with open(self.AM_PATH+"/phones.txt") as f:
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

    # remove extra symbols
    def parse_text(self, text):
        text = re.sub(r"<unk>", "", text)  # remove <unk> symbol
        text = re.sub(r"#nonterm:[^ ]* ", "", text)  # remove entity's mark
        text = re.sub(r"' ", "'", text)  # remove space after quote '
        text = re.sub(r" +", " ", text)  # remove multiple spaces
        text = text.strip()
        return text

    # Postprocess response
    def get_response(self, dataJson, is_metadata):
        if dataJson is not None:
            data = json.loads(dataJson)
            if not is_metadata:
                text = data['text']  # get text from response
                return self.parse_text(text)

            elif 'words' in data:
                # Do speaker diarization and get speaker segments
                spk = SpeakerDiarization()
                spkrs = spk.run(self.file_path)

                # Generate final output data
                return self.process_output(data, spkrs)
            elif 'text' in data:
                return {'speakers': [], 'text': data['text'], 'words': []}
            else:
                return {'speakers': [], 'text': '', 'words': []}
        else:
            return {'speakers': [], 'text': '', 'words': []}

    # return a json object including word-data, speaker-data

    def process_output(self, data, spkrs):
        speakers = []
        text = []
        i = 0
        text_ = ""
        words = []
        for word in data['words']:
            if i+1 == len(spkrs):
                continue
            if i+1 < len(spkrs) and word["end"] < spkrs[i+1][0]:
                text_ += word["word"] + " "
                words.append(word)
            else:
                speaker = {}
                speaker["start"] = words[0]["start"]
                speaker["end"] = words[len(words)-1]["end"]
                speaker["speaker_id"] = 'spk'+str(int(spkrs[i][2]))
                speaker["words"] = words

                text.append(
                    'spk'+str(int(spkrs[i][2]))+' : ' + self.parse_text(text_))
                speakers.append(speaker)

                words = [word]
                text_ = word["word"] + " "
                i += 1

        speaker = {}
        speaker["start"] = words[0]["start"]
        speaker["end"] = words[len(words)-1]["end"]
        speaker["speaker_id"] = 'spk'+str(int(spkrs[i][2]))
        speaker["words"] = words

        text.append('spk'+str(int(spkrs[i][2]))+' : ' + self.parse_text(text_))
        speakers.append(speaker)

        return {'speakers': speakers, 'text': text}


class SpeakerDiarization:
    def __init__(self):
        self.log = logging.getLogger(
            '__stt-standelone-worker__.SPKDiarization')

       # MFCC FEATURES PARAMETERS
        self.frame_length_s = 0.025
        self.frame_shift_s = 0.01
        self.num_bins = 30
        self.num_ceps = 30
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
 
    def compute_feat_Librosa(self,audioFile):
        try:
            self.data, self.sr = librosa.load(audioFile,sr=None)
            frame_length_inSample = self.frame_length_s * self.sr
            hop = int(self.frame_shift_s * self.sr)
            NFFT = int(2**np.ceil(np.log2(frame_length_inSample)))
            if self.sr >= 16000:
                mfccNumpy = librosa.feature.mfcc(y=self.data,
                                                sr=self.sr,
                                                dct_type=2,
                                                n_mfcc=self.num_ceps,
                                                n_mels=self.num_bins,
                                                n_fft=NFFT,
                                                hop_length=hop,
                                                fmin=20,
                                                fmax=7600).T
            else:
                mfccNumpy = librosa.feature.mfcc(y=self.data,
                                                sr=self.sr,
                                                dct_type=2,
                                                n_mfcc=self.num_ceps,
                                                n_mels=self.num_bins,
                                                n_fft=NFFT,
                                                hop_length=hop).T

        except Exception as e:
            self.log.error(e)
            raise ValueError("Speaker diarization failed when extracting features!!!")
        else:
            return mfccNumpy

    def computeVAD_WEBRTC(self, data, sr, nFeatures):
        try:
            va_framed = py_webrtcvad(data, fs=sr, fs_vad=sr, hoplength=30, vad_mode=0)
            segments = get_py_webrtcvad_segments(va_framed,sr)
            maskSAD = np.zeros([1,nFeatures])
            for seg in segments:
                start=int(np.round(seg[0]/self.frame_shift_s))
                end=int(np.round(seg[1]/self.frame_shift_s))
                maskSAD[0][start:end]=1
        except Exception as e:
            self.log.error(e)
            raise ValueError("Speaker diarization failed while voice activity detection!!!")
        else:
            return maskSAD

    def run(self, audioFile):
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

            self.log.info('Start Speaker diarization')

            feats = self.compute_feat_Librosa(audioFile)
            nFeatures = feats.shape[0]
            duration = nFeatures * self.frame_shift_s

            if duration < 5:
                return [[0, duration, 1],
                        [duration, -1, -1]]

            maskSAD = self.computeVAD_WEBRTC(self.data, self.sr, nFeatures)
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
                #self.log.info('The audio is to short in order to perform the speaker diarization!!!')
                return [[0, duration, 1],
                        [duration, -1, -1]]

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
                seg = getSegments(self.frame_shift_s, finalSegmentTable, np.squeeze(
                    finalClusteringTableResegmentation), duration)
            else:
                return [[0, duration, 1],
                    [duration, -1, -1]]

            self.log.info("Speaker Diarization time in seconds: %d" %
                          int(time.time() - start_time))
        except ValueError as v:
            self.log.error(v)
            return [[0, duration, 1],
                    [duration, -1, -1]]
        except Exception as e:
            self.log.error(e)
            return [[0, duration, 1],
                    [duration, -1, -1]]
        else:
            return seg
