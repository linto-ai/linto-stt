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

## other packages
import configparser, sys, sox, time, logging
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
        feat_pipeline = OnlineNnetFeaturePipeline(self.feat_info)
        feat_pipeline.accept_waveform(audio.sr, audio.getDataKaldyVector())
        feat_pipeline.input_finished()
        return feat_pipeline
        
    def decoder(self,feats):
        try:
            start_time = time.time()
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

class SttStandelone:
    def __init__(self,asr,metadata=False):
        self.log = logging.getLogger('__stt-standelone-worker__.SttStandelone')
        self.metadata = metadata

    def run(self,audio,asr):
        feats = asr.compute_feat(audio)
        decode = asr.decoder(feats)
        if self.metadata:
            timestamps = asr.wordTimestamp(decode)
            self.formatOutput(timestamps,asr.frame_shift, asr.decodable_opts.frame_subsampling_factor)
            return self.output
        else:
            return decode["text"]

    def formatOutput(self,timestamps,frame_shift, frame_subsampling):
        self.output = {}
        text = ""
        self.output["words"] = []
        for i in range(len(timestamps["words"])):
            if timestamps["words"][i] != "<eps>":
                meta = {}
                meta["word"] = timestamps["words"][i]
                meta["begin"] = round(timestamps["start"][i] * frame_shift * frame_subsampling,2)
                meta["end"] = round((timestamps["start"][i]+timestamps["dur"][i]) * frame_shift * frame_subsampling, 2)
                self.output["words"].append(meta)
                text += " "+meta["word"]
        self.output["transcription"] = text
        
        
class Audio:
    def __init__(self):
        self.log = logging.getLogger('__stt-standelone-worker__.Audio')
        self.bit = 16
        self.channels = 1
        self.sr = -1
    
    def set_sample_rate(self,sr):
        self.sr = sr
    
    def set_logger(self,log):
        self.log = log
    
    def transform(self,file_name):
        try:
            tfm = sox.Transformer()
            tfm.set_output_format(rate=self.sr,
                                  bits=self.bit,
                                  channels=self.channels)
            self.data = tfm.build_array(input_filepath=file_name)
        except Exception as e:
            self.log.error(e)
            raise ValueError("The uploaded file format is not supported!!!")
    
    def getDataKaldyVector(self):
        return Vector(self.data)