#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from flask import Flask, request, abort, Response, json
from flask_cors import CORS
from os import path
import uuid, os
import configparser
import subprocess
import shlex
import re

app = Flask(__name__)
CORS(app)

global busy
busy=0

AM_PATH = '/opt/models/AM'
LM_PATH = '/opt/models/LM'
TEMP_FILE_PATH = '/opt/tmp'  #/opt/wavs
TEMP_FILE_PATH1= '/opt/models'

#json output packages
import uuid #!!need way to generate unique ids
import simplejson as json

# Speaker diarization packages and parameters
import numpy as np
import uisrnn
import librosa
import sys
sys.path.append('ghostvlad')
import toolkits
import model as spkModel
import argparse
from speakerDiarization import append2dict, arrangeResult, genMap, load_wav, lin_spectogram_from_wav, load_data

parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default=r'pretrained/weights.h5', type=str)
parser.add_argument('--data_path', default='', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)
SAVED_MODEL_NAME = 'pretrained/saved_model.uisrnn_benchmark'

global args
args = parser.parse_args()
toolkits.initialize_GPU(args)



def dockerId():
    with open('/proc/self/cgroup') as f:
        lines = f.readlines() 
    for l in lines:
        if '/docker/' in l:
            return l.split('/')[2][:20]

def run_shell_command(command_line):
    try:
        command_line_args = shlex.split(command_line)
        process = subprocess.Popen(command_line_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, error = process.communicate()
        return False, output
    except OSError as err:
        print("OS error: {0}".format(err))
        return True, ''
    except ValueError:
        print("data error.")
        return True, ''
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return True, ''


def diarization(wav_path, embedding_per_second=1.0, overlap_rate=0.5):
    # Load speaker diarization model
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }
    network_eval = spkModel.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)
    network_eval.load_weights(args.resume, by_name=True)
    model_args, _, inference_args = uisrnn.parse_arguments()
    model_args.observation_dim = 512
    uisrnnModel = uisrnn.UISRNN(model_args)
    uisrnnModel.load(SAVED_MODEL_NAME)
    specs, intervals = load_data(wav_path, embedding_per_second=embedding_per_second, overlap_rate=overlap_rate)
    if len(specs) == 0:
        return {}, 0
    mapTable, keys = genMap(intervals)
    feats = []
    for spec in specs:
        spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        v = network_eval.predict(spec)
        feats += [v]
    feats = np.array(feats)[:,0,:].astype(float)  # [splits, embedding dim]
    predicted_label = uisrnnModel.predict(feats, inference_args)
    time_spec_rate = 1000*(1.0/embedding_per_second)*(1.0-overlap_rate) # speaker embedding every ?ms
    center_duration = int(1000*(1.0/embedding_per_second)//2)
    speakerSlice = arrangeResult(predicted_label, time_spec_rate)
    for spk,timeDicts in speakerSlice.items():    # time map to orgin wav(contains mute)
        for tid,timeDict in enumerate(timeDicts):
            s = 0
            e = 0
            for i,key in enumerate(keys):
                if(s!=0 and e!=0):
                    break
                if(s==0 and key>timeDict['start']):
                    offset = timeDict['start'] - keys[i-1]
                    s = mapTable[keys[i-1]] + offset
                if(e==0 and key>timeDict['stop']):
                    offset = timeDict['stop'] - keys[i-1]
                    e = mapTable[keys[i-1]] + offset
            speakerSlice[spk][tid]['start'] = s
            speakerSlice[spk][tid]['stop'] = e
    speakers = []
    nbSpk = len(speakerSlice)
    for spk,timeDicts in speakerSlice.items():
        for timeDict in timeDicts:
            list = []
            list.append(spk)
            list.append(timeDict['start']/1000.0)
            list.append(timeDict['stop']/1000.0)
            speakers.append(list)
    speakers.sort(key = lambda speakers: speakers[1])
    if speakers[0][0] != 0:
        speakers.insert(0,[0,0.0,speakers[0][1]])
    for i in range(len(speakers)-1):
        speakers[i][2] = speakers[i+1][1]
    speakers[i+1][2] += 1000
    app.logger.info(speakers)
    return speakers, nbSpk

def decode(audio_file,wav_name,do_word_tStamp,do_speaker_diarization, do_cm_json):
    # Normalize audio file and convert it to wave format
    error, output = run_shell_command("sox "+audio_file+" -t wav -b 16 -r 16000 -c 1 "+audio_file+".wav")
    if not path.exists(audio_file+".wav"):
        app.logger.info(output)
        return False, 'Error during audio file conversion!!! Supported formats are wav, mp3, aiff, flac, and ogg.'


    decode_file  = audio_file+".wav"
    decode_conf  = TEMP_FILE_PATH1+"/online.conf"
    decode_mdl   = AM_PATH+"/"+AM_FILE_PATH+"/final.mdl"
    decode_graph = LM_PATH+"/HCLG.fst"
    decode_words = LM_PATH+"/words.txt"
    decode_words_boundary = LM_PATH+"/word_boundary.int"


    # Decode the audio file
    app.logger.info("Do speech decoding")
    if DECODER_SYS == 'dnn3':
        error, output = run_shell_command("kaldi-nnet3-latgen-faster --do-endpointing=false --frame-subsampling-factor="+DECODER_FSF+" --frames-per-chunk=20 --online=false --config="+decode_conf+" --minimize=false --min-active="+DECODER_MINACT+" --max-active="+DECODER_MAXACT+" --beam="+DECODER_BEAM+" --lattice-beam="+DECODER_LATBEAM+" --acoustic-scale="+DECODER_ACWT+" --word-symbol-table="+decode_words+" "+decode_mdl+" "+decode_graph+" \"ark:echo "+wav_name+" "+wav_name+"|\" \"scp:echo "+wav_name+" "+decode_file+"|\" ark:"+TEMP_FILE_PATH+"/"+wav_name+".lat")
    elif DECODER_SYS == 'dnn2' or DECODER_SYS == 'dnn':
        error, output = run_shell_command("kaldi-nnet2-latgen-faster --do-endpointing=false --online=false --config="+decode_conf+" --min-active="+DECODER_MINACT+" --max-active="+DECODER_MAXACT+" --beam="+DECODER_BEAM+" --lattice-beam="+DECODER_LATBEAM+" --acoustic-scale="+DECODER_ACWT+" --word-symbol-table="+decode_words+" "+decode_mdl+" "+decode_graph+" \"ark:echo "+wav_name+" "+wav_name+"|\" \"scp:echo "+wav_name+" "+decode_file+"|\" ark:"+TEMP_FILE_PATH+"/"+wav_name+".lat")
    else:
        return False, 'The "decoder" parameter of the acoustic model is not supported!!!'

    if not path.exists(TEMP_FILE_PATH+"/"+wav_name+".lat"):
        app.logger.info(output)
        return False, 'One or multiple parameters of the acoustic model are not correct!!!'


    # Normalize the obtained transcription
    hypothesis = re.findall('\n'+wav_name+'.*',output)
    transcription=re.sub(wav_name,'',hypothesis[0]).strip()
    transcription=re.sub(r"#nonterm:[^ ]* ", "", transcription)
    transcription=re.sub(r" <unk> ", " ", " "+transcription+" ")


    # Get the begin and end time stamp from the decoder output
    if do_speaker_diarization or do_cm_json or do_word_tStamp:
        app.logger.info("Do word time-stamp estimation")
        shift = int(DECODER_FSF) * float(DECODER_FSHIFT)
        error, output = run_shell_command("kaldi-lattice-1best --acoustic-scale="+DECODER_ACWT+" ark:"+TEMP_FILE_PATH+"/"+wav_name+".lat ark:"+TEMP_FILE_PATH+"/"+wav_name+".1best")
        error, output = run_shell_command("kaldi-lattice-align-words "+decode_words_boundary+" "+decode_mdl+" ark:"+TEMP_FILE_PATH+"/"+wav_name+".1best ark:"+TEMP_FILE_PATH+"/"+wav_name+".words") 
        error, output = run_shell_command("kaldi-nbest-to-ctm --frame-shift="+str(shift)+"  ark:"+TEMP_FILE_PATH+"/"+wav_name+".words "+TEMP_FILE_PATH+"/"+wav_name+".ctm")
        error, output = run_shell_command("int2sym.pl -f 5 "+decode_words+" "+TEMP_FILE_PATH+"/"+wav_name+".ctm")
        data = []
        if not error and output != "":
            words = output.split("\n")
            transcription = ""
            
            for word in words:
                _word = word.strip().split(' ')
                if len(_word) == 5:
                    meta = []
                    word = re.sub("<unk>","",_word[4])
                    word = re.sub("<unk>","",_word[4])
                    if word != "":
                        transcription += " "+word
                        meta.append(word)
                        meta.append(float(_word[2]))
                        meta.append(float(_word[2]) + float(_word[3]))
                        meta.append(float(_word[1]))
                        data.append(meta)
    transcription = [transcription.strip()]
    # Get speaker information
    if do_speaker_diarization:
        app.logger.info("Do speaker diarization")
        speakers, nbSpk = diarization(decode_file, 0.5, 0.1)
        if nbSpk > 1:

            trans = ""
            pos = 0
            transcription = []
            for meta in data:
                if meta[1] <= speakers[pos][2]:
                    trans += ' '+meta[0]
                else:
                    transcription.append('Speaker_'+str(speakers[pos][0])+' : '+trans.strip())
                    trans = meta[0]
                    pos += 1
            transcription.append('Speaker_'+str(speakers[pos][0])+' : '+trans.strip())
        else:
            transcription = "Speaker_0 :"
            for meta in data:
                transcription += ' '+meta[0]
            transcription.strip()
            transcription = [transcription]

    if do_cm_json:
        # KATE ADDITION return a json object that organizes word and speaker info in turns
        app.logger.info("Return json") #<--not sure what this does?? add option to input?

        speakers, nbSpk = diarization(decode_file, 0.5, 0.1)

        transcription = [] #this is the text list for the json object
        speaker_list = [] #this is the speaker id dict for the json object

        if nbSpk > 1:

            turn_id = 0
            turn_dict = {}
            turn_dict['words'] = []

            pos = 0

            for meta in data:
                word_dict = {}
                if meta[1] <= speakers[pos][2]:
                    word_dict['wid'] = uuid.uuid4().hex
                    word_dict['word'] = meta[0]
                    word_dict['stime'] = meta[1]
                    word_dict['etime'] = meta[2]
                    # word_dict['score'] = meta[3] --> ignore score for now
                    turn_dict['words'].append(word_dict)
                else:
                    #complete turn by adding speaker id and turn id
                    turn_dict['speaker_id'] = 'spk_'+str(speakers[pos][0])
                    turn_dict['turn_id'] = turn_id
                    #append to text list
                    transcription.append(turn_dict)
                    #increment turn id
                    turn_id += 1
                    #reset turn dict
                    turn_dict = {}
                    turn_dict['words'] = []
                    #incremement pos in speaker list
                    pos += 1
                    #insert current word into reset turn dict
                    word_dict['wid'] = uuid.uuid4().hex
                    word_dict['word'] = meta[0]
                    word_dict['stime'] = meta[1]
                    word_dict['etime'] = meta[2]
                    # word_dict['score'] = meta[3] --> ignore score for now
                    turn_dict['words'].append(word_dict)

            #create speaker list
            speaker_set = list(set([l[0] for l in speakers]))
            for s in speaker_set:
                speaker_dict = {}
                speaker_dict['speaker_id'] = 'spk_'+str(s)
                speaker_dict['speaker_name'] = 'speaker ' + str(s)

                times = [l[1:3] for l in speakers if l[0] == s][0] #get first example of speaker talking
                speaker_dict['stime'] = times[0]
                speaker_dict['etime'] = times[1]

                speaker_list.append(speaker_dict)
        else:

            turn_id = 0
            turn_dict = {}
            turn_dict['words'] = []

            for meta in data:
                word_dict = {}
                word_dict['wid'] = uuid.uuid4().hex
                word_dict['word'] = meta[0]
                word_dict['stime'] = meta[1]
                word_dict['etime'] = meta[2]
                # word_dict['score'] = meta[3] --> ignore score for now

                turn_dict['words'].append(word_dict)

            turn_dict['speaker_id'] = 'spk_0'
            turn_dict['turn_id'] = turn_id
            #append to text list
            transcription.append(turn_dict)

            #create speaker list
            speaker_dict = {}
            speaker_dict['speaker_id'] = 'spk_0'
            speaker_dict['speaker_name'] = 'speaker 0' 

            speaker_list.append(speaker_dict)

        #create final json
        """
        json format: 
        
        {
            'speakers' : [
                {<speaker info>}, 
                {}...
                ], 
            'text' : [
                {
                    turnid,
                    speakerid
                    words : [
                        {
                            word 
                            word id
                            word stime
                            word etime
                        }
                    ]
                }, {} ...
            ]
        }
        """
        json_dict = {}
        json_dict['speakers'] = speaker_list
        json_dict['text'] = transcription  
        #convert to json object --> how to output this????
        final_json = json.dumps(json_dict)

# transcription = [transcription]
    # Get the begin and end time stamp from the decoder output
    if do_word_tStamp:
        if len(data) != 0:
            output = {}
            output["words"] = []
            for d in data:
                meta = {}
                meta["word"] = d[0]
                meta["stime"] = d[1]
                meta["etime"] = d[2]
                meta["score"] = d[3]
                output["words"].append(meta)
            output["transcription"] = transcription
            
            
#        app.logger.info("Do word time-stamp estimation")
#        shift = int(DECODER_FSF) * float(DECODER_FSHIFT)
#        error, output = run_shell_command("kaldi-lattice-1best --acoustic-scale="+DECODER_ACWT+" ark:"+TEMP_FILE_PATH+"/"+wav_name+".lat ark:"+TEMP_FILE_PATH+"/"+wav_name+".1best")
#        error, output = run_shell_command("kaldi-lattice-align-words "+decode_words_boundary+" "+decode_mdl+" ark:"+TEMP_FILE_PATH+"/"+wav_name+".1best ark:"+TEMP_FILE_PATH+"/"+wav_name+".words") 
#        error, output = run_shell_command("kaldi-nbest-to-ctm --frame-shift="+str(shift)+"  ark:"+TEMP_FILE_PATH+"/"+wav_name+".words "+TEMP_FILE_PATH+"/"+wav_name+".ctm")
#        error, output = run_shell_command("int2sym.pl -f 5 "+decode_words+" "+TEMP_FILE_PATH+"/"+wav_name+".ctm")
#        if not error and output != "":
#            words = output.split("\n")
#            trans = ""
#            data = {}
#            data["words"] = []
#            for word in words:
#                _word = word.strip().split(' ')
#                if len(_word) == 5:
#                    meta = {}
#                    word = re.sub("<unk>","",_word[4])
#                    word = re.sub("<unk>","",_word[4])
#                    if word != "":
#                        trans = trans+" "+word
#                        meta["word"] = word
#                        meta["stime"] = float(_word[2])
#                        meta["etime"] = (float(_word[2]) + float(_word[3]))
#                        meta["score"] = float(_word[1])
#                        data["words"].append(meta)
#            data["transcription"] = trans.strip()
            return True, output
        else:
            app.logger.info("error during word time stamp generation: verify the LM and the file word_boundary.int")

    return True, transcription

@app.route('/transcribe', methods=['POST'])
def transcribe():
    global busy
    busy=1
    fileid = str(uuid.uuid4())
    if request.headers.get('accept').lower() == 'application/json':
        metadata = True
    elif request.headers.get('accept').lower() == 'text/plain':
        metadata = False
    else:
        return 'Not accepted header', 400
    
    speaker = True if request.args.get('speaker').lower() == 'yes' else False
    
    if 'file' in request.files.keys():
        file = request.files['file']
        file_ext = file.filename.rsplit('.', 1)[-1].lower()
        file_type = file.content_type.rsplit('/', 1)[0]
        if file_type == "audio":
            filename = TEMP_FILE_PATH+'/'+fileid+'.'+file_ext
            file.save(filename)
            b, out = decode(filename,fileid,metadata,speaker)
            if not b:
                busy=0
                return 'Error while file transcription: '+out, 400
        else:
            busy=0
            return 'Error while file transcription: The uploaded file format is not supported!!! Supported formats are wav, mp3, aiff, flac, and ogg.', 400
    else:
        busy=0
        return 'No audio file was uploaded', 400

    # Delete temporary files
    for file in os.listdir(TEMP_FILE_PATH):
        os.remove(TEMP_FILE_PATH+"/"+file)
    busy=0
    if metadata:
        json_string = json.dumps(out, ensure_ascii=False)
        return Response(json_string,content_type="application/json; charset=utf-8" ), 200
    else:
        return Response(', '.join(out),content_type="text/plain; charset=utf-8" ), 200

@app.route('/check', methods=['GET'])
def check():
    return '1', 200

@app.route('/stop', methods=['POST'])
def stop():
    while(busy==1):
        continue
    subprocess.call("kill 1",shell=True)
    return '1', 200

if __name__ == '__main__':
    SERVICE_PORT = os.environ['SERVICE_PORT']

    #Decoder parameters applied for both GMM and DNN based ASR systems
    decoder_settings = configparser.ConfigParser()
    decoder_settings.read(AM_PATH+'/decode.cfg')
    DECODER_SYS = decoder_settings.get('decoder_params', 'decoder')
    AM_FILE_PATH = decoder_settings.get('decoder_params', 'ampath')
    DECODER_MINACT = decoder_settings.get('decoder_params', 'min_active')
    DECODER_MAXACT = decoder_settings.get('decoder_params', 'max_active')
    DECODER_BEAM = decoder_settings.get('decoder_params', 'beam')
    DECODER_LATBEAM = decoder_settings.get('decoder_params', 'lattice_beam')
    DECODER_ACWT = decoder_settings.get('decoder_params', 'acwt')
    DECODER_FSF = decoder_settings.get('decoder_params', 'frame_subsampling_factor')
    DECODER_FSHIFT = decoder_settings.get('decoder_params', 'frame_shift') if decoder_settings.has_option('decoder_params', 'frame_shift') else 0.01

    #Prepare config files
    AM_FINAL_PATH=AM_PATH+"/"+AM_FILE_PATH
    with open(AM_FINAL_PATH+"/conf/online.conf") as f:
        values = f.readlines()
        with open(TEMP_FILE_PATH1+"/online.conf", 'w') as f:
            for i in values:
                f.write(i)
            f.write("--ivector-extraction-config="+TEMP_FILE_PATH1+"/ivector_extractor.conf\n")
            f.write("--mfcc-config="+AM_FINAL_PATH+"/conf/mfcc.conf")

    with open(AM_FINAL_PATH+"/conf/ivector_extractor.conf") as f:
        values = f.readlines()
        with open(TEMP_FILE_PATH1+"/ivector_extractor.conf", 'w') as f:
            for i in values:
                f.write(i)
            f.write("--splice-config="+AM_FINAL_PATH+"/conf/splice.conf\n")
            f.write("--cmvn-config="+AM_FINAL_PATH+"/conf/online_cmvn.conf\n")
            f.write("--lda-matrix="+AM_FINAL_PATH+"/ivector_extractor/final.mat\n")
            f.write("--global-cmvn-stats="+AM_FINAL_PATH+"/ivector_extractor/global_cmvn.stats\n")
            f.write("--diag-ubm="+AM_FINAL_PATH+"/ivector_extractor/final.dubm\n")
            f.write("--ivector-extractor="+AM_FINAL_PATH+"/ivector_extractor/final.ie")

    #Run server
    app.run(host='0.0.0.0', port=SERVICE_PORT, debug=True, threaded=False, processes=1)

