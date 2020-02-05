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

def decode(audio_file,wav_name,do_word_tStamp):
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
    trans=re.sub(wav_name,'',hypothesis[0]).strip()
    trans=re.sub(r"#nonterm:[^ ]* ", "", trans)
    trans=re.sub(r" <unk> ", " ", " "+trans+" ")


    # Get the begin and end time stamp from the decoder output
    if do_word_tStamp:
        error, output = run_shell_command("kaldi-lattice-1best --acoustic-scale="+DECODER_ACWT+" ark:"+TEMP_FILE_PATH+"/"+wav_name+".lat ark:"+TEMP_FILE_PATH+"/"+wav_name+".1best")
        error, output = run_shell_command("kaldi-lattice-align-words "+decode_words_boundary+" "+decode_mdl+" ark:"+TEMP_FILE_PATH+"/"+wav_name+".1best ark:"+TEMP_FILE_PATH+"/"+wav_name+".words") 
        error, output = run_shell_command("kaldi-nbest-to-ctm ark:"+TEMP_FILE_PATH+"/"+wav_name+".words "+TEMP_FILE_PATH+"/"+wav_name+".ctm")
        error, output = run_shell_command("int2sym.pl -f 5 "+decode_words+" "+TEMP_FILE_PATH+"/"+wav_name+".ctm")
        if not error and output != "":
            words = output.split("\n")
            trans = ""
            data = {}
            data["words"] = []
            for word in words:
                _word = word.strip().split(' ')
                if len(_word) == 5:
                    meta = {}
                    word = re.sub("<unk>","",_word[4])
                    word = re.sub("<unk>","",_word[4])
                    if word != "":
                        trans = trans+" "+word
                        meta["word"] = word
                        meta["stime"] = float(_word[2])
                        meta["etime"] = (float(_word[2]) + float(_word[3]))
                        meta["score"] = float(_word[1])
                        data["words"].append(meta)
            data["transcription"] = trans.strip()
            return True, data
        else:
            app.logger.info("error during word time stamp generation")

    return True, trans.strip()

@app.route('/transcribe', methods=['POST'])
def transcribe():
    global busy
    busy=1
    fileid = str(uuid.uuid4())
    metadata = True if request.args.get('metadata').lower() == 'json' else False
    
    if 'file' in request.files.keys():
        file = request.files['file']
        file_ext = file.filename.rsplit('.', 1)[-1].lower()
        file_type = file.content_type.rsplit('/', 1)[0]
        if file_type == "audio":
            filename = TEMP_FILE_PATH+'/'+fileid+'.'+file_ext
            file.save(filename)
            b, out = decode(filename,fileid,metadata)
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
    json_string = json.dumps(out, ensure_ascii=False)
    return Response(json_string,content_type="application/json; charset=utf-8" ), 200

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

