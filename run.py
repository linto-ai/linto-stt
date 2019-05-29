#!/usr/bin/env python2

from flask import Flask, request, abort, jsonify
import uuid, os
import configparser
import subprocess
import shlex
import requests
import json
import re

app = Flask(__name__)

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
        return True, output
    except OSError as err:
        print("OS error: {0}".format(err))
        return False, ''
    except ValueError:
        print("data error.")
        return False, ''
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return False, ''

def decode(wav_file,wav_name):
    b,o=run_shell_command("sox "+wav_file+".wav -t wav -b 16 -r 16000 -c 1 "+wav_file+"_tmp.wav")
    if not b:
        return False, ''
    b,o=run_shell_command("mv "+wav_file+"_tmp.wav "+wav_file+".wav")
    if not b:
        return False, ''


    decode_conf  = TEMP_FILE_PATH1+"/online.conf"
    decode_mdl   = AM_PATH+"/"+AM_FILE_PATH+"/final.mdl"
    decode_graph = LM_PATH+"/HCLG.fst"
    decode_words = LM_PATH+"/words.txt"

    if DECODER_SYS == 'dnn3':
        b,o=run_shell_command("kaldi-nnet3-latgen-faster --do-endpointing=false --frame-subsampling-factor="+DECODER_FSF+" --frames-per-chunk=20 --online=false --config="+decode_conf+" --minimize=false --min-active="+DECODER_MINACT+" --max-active="+DECODER_MAXACT+" --beam="+DECODER_BEAM+" --lattice-beam="+DECODER_LATBEAM+" --acoustic-scale="+DECODER_ACWT+" --word-symbol-table="+decode_words+" "+decode_mdl+" "+decode_graph+" \"ark:echo "+wav_name+" "+wav_name+"|\" \"scp:echo "+wav_name+" "+wav_file+"|\" ark:"+TEMP_FILE_PATH+"/"+wav_name+".lat")
    elif DECODER_SYS == 'dnn2' or DECODER_SYS == 'dnn':
        b,o=run_shell_command("kaldi-nnet2-latgen-faster --do-endpointing=false --online=false --config="+decode_conf+" --min-active="+DECODER_MINACT+" --max-active="+DECODER_MAXACT+" --beam="+DECODER_BEAM+" --lattice-beam="+DECODER_LATBEAM+" --acoustic-scale="+DECODER_ACWT+" --word-symbol-table="+decode_words+" "+decode_mdl+" "+decode_graph+" \"ark:echo "+wav_name+" "+wav_name+"|\" \"scp:echo "+wav_name+" "+wav_file+"|\" ark:"+TEMP_FILE_PATH+"/"+wav_name+".lat")
    else:
        b=False
        o='KaldiFatalError decode param is not recognized'

    if not b or 'KaldiFatalError' in o:
        print(o)
        return False, ''

    hypothesis = re.findall('\n'+wav_name+'.*',o)
    #app.logger.info(hypothesis)
    o=re.sub(wav_name,'',hypothesis[0]).strip()
    o=re.sub(r"#nonterm:[^ ]* ", "", o)

    return True, o

@app.route('/transcribe', methods=['POST'])
def transcribe():
    global busy
    busy=1
    fileid = str(uuid.uuid4())
    if 'wavFile' in request.files.keys():
        file = request.files['wavFile']
        filename = TEMP_FILE_PATH+'/'+fileid+'.wav'
        file.save(filename)
        b, out = decode(filename,fileid)
        if not b:
            busy=0
            abort(403)
    else:
        busy=0
        return 'No wave file was uploaded', 404

    # Delete temporary files
    for file in os.listdir(TEMP_FILE_PATH):
        os.remove(TEMP_FILE_PATH+"/"+file)
    busy=0
    return jsonify({'transcript':{'transcription':out}}), 200

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

