#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, abort, Response, json
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
from tools import ASR, Audio, SpeakerDiarization, SttStandelone
import yaml, os, sox, logging
from time import gmtime, strftime
from gevent.pywsgi import WSGIServer

app = Flask("__stt-standelone-worker__")

# Set logger config
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Main parameters
AM_PATH = '/opt/models/AM'
LM_PATH = '/opt/models/LM'
TEMP_FILE_PATH = '/opt/tmp'
CONFIG_FILES_PATH = '/opt/config'
NBR_PROCESSES = 1
SAVE_AUDIO = False
SERVICE_PORT = 80
SWAGGER_URL = '/api-doc'
SWAGGER_PATH = ''
asr = ASR(AM_PATH,LM_PATH, CONFIG_FILES_PATH)

if not os.path.isdir(TEMP_FILE_PATH):
    os.mkdir(TEMP_FILE_PATH)
if not os.path.isdir(CONFIG_FILES_PATH):
    os.mkdir(CONFIG_FILES_PATH)

# Environment parameters
if 'SERVICE_PORT' in os.environ:
    SERVICE_PORT = os.environ['SERVICE_PORT']
if 'SAVE_AUDIO' in os.environ:
    SAVE_AUDIO = os.environ['SAVE_AUDIO']
if 'NBR_PROCESSES' in os.environ:
    if int(os.environ['NBR_PROCESSES']) > 0:
        NBR_PROCESSES = int(os.environ['NBR_PROCESSES'])
    else:
        exit("You must to provide a positif number of processes 'NBR_PROCESSES'")
if 'SWAGGER_PATH' in os.environ:
    SWAGGER_PATH = os.environ['SWAGGER_PATH']

def swaggerUI():
    ### swagger specific ###
    swagger_yml = yaml.load(open(SWAGGER_PATH, 'r'), Loader=yaml.Loader)
    swaggerui = get_swaggerui_blueprint(
        SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
        SWAGGER_PATH,
        config={  # Swagger UI config overrides
            'app_name': "STT API Documentation",
            'spec': swagger_yml
        }
    )
    app.register_blueprint(swaggerui, url_prefix=SWAGGER_URL)
    ### end swagger specific ###

def getAudio(file,audio):
    file_path = TEMP_FILE_PATH+file.filename.lower()
    file.save(file_path)
    audio.read_audio(file_path)
    if not SAVE_AUDIO:
        os.remove(file_path)
    
@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        app.logger.info('[%s] New user entry on /transcribe' % (strftime("%d/%b/%d %H:%M:%S", gmtime())))
        # create main objects
        spk = SpeakerDiarization()
        audio = Audio(asr.get_sample_rate())
        
        #get response content type
        metadata = False
        if request.headers.get('accept').lower() == 'application/json':
            metadata = True
        elif request.headers.get('accept').lower() == 'text/plain':
            metadata = False
        else:
            raise ValueError('Not accepted header')

        #get speaker parameter
        spkDiarization = False
        if request.form.get('speaker') != None and (request.form.get('speaker').lower() == 'yes' or request.form.get('speaker').lower() == 'no'):
            spkDiarization = True if request.form.get('speaker').lower() == 'yes' else False
            #get number of speakers parameter
            try:
                if request.form.get('nbrSpeaker') != None and spkDiarization and int(request.form.get('nbrSpeaker')) > 0:
                    spk.set_maxNrSpeakers(int(request.form.get('nbrSpeaker')))
                elif request.form.get('nbrSpeaker') != None and spkDiarization:
                    raise ValueError('Not accepted "nbrSpeaker" field value (nbrSpeaker>0)')
            except Exception as e:
                app.logger.error(e)
                raise ValueError('Not accepted "nbrSpeaker" field value (nbrSpeaker>0)')
        else:
            if request.form.get('speaker') != None:
                raise ValueError('Not accepted "speaker" field value (yes|no)')

        stt = SttStandelone(metadata,spkDiarization)
        
        #get input file
        if 'file' in request.files.keys():
            file = request.files['file']
            getAudio(file,audio)
            output = stt.run(audio,asr,spk)
        else:
            raise ValueError('No audio file was uploaded')

        return output, 200
    except ValueError as error:
        return str(error), 400
    except Exception as e:
        app.logger.error(e)
        return 'Server Error', 500

# Rejected request handlers
@app.errorhandler(405)
def method_not_allowed(error):
    return 'The method is not allowed for the requested URL', 405

@app.errorhandler(404)
def page_not_found(error):
    return 'The requested URL was not found', 404

@app.errorhandler(500)
def server_error(error):
    app.logger.error(error)
    return 'Server Error', 500

if __name__ == '__main__':
    try:
        #start SwaggerUI
        if SWAGGER_PATH != '':
            swaggerUI()

        #Run ASR engine
        asr.run()

        #Run server
        app.logger.info('Server ready for transcription...')
        http_server = WSGIServer(('', SERVICE_PORT), app)
        http_server.serve_forever()
    except Exception as e:
        app.logger.error(e)
        exit(e)