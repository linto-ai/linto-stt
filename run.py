#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, abort, Response, json
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
from tools import ASR, Audio, Logger
import yaml, os, sox, logging

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)


# Main parameters
AM_PATH = '/opt/models/AM'
LM_PATH = '/opt/models/LM'
TEMP_FILE_PATH = '/opt/tmp'
CONFIG_FILES_PATH = '/opt/config'
SAVE_AUDIO = False
SERVICE_PORT = 80
SWAGGER_URL = '/api-doc'
asr = ASR(AM_PATH,LM_PATH, CONFIG_FILES_PATH)
audio = Audio()
asr.set_logger(Logger(app,"ASR"))
audio.set_logger(Logger(app,"AUDIO"))

if not os.path.isdir(TEMP_FILE_PATH):
    os.mkdir(TEMP_FILE_PATH)
if not os.path.isdir(CONFIG_FILES_PATH):
    os.mkdir(CONFIG_FILES_PATH)

# Environment parameters
if 'SERVICE_PORT' in os.environ:
    SERVICE_PORT = os.environ['SERVICE_PORT']
if 'SWAGGER_PATH' not in os.environ:
    exit("You have to provide a 'SWAGGER_PATH'")
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

def getAudio(file):
    file_path = TEMP_FILE_PATH+file.filename.lower()
    file.save(file_path)
    audio.transform(file_path)
    if not SAVE_AUDIO:
        os.remove(file_path)
    
@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        #get response content type
        if request.headers.get('accept').lower() == 'application/json':
            metadata = True
        elif request.headers.get('accept').lower() == 'text/plain':
            metadata = False
        else:
            raise ValueError('Not accepted header')
        
        #get input file
        if 'file' in request.files.keys():
            file = request.files['file']
            getAudio(file)
            text = asr.decoder(audio)
        else:
            raise ValueError('No audio file was uploaded')

        return text, 200
    except ValueError as error:
        return str(error), 400
    except Exception as e:
        app.logger.error(e)
        return 'Server Error', 500

@app.route('/healthcheck', methods=['GET'])
def check():
    return '', 200

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
    #start SwaggerUI
    swaggerUI()
    #Run ASR engine
    asr.run()
    #Set Audio Sample Rate
    audio.set_sample_rate(asr.get_sample_rate())

    #Run server
    app.run(host='0.0.0.0', port=SERVICE_PORT, debug=True, threaded=False, processes=1)