#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, abort, Response, json
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS
from tools import ASR
import yaml, os

app = Flask(__name__)

# Main parameters
AM_PATH = '/opt/models/AM'
LM_PATH = '/opt/models/LM'
TEMP_FILE_PATH = '/opt/tmp'
CONFIG_FILES_PATH = '/opt/config'
SERVICE_PORT=80
SWAGGER_URL='/api-doc'
asr = ASR(AM_PATH,LM_PATH, CONFIG_FILES_PATH)


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

@app.route('/transcribe', methods=['POST'])
def transcribe():
    return 'Test', 200

@app.route('/healthcheck', methods=['GET'])
def check():
    return '1', 200

# Rejected request handlers
@app.errorhandler(405)
def page_not_found(error):
    return 'The method is not allowed for the requested URL', 405

@app.errorhandler(404)
def page_not_found(error):
    return 'The requested URL was not found', 404

if __name__ == '__main__':
    #start SwaggerUI
    swaggerUI()
    
    #Run ASR engine
    asr.run()

    #Run server
    app.run(host='0.0.0.0', port=SERVICE_PORT, debug=True, threaded=False, processes=1)