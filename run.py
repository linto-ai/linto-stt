#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, abort, Response, json
from tools import ASR, SttStandelone
from time import gmtime, strftime
from gevent.pywsgi import WSGIServer
import os

app = Flask("__stt-standelone-worker__")

stt = SttStandelone()

# Load ASR models (acoustic model and decoding graph)
stt.log.info('Load acoustic model and decoding graph')
asr = ASR(stt.AM_PATH, stt.LM_PATH, stt.CONFIG_FILES_PATH)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        stt.log.info('[%s] New user entry on /transcribe' % (strftime("%d/%b/%d %H:%M:%S", gmtime())))
        
        #get response content type
        metadata = False
        if request.headers.get('accept').lower() == 'application/json':
            metadata = True
        elif request.headers.get('accept').lower() == 'text/plain':
            metadata = False
        else:
            raise ValueError('Not accepted header')

        #get input file
        if 'file' in request.files.keys():
            file = request.files['file']
            stt.read_audio(file,asr.get_sample_rate())
            output = stt.run(asr, metadata)
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
        # start SwaggerUI
        if os.path.exists(stt.SWAGGER_PATH):
            stt.swaggerUI(app)

        #Run server
        app.logger.info('Server ready for transcription...')
        http_server = WSGIServer(('', stt.SERVICE_PORT), app)
        http_server.serve_forever()
    except Exception as e:
        app.logger.error(e)
        exit(e)