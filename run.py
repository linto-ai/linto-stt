#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, abort, Response, json
from vosk import Model, KaldiRecognizer
from tools import Worker
from time import gmtime, strftime
from gevent.pywsgi import WSGIServer



app = Flask("__stt-standelone-worker__")

# create WorkerStreaming object
worker = Worker()

# Load ASR models (acoustic model and decoding graph)
worker.log.info('Load acoustic model and decoding graph')
model = Model(worker.AM_PATH, worker.LM_PATH,
              worker.CONFIG_FILES_PATH+"/online.conf")
spkModel = None

# API
@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        worker.log.info('[%s] New user entry on /transcribe' %
                        (strftime("%d/%b/%d %H:%M:%S", gmtime())))

        is_metadata = False

        # get response content type
        if request.headers.get('accept').lower() == 'application/json':
            is_metadata = True
        elif request.headers.get('accept').lower() == 'text/plain':
            is_metadata = False
        else:
            raise ValueError('Not accepted header')

        # get input file
        if 'file' in request.files.keys():
            file = request.files['file']
            worker.getAudio(file)
            rec = KaldiRecognizer(model, spkModel, worker.rate, worker.ONLINE)
            rec.AcceptWaveform(worker.data)
            data_ = rec.FinalResult()
            confidence = rec.uttConfidence()
            if is_metadata:
                data_ = rec.GetMetadata()
            data = worker.get_response(data_, confidence, is_metadata)
            worker.clean()
        else:
            raise ValueError('No audio file was uploaded')

        return data, 200
    except ValueError as error:
        return str(error), 400
    except Exception as e:
        worker.log.error(e)
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
    worker.log.error(error)
    return 'Server Error', 500


if __name__ == '__main__':
    try:
        # start SwaggerUI
        if worker.SWAGGER_PATH != '':
            worker.swaggerUI(app)
        # Run server

        http_server = WSGIServer(('', worker.SERVICE_PORT), app)
        http_server.serve_forever()

    except Exception as e:
        worker.log.error(e)
        exit(e)
