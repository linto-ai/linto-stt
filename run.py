#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, abort, Response, json
from vosk import Model, KaldiRecognizer
from tools import WorkerStreaming
from time import gmtime, strftime

from gevent.pywsgi import WSGIServer



app = Flask("__stt-standelone-worker__")

# create WorkerStreaming object
worker = WorkerStreaming()

# Load ASR models (acoustic model and decoding graph)
worker.log.info('Load acoustic model and decoding graph')
model = Model(worker.AM_PATH, worker.LM_PATH,
              worker.CONFIG_FILES_PATH+"/online.conf")


# API
@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        worker.log.info('[%s] New user entry on /transcribe' %
                        (strftime("%d/%b/%d %H:%M:%S", gmtime())))

        metadata = worker.METADATA
        nbrSpk = 10

        # get response content type
        if request.headers.get('accept').lower() == 'application/json':
            metadata = True
        elif request.headers.get('accept').lower() == 'text/plain':
            metadata = False
        else:
            raise ValueError('Not accepted header')

        # get speaker parameter
        spkDiarization = False
        if request.form.get('speaker') != None and (request.form.get('speaker').lower() == 'yes' or request.form.get('speaker').lower() == 'no'):
            spkDiarization = True if request.form.get(
                'speaker').lower() == 'yes' else False
            # get number of speakers parameter
            try:
                if request.form.get('nbrSpeaker') != None and spkDiarization and int(request.form.get('nbrSpeaker')) > 0:
                    nbrSpk = int(request.form.get('nbrSpeaker'))
                elif request.form.get('nbrSpeaker') != None and spkDiarization:
                    raise ValueError(
                        'Not accepted "nbrSpeaker" field value (nbrSpeaker>0)')
            except Exception as e:
                worker.log.error(e)
                raise ValueError(
                    'Not accepted "nbrSpeaker" field value (nbrSpeaker>0)')
        else:
            if request.form.get('speaker') != None:
                raise ValueError('Not accepted "speaker" field value (yes|no)')

        # get input file
        if 'file' in request.files.keys():
            file = request.files['file']
            worker.getAudio(file)
            rec = KaldiRecognizer(model, worker.rate, metadata)
            response = rec.Decode(worker.data)
            if metadata:
                obj = rec.GetMetadata()
                data = json.loads(obj)
                response = worker.process_metadata(data, spkDiarization, nbrSpk)
        else:
            raise ValueError('No audio file was uploaded')

        return response, 200
    except ValueError as error:
        return str(error), 400
    except Exception as e:
        worker.log.error(e)
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
