#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, abort, Response, json
from vosk import Model, KaldiRecognizer
from tools import Worker, SpeakerDiarization, Punctuation
from time import gmtime, strftime
from gevent.pywsgi import WSGIServer
import argparse
import os
import _thread
import uuid

app = Flask("__stt-standelone-worker__")

max_duration = 1800

# instantiate services
worker = Worker()
punctuation = Punctuation()
speakerdiarization = SpeakerDiarization()

# Load ASR models (acoustic model and decoding graph)
worker.log.info('Load acoustic model and decoding graph')
model = Model(worker.AM_PATH, worker.LM_PATH,
              worker.CONFIG_FILES_PATH+"/online.conf")
spkModel = None

def decode(is_metadata):
    if len(worker.data) / worker.rate > max_duration :
        recognizer = KaldiRecognizer(model, spkModel, worker.rate, is_metadata, True)
        for i in range(0, len(worker.data), int(worker.rate/4)):
            if recognizer.AcceptWaveform(worker.data[i:i + int(worker.rate/4)]):
                recognizer.Result()
    else:
        recognizer = KaldiRecognizer(model, None, worker.rate, is_metadata, False)
        recognizer.AcceptWaveform(worker.data)

    data = recognizer.FinalResult()
    confidence = recognizer.uttConfidence()
    if is_metadata:
        data = recognizer.GetMetadata()
    return data, confidence

def processing(is_metadata, do_spk, audio_buffer, file_path=None):
    try:
        worker.log.info("Start decoding")
        data, confidence = decode(is_metadata)
        worker.log.info("Decoding complete")
        worker.log.info("Post Processing ...")
        spk = None
        if do_spk:
            spk = speakerdiarization.get(audio_buffer)
        trans = worker.get_response(data, spk, confidence, is_metadata)
        response = punctuation.get(trans)
        worker.log.info("... Complete")
        if file_path is not None:
            with open(file_path, 'w') as outfile:
                json.dump(response, outfile)
        else:
            return response
    except Exception as e:
        worker.log.error(e)
        exit(1)

# API
@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return "1", 200

@app.route('/transcription/<jobid>', methods=['GET'])
def transcription(jobid):
    file_path = worker.TRANS_FILES_PATH + "/" + str(jobid)
    if os.path.exists(file_path):
        return json.load(open(file_path,)), 200
    else:
        return "jobid {} is invalid".format(str(jobid)), 400

@app.route('/jobids', methods=['GET'])
def get():
    return json.load(open(worker.TRANS_FILES_PATH + "/jobids.json")), 200

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        worker.log.info('[%s] Transcribe request received' %
                        (strftime("%d/%b/%d %H:%M:%S", gmtime())))

        is_metadata = False
        do_spk = True

        # get response content type
        if request.headers.get('accept').lower() == 'application/json':
            is_metadata = True
        elif request.headers.get('accept').lower() == 'application/json-nospk':
            is_metadata = True
            do_spk = False
        elif request.headers.get('accept').lower() == 'text/plain':
            is_metadata = False
            do_spk = False
        else:
            raise ValueError('Not accepted header')

        # get input file
        if 'file' not in request.files.keys():
            raise ValueError('No audio file was uploaded')

        audio_buffer = request.files['file'].read()
        worker.getAudio(audio_buffer)
        duration = int(len(worker.data) / worker.rate)
        if duration > max_duration:
            jobid = str(uuid.uuid4())
            file_path = worker.TRANS_FILES_PATH + "/" + jobid
        
            pids = json.load(open(worker.TRANS_FILES_PATH + "/jobids.json"))
            pids['jobids'].append({'jobid':jobid, 'time':strftime("%d/%b/%d %H:%M:%S", gmtime())})
            with open(worker.TRANS_FILES_PATH + "/jobids.json", 'w') as pids_file:
                json.dump(pids, pids_file)

            _thread.start_new_thread(processing, (is_metadata, do_spk, audio_buffer, file_path,))
            estdur = str(int(duration*0.3)) if is_metadata else str(int(duration*0.18))
            response = {
                'jobid': jobid,
                'decoding_time': '~' + estdur + ' seconds',
                'message': "Use the jobid to get the transcription after decoding",
            }
            return response, 200
        response = processing(is_metadata, do_spk, audio_buffer)
        
        return response, 200
    except ValueError as e:
        worker.log.error(e)
        return str(e), 400
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
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--puctuation',
            type=int,
            help='punctuation service status',
            default=0)
        parser.add_argument(
            '--speaker_diarization',
            type=int,
            help='speaker diarization service status',
            default=0)
        args = parser.parse_args()

        punctuation.setParam(True if args.puctuation else False)
        speakerdiarization.setParam(True if args.speaker_diarization else False)
        
        # start SwaggerUI
        if worker.SWAGGER_PATH != '':
            worker.swaggerUI(app)
        # Run server

        http_server = WSGIServer(('', worker.SERVICE_PORT), app)
        http_server.serve_forever()

    except Exception as e:
        worker.log.error(e)
        exit(e)
