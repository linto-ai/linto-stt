#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import configparser
import logging
import os
import io
import re
import uuid
import json
import yaml
import numpy as np
import wavio
from flask_swagger_ui import get_swaggerui_blueprint
import requests


class Worker:
    def __init__(self):
        # Set logger config
        self.log = logging.getLogger("__stt-standelone-worker__.Worker")
        logging.basicConfig(level=logging.INFO)

        # Main parameters
        self.AM_PATH = '/opt/models/AM'
        self.LM_PATH = '/opt/models/LM'
        self.TEMP_FILE_PATH = '/opt/tmp'
        self.TRANS_FILES_PATH = '/opt/trans'
        self.CONFIG_FILES_PATH = '/opt/config'
        self.SAVE_AUDIO = False
        self.SERVICE_PORT = 80
        self.SWAGGER_URL = '/api-doc'
        self.SWAGGER_PREFIX = ''
        self.SWAGGER_PATH = ''
        self.ONLINE = False

        if not os.path.isdir(self.CONFIG_FILES_PATH):
            os.mkdir(self.CONFIG_FILES_PATH)

        if not os.path.isdir(self.TEMP_FILE_PATH):
            os.mkdir(self.TEMP_FILE_PATH)

        if not os.path.isdir(self.TRANS_FILES_PATH):
            os.mkdir(self.TRANS_FILES_PATH)

        with open(self.TRANS_FILES_PATH + "/jobids.json", 'w') as outfile:
            json.dump({'jobids':[]}, outfile)

        # Environment parameters
        if 'SAVE_AUDIO' in os.environ:
            self.SAVE_AUDIO = True if os.environ['SAVE_AUDIO'].lower(
            ) == "true" else False
        if 'SWAGGER_PATH' in os.environ:
            self.SWAGGER_PATH = os.environ['SWAGGER_PATH']
        if 'SWAGGER_PREFIX' in os.environ:
            self.SWAGGER_PREFIX = os.environ['SWAGGER_PREFIX']

        # start loading ASR configuration
        self.log.info("Create the new config files")
        self.loadConfig()

    def swaggerUI(self, app):
        ### swagger specific ###
        swagger_yml = yaml.load(
            open(self.SWAGGER_PATH, 'r'), Loader=yaml.Loader)
        swaggerui = get_swaggerui_blueprint(
            # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
            self.SWAGGER_PREFIX+self.SWAGGER_URL,
            self.SWAGGER_PATH,
            config={  # Swagger UI config overrides
                'app_name': "STT API Documentation",
                'spec': swagger_yml
            }
        )
        app.register_blueprint(swaggerui, url_prefix=self.SWAGGER_URL)
        ### end swagger specific ###

    def getAudio(self, file):
        try:
            file_content = wavio.read(io.BytesIO(file))
            self.rate = file_content.rate
            self.data = file_content.data
            # if stereo file, convert to mono by computing the mean of the channels
            if len(self.data.shape) == 2 and self.data.shape[1] == 2:
                self.data = np.mean(self.data, axis=1, dtype=np.int16)
        except Exception as e:
            self.log.error(e)
            raise ValueError("The uploaded file format is not supported!!!")

    def saveFile(self, file):
        if self.SAVE_AUDIO:
            filename = str(uuid.uuid4())
            self.file_path = self.TEMP_FILE_PATH+"/"+filename
            file.save(self.file_path)
            

    # re-create config files
    def loadConfig(self):
        # load decoder parameters from "decode.cfg"
        decoder_settings = configparser.ConfigParser()
        if os.path.exists(self.AM_PATH+'/decode.cfg') == False:
            return False
        decoder_settings.read(self.AM_PATH+'/decode.cfg')

        # Prepare "online.conf"
        self.AM_PATH = self.AM_PATH+"/" + \
            decoder_settings.get('decoder_params', 'ampath')
        with open(self.AM_PATH+"/conf/online.conf") as f:
            values = f.readlines()
            with open(self.CONFIG_FILES_PATH+"/online.conf", 'w') as f:
                for i in values:
                    f.write(i)
                f.write("--ivector-extraction-config=" +
                        self.CONFIG_FILES_PATH+"/ivector_extractor.conf\n")
                f.write("--mfcc-config="+self.AM_PATH+"/conf/mfcc.conf\n")
                f.write(
                    "--beam="+decoder_settings.get('decoder_params', 'beam')+"\n")
                f.write(
                    "--lattice-beam="+decoder_settings.get('decoder_params', 'lattice_beam')+"\n")
                f.write("--acoustic-scale=" +
                        decoder_settings.get('decoder_params', 'acwt')+"\n")
                f.write(
                    "--min-active="+decoder_settings.get('decoder_params', 'min_active')+"\n")
                f.write(
                    "--max-active="+decoder_settings.get('decoder_params', 'max_active')+"\n")
                f.write("--frame-subsampling-factor="+decoder_settings.get(
                    'decoder_params', 'frame_subsampling_factor')+"\n")

        # Prepare "ivector_extractor.conf"
        with open(self.AM_PATH+"/conf/ivector_extractor.conf") as f:
            values = f.readlines()
            with open(self.CONFIG_FILES_PATH+"/ivector_extractor.conf", 'w') as f:
                for i in values:
                    f.write(i)
                f.write("--splice-config="+self.AM_PATH+"/conf/splice.conf\n")
                f.write("--cmvn-config="+self.AM_PATH +
                        "/conf/online_cmvn.conf\n")
                f.write("--lda-matrix="+self.AM_PATH +
                        "/ivector_extractor/final.mat\n")
                f.write("--global-cmvn-stats="+self.AM_PATH +
                        "/ivector_extractor/global_cmvn.stats\n")
                f.write("--diag-ubm="+self.AM_PATH +
                        "/ivector_extractor/final.dubm\n")
                f.write("--ivector-extractor="+self.AM_PATH +
                        "/ivector_extractor/final.ie")

        # Prepare "word_boundary.int" if not exist
        if not os.path.exists(self.LM_PATH+"/word_boundary.int") and os.path.exists(self.AM_PATH+"/phones.txt"):
            self.log.info("Create word_boundary.int based on phones.txt")
            with open(self.AM_PATH+"/phones.txt") as f:
                phones = f.readlines()

            with open(self.LM_PATH+"/word_boundary.int", "w") as f:
                for phone in phones:
                    phone = phone.strip()
                    phone = re.sub('^<eps> .*', '', phone)
                    phone = re.sub('^#\d+ .*', '', phone)
                    if phone != '':
                        id = phone.split(' ')[1]
                        if '_I ' in phone:
                            f.write(id+" internal\n")
                        elif '_B ' in phone:
                            f.write(id+" begin\n")
                        elif '_E ' in phone:
                            f.write(id+" end\n")
                        elif '_S ' in phone:
                            f.write(id+" singleton\n")
                        else:
                            f.write(id+" nonword\n")

    # remove extra symbols
    def parse_text(self, text):
        text = re.sub(r"<unk>", "", text)  # remove <unk> symbol
        text = re.sub(r"#nonterm:[^ ]* ", "", text)  # remove entity's mark
        text = re.sub(r"' ", "'", text)  # remove space after quote '
        text = re.sub(r" +", " ", text)  # remove multiple spaces
        text = text.strip()
        return text

    # Postprocess response
    def get_response(self, dataJson, speakers, confidence, is_metadata):
        if dataJson is not None:
            data = json.loads(dataJson)
            data['conf'] = confidence
            if 'text' in data:
                if not is_metadata:
                    text = data['text']  # get text from response
                    return self.parse_text(text)
                elif 'words' in data and len(data['words']) > 0:
                    # Generate final output data
                    return self.process_output(data, speakers)
        return None

    # return a json object including word-data, speaker-data
    def process_output(self, data, spkrs):
        try:
            speakers = []
            text = []
            i = 0
            text_ = ""
            words = []

            # Capitalize first word
            data['words'][0]['word'] = data['words'][0]['word'].capitalize()
            
            for word in data['words']:
                if i+1 == len(spkrs):
                    continue
                if i+1 < len(spkrs) and word["end"] < spkrs[i+1]["seg_begin"]:
                    text_ += word["word"] + " "
                    words.append(word)
                elif len(words) != 0:
                    speaker = {}
                    speaker["start"] = words[0]["start"]
                    speaker["end"] = words[-1]["end"]
                    speaker["speaker_id"] = str(spkrs[i]["spk_id"])
                    speaker["words"] = words

                    text.append(
                        str(spkrs[i]["spk_id"])+' : ' + self.parse_text(text_))
                    speakers.append(speaker)

                    words = [word]
                    text_ = word["word"] + " "
                    i += 1
                else:
                    words = [word]
                    text_ = word["word"] + " "
                    i += 1

            if i == 0:
                words = data['words']
                text_ = data['text'].capitalize()

            speaker = {}
            speaker["start"] = words[0]["start"]
            speaker["end"] = words[-1]["end"]
            speaker["speaker_id"] = str(spkrs[i]["spk_id"])
            speaker["words"] = words

            text.append(str(spkrs[i]["spk_id"]) +
                        ' : ' + self.parse_text(text_))
            speakers.append(speaker)

            return {'speakers': speakers, 'text': text, 'confidence-score': data['conf']}
        except Exception as e:
            self.log.error(e)
            return {'text': data['text'], 'words': data['words'], 'confidence-score': data['conf'], 'speakers': []}


class SpeakerDiarization:
    def __init__(self):
        self.SPEAKER_DIARIZATION_ISON = False
        self.SPEAKER_DIARIZATION_HOST = None
        self.SPEAKER_DIARIZATION_PORT = None
        self.url = None
        self.log = logging.getLogger(
            "__stt-standelone-worker__.SpeakerDiarization")
        logging.basicConfig(level=logging.INFO)

    def setParam(self, SPEAKER_DIARIZATION_ISON):
        self.SPEAKER_DIARIZATION_ISON = SPEAKER_DIARIZATION_ISON
        if self.SPEAKER_DIARIZATION_ISON:
            self.SPEAKER_DIARIZATION_HOST = os.environ['SPEAKER_DIARIZATION_HOST']
            self.SPEAKER_DIARIZATION_PORT = os.environ['SPEAKER_DIARIZATION_PORT']
            self.url = "http://"+self.SPEAKER_DIARIZATION_HOST + \
                ":"+self.SPEAKER_DIARIZATION_PORT+"/"
        self.log.info(self.url) if self.url is not None else self.log.warn(
            "The Speaker Diarization service is not running!")

    def get(self, audio_buffer, duration):
        emptyReturn = [{
                        "seg_id":1,
                        "spk_id":"spk1",
                        "seg_begin":0,
                        "seg_end":duration,
                    }]

        try:
            if self.SPEAKER_DIARIZATION_ISON:
                result = requests.post(self.url, files={'file': audio_buffer})
                if result.status_code != 200:
                    raise ValueError(result.text)

                speakers = json.loads(result.text)
                speakers = speakers["segments"]

                last_spk = {
                    'seg_begin': speakers[len(speakers) - 1]["seg_end"] + 10,
                    'seg_end': -1,
                    'spk_id': -1,
                    'seg_id': -1,
                }
                speakers.append(last_spk)
                
                return speakers
            else:
                return emptyReturn
        except Exception as e:
            self.log.error(str(e))
            return emptyReturn
        except ValueError as error:
            self.log.error(str(error))
            return emptyReturn


class Punctuation:
    def __init__(self):
        self.PUCTUATION_ISON = False
        self.PUCTUATION_HOST = None
        self.PUCTUATION_PORT = None
        self.PUCTUATION_ROUTE = None
        self.url = None
        self.log = logging.getLogger("__stt-standelone-worker__.Punctuation")
        logging.basicConfig(level=logging.INFO)

    def setParam(self, PUCTUATION_ISON):
        self.PUCTUATION_ISON = PUCTUATION_ISON
        if self.PUCTUATION_ISON:
            self.PUCTUATION_HOST = os.environ['PUCTUATION_HOST']
            self.PUCTUATION_PORT = os.environ['PUCTUATION_PORT']
            self.PUCTUATION_ROUTE = os.environ['PUCTUATION_ROUTE']
            self.PUCTUATION_ROUTE = re.sub('^/','',self.PUCTUATION_ROUTE)
            self.PUCTUATION_ROUTE = re.sub('"|\'','',self.PUCTUATION_ROUTE)
            self.url = "http://"+self.PUCTUATION_HOST+":"+self.PUCTUATION_PORT+"/"+self.PUCTUATION_ROUTE
        self.log.info(self.url) if self.url is not None else self.log.warn(
            "The Punctuation service is not running!")

    def get(self, obj):
        try:
            if self.PUCTUATION_ISON:
                if isinstance(obj, dict):          
                    if isinstance(obj['text'], list):
                        text_punc = []
                        for utterance in obj['text']:
                            data = utterance.split(':')
                            result = requests.post(self.url, data=data[1].strip().encode('utf-8'), headers={'content-type': 'application/octet-stream'})
                            if result.status_code != 200:
                                raise ValueError(result.text)
                            
                            text_punc.append(data[0]+": "+result.text)
                        obj['text-punc'] = text_punc
                    else:
                        result = requests.post(self.url, data=obj['text'].strip().encode('utf-8'), headers={'content-type': 'application/octet-stream'})
                        obj['text-punc'] = result.text
                    return obj
                else:
                    result = requests.post(self.url, data=text.encode('utf-8'), headers={'content-type': 'application/octet-stream'})
                    if result.status_code != 200:
                        raise ValueError(result.text)

                    return result.text
            else:
                return obj
        except Exception as e:
            self.log.error(str(e))
            return obj
        except ValueError as error:
            self.log.error(str(error))
            return obj

