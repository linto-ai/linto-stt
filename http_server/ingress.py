#!/usr/bin/env python3

import json
import logging
import time

from confparser import createParser
from flask import Flask, json, request
from serving import GunicornServing, GeventServing
from swagger import setupSwaggerUI

from stt.processing import decode, load_wave_buffer, model, alignment_model, use_gpu
from stt import logger as stt_logger

app = Flask("__stt-standalone-worker__")
app.config["JSON_AS_ASCII"] = False
app.config["JSON_SORT_KEYS"] = False

logger = logging.getLogger("__stt-standalone-worker__")


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return json.dumps({"healthcheck": "OK"}), 200


@app.route("/oas_docs", methods=["GET"])
def oas_docs():
    return "Not Implemented", 501


@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        logger.info(f"Transcribe request received")

        # get response content type
        # logger.debug(request.headers.get("accept").lower())
        if request.headers.get("accept").lower() == "application/json":
            join_metadata = True
        elif request.headers.get("accept").lower() == "text/plain":
            join_metadata = False
        else:
            raise ValueError(f"Not accepted header (accept={request.headers.get('accept')} should be either application/json or text/plain)")
        # logger.debug("Metadata: {}".format(join_metadata))

        # get input file
        if "file" not in request.files.keys():
            raise ValueError(f"No audio file was uploaded (missing 'file' key)")

        file_buffer = request.files["file"].read()
        
        audio_data = load_wave_buffer(file_buffer)

        # Transcription
        transcription = decode(
            audio_data, model, alignment_model, join_metadata)

        if join_metadata:
            return json.dumps(transcription, ensure_ascii=False), 200
        return transcription["text"], 200

    except Exception as error:
        import traceback
        print(traceback.format_exc())
        logger.error(repr(error))
        return "Server Error: {}".format(str(error)), 400 if isinstance(error, ValueError) else 500


@app.errorhandler(405)
def method_not_allowed(_):
    return "The method is not allowed for the requested URL", 405


@app.errorhandler(404)
def page_not_found(_):
    return "The requested URL was not found", 404


@app.errorhandler(500)
def server_error(error):
    logger.error(error)
    return "Server Error", 500


if __name__ == "__main__":
    logger.info("Startup...")

    parser = createParser()
    args = parser.parse_args()
    logger_level = logging.DEBUG if args.debug else logging.INFO
    logger.setLevel(logger_level)
    stt_logger.setLevel(logger_level)
    try:
        # Setup SwaggerUI
        if args.swagger_path is not None:
            setupSwaggerUI(app, args)
            logger.debug("Swagger UI set.")
    except Exception as err:
        logger.warning("Could not setup swagger: {}".format(str(err)))

    logger.info(f"Using {args.workers} workers")
    
    if use_gpu: # TODO: get rid of this?
        serving_type = GeventServing
        logger.debug("Serving with gevent")
    else:
        serving_type = GunicornServing
        logger.debug("Serving with gunicorn")

    serving = serving_type(
        app,
        {
            "bind": f"0.0.0.0:{args.service_port}",
            "workers": args.workers,
            "timeout": 3600,
        },
    )
    logger.info(args)
    try:
        serving.run()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as err:
        logger.error(str(err))
        logger.critical("Service is shut down (Error)")
        exit(err)
