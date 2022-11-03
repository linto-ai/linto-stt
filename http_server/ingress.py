#!/usr/bin/env python3

import json
import logging
import os
from time import time

from confparser import createParser
from flask import Flask, Response, abort, json, request
from flask_sock import Sock
from serving import GunicornServing
from swagger import setupSwaggerUI

from stt.processing import decode, formatAudio, model
from stt.processing.streaming import ws_streaming

app = Flask("__stt-standalone-worker__")
app.config["JSON_AS_ASCII"] = False
app.config["JSON_SORT_KEYS"] = False

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger("__stt-standalone-worker__")

# If websocket streaming route is enabled
if os.environ.get("ENABLE_STREAMING", False) in [True, "true", 1]:
    logger.info("Init websocket serving ...")
    sock = Sock(app)
    logger.info("Streaming is enabled")

    @sock.route("/streaming")
    def streaming(web_socket):
        ws_streaming(web_socket, model)


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return json.dumps({"healthcheck": "OK"}), 200


@app.route("/oas_docs", methods=["GET"])
def oas_docs():
    return "Not Implemented", 501


@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        logger.info("Transcribe request received")

        # get response content type
        logger.debug(request.headers.get("accept").lower())
        if request.headers.get("accept").lower() == "application/json":
            join_metadata = True
        elif request.headers.get("accept").lower() == "text/plain":
            join_metadata = False
        else:
            raise ValueError("Not accepted header")
        logger.debug("Metadata: {}".format(join_metadata))

        # get input file
        if "file" in request.files.keys():
            file_buffer = request.files["file"].read()
            audio_data, sampling_rate = formatAudio(file_buffer)
            start_t = time()

            # Transcription
            transcription = decode(audio_data, model, sampling_rate, join_metadata)
            logger.debug("Transcription complete (t={}s)".format(time() - start_t))

            logger.debug("... Complete")

        else:
            raise ValueError("No audio file was uploaded")

        if join_metadata:
            return json.dumps(transcription, ensure_ascii=False), 200
        return transcription["text"], 200

    except ValueError as error:
        return str(error), 400
    except Exception as error:
        logger.error(error)
        return "Server Error: {}".format(str(error)), 500


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
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    try:
        # Setup SwaggerUI
        if args.swagger_path is not None:
            setupSwaggerUI(app, args)
            logger.debug("Swagger UI set.")
    except Exception as err:
        logger.warning("Could not setup swagger: {}".format(str(err)))

    serving = GunicornServing(
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
