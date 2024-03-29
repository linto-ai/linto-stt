#!/usr/bin/env python3

import json
import logging
import os
import time

from confparser import createParser
from flask import Flask, json, request
from serving import GeventServing, GunicornServing
from stt import logger as stt_logger
from stt.processing import MODEL, USE_GPU, decode, load_wave_buffer
from swagger import setupSwaggerUI

app = Flask("__stt-standalone-worker__")
app.config["JSON_AS_ASCII"] = False
app.config["JSON_SORT_KEYS"] = False

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logger = logging.getLogger("__stt-standalone-worker__")
logger.setLevel(logging.INFO)

# If websocket streaming route is enabled
if os.environ.get("ENABLE_STREAMING", False) in [True, "true", 1]:
    from flask_sock import Sock
    from stt.processing.streaming import ws_streaming

    logger.info("Init websocket serving ...")
    sock = Sock(app)
    logger.info("Streaming is enabled")

    @sock.route("/streaming")
    def streaming(web_socket):
        ws_streaming(web_socket, MODEL)


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
        # logger.debug(request.headers.get("accept").lower())
        if request.headers.get("accept").lower() == "application/json":
            join_metadata = True
        elif request.headers.get("accept").lower() == "text/plain":
            join_metadata = False
        else:
            raise ValueError(
                f"Not accepted header (accept={request.headers.get('accept')} should be either application/json or text/plain)"
            )
        # logger.debug("Metadata: {}".format(join_metadata))

        # get input file
        if "file" not in request.files.keys():
            raise ValueError(f"No audio file was uploaded (missing 'file' key)")

        file_buffer = request.files["file"].read()

        audio_data = load_wave_buffer(file_buffer)

        # Transcription
        transcription = decode(audio_data, MODEL, join_metadata)

        if join_metadata:
            return json.dumps(transcription, ensure_ascii=False), 200
        return transcription["text"], 200

    except Exception as error:
        import traceback

        logger.error(traceback.format_exc())
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

    if USE_GPU:  # TODO: get rid of this?
        serving_type = GeventServing
        logger.debug("Serving with gevent")
    else:
        serving_type = GunicornServing
        logger.debug("Serving with gunicorn")
    # serving_type = GunicornServing
    # logger.debug("Serving with gunicorn")
    
    def worker_started(worker):
        logger.info(f"Worker started {worker.pid}")
        MODEL[0].check_loaded()
        logger.info("Worker fully initialized")
        
    
    # def post_fork(server, worker):
    #     logger.info("Worker post fork")
    #     MODEL[0].check_loaded()
    #     logger.info("Worker f")
        

    serving = serving_type(
        app,
        {
            "bind": f"0.0.0.0:{args.service_port}",
            "workers": args.workers,
            "timeout": 3600 * 24,
            # "on_starting": lambda server: logger.info("Server started"),
            "post_worker_init": worker_started,
            # "post_fork": post_fork
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
