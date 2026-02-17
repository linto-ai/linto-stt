import logging
import argparse
import importlib
import uvicorn
import os
from typing import Optional
from dotenv import load_dotenv

from .websocket.websocketserver import main as ws_server
from .http_server import main as http_server


def load_backend_env(backend: str):

    backends_dir = os.path.join(os.path.dirname(__file__), "backends")
    load_dotenv(".env")
    load_dotenv(os.path.join(backends_dir, backend, ".envdefault"))


def import_stt_module(backend: str, submodule: str = None):
    module_path = f".backends.{backend}.stt.processing"
    if submodule:
        module_path = f"{module_path}.{submodule}"
    return importlib.import_module(module_path, package="linto_stt")


logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)


def server_factory():
    backend = os.environ.get("SERVICE_NAME", 'nemo')
    load_backend_env(backend)
    stt = import_stt_module(backend)
    return http_server(stt.MODEL, stt.USE_GPU, stt.decode, stt.load_wave_buffer, stt.warmup)


def main():
    parser = argparse.ArgumentParser(
        prog='LinTO-STT',
        description='STT for LinTO with multiple backends')

    parser.add_argument('-m', '--mode', required=True,
                        choices=['http', 'task', 'websocket'])
    parser.add_argument('-b', '--backend', required=False, choices=[
                        'nemo', 'whisper', 'kaldi', 'kyutai'], default=os.environ.get("SERVICE_NAME", 'nemo'))
    parser.add_argument('-p', '--port', required=False, default=8080, type=int)
    parser.add_argument('-i', '--host', required=False, default="127.0.0.1")
    parser.add_argument(
        '-w',
        "--workers",
        type=int,
        required=False,
        default=1,
        help="Number of Gunicorn workers",
    )

    args = parser.parse_args()
    os.environ["SERVICE_NAME"] = args.backend
    mode = args.mode if args.mode else os.environ.get("SERVICE_MODE")

    if mode is None:
        logging.error(
            "No mode specified, must specify an environment variable SERVICE_MODE in [ http | task | websocket ] or use -m option")

    if mode == 'websocket':
        run_websocket_server(args.host, args.port)

    elif mode == 'http':
        run_http_server(args.host, args.port, args.workers)

    elif mode == 'task':
        run_celery_server()

    else:
        logging.error(
            "Unknown mode, must specify an environment variable SERVICE_MODE in [ http | task | websocket ] or use -m option")


def run_http_server(host, port, workers):
    uvicorn.run('linto_stt:server_factory', host=host,
                port=port, factory=True, workers=workers)


def run_websocket_server(host, port):
    backend = os.environ.get("SERVICE_NAME", 'nemo')
    load_backend_env(backend)
    stt = import_stt_module(backend)
    stt_streaming = import_stt_module(backend, "streaming")
    ws_server(host, port, stt_streaming.wssDecode, stt.MODEL)


def run_celery_server():
    from .celery import celery as app

    @app.task(name='transcribe_task')
    def transcribe_task(file_name: str, with_metadata: bool, language: Optional[str] = None):
        backend = os.environ.get("SERVICE_NAME", 'nemo')
        load_backend_env(backend)
        stt = import_stt_module(backend)
        stt_utils = import_stt_module(backend, "utils")

        audio_dir = os.environ.get("AUDIO_DIR", "/opt/audio")
        file_path = os.path.join(audio_dir, file_name)
        try:
            file_content = stt_utils.load_audiofile(file_path)
        except Exception as err:
            import traceback
            msg = f"{traceback.format_exc()}\nFailed to load ressource {file_path}"
            raise Exception(msg)  # from err

        # Decode
        try:
            result = stt.decode(file_content, stt.MODEL,
                                with_metadata, language=language)
        except Exception as err:
            import traceback

            msg = f"{traceback.format_exc()}\nFailed to decode {file_path}"
            raise Exception(msg)  # from err

        return result
    app.worker_main(argv=['worker', '--loglevel=info'])
