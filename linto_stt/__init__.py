import logging
import argparse
import importlib
import uvicorn
import os
from typing import Optional
from dotenv import load_dotenv

from .websocket.websocketserver import main as ws_server
from .http_server import main as http_server


def load_engine_env(engine: str):

    engines_dir = os.path.join(os.path.dirname(__file__), "engines")
    load_dotenv(".env")
    load_dotenv(os.path.join(engines_dir, engine, ".envdefault"))


def import_stt_module(engine: str, submodule: str = None):
    module_path = f".engines.{engine}.stt.processing"
    if submodule:
        module_path = f"{module_path}.{submodule}"
    return importlib.import_module(module_path, package="linto_stt")


logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)


def server_factory():
    engine = os.environ.get("STT_ENGINE", 'nemo')
    load_engine_env(engine)
    stt = import_stt_module(engine)
    return http_server(stt.MODEL, stt.USE_GPU, stt.decode, stt.load_wave_buffer, stt.warmup)


def main():
    parser = argparse.ArgumentParser(
        prog='LinTO-STT',
        description='STT for LinTO with multiple engines')

    parser.add_argument('-m', '--mode', required=True,
                        choices=['http', 'task', 'websocket'])
    parser.add_argument('-e', '--engine', required=False, choices=[
                        'nemo', 'whisper', 'kaldi', 'kyutai'], default=os.environ.get("STT_ENGINE", 'nemo'))
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
    os.environ["STT_ENGINE"] = args.engine
    mode = args.mode

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
    engine = os.environ.get("STT_ENGINE", 'nemo')
    load_engine_env(engine)
    stt = import_stt_module(engine)
    stt_streaming = import_stt_module(engine, "streaming")
    ws_server(host, port, stt_streaming.wssDecode, stt.MODEL)


def run_celery_server():
    from .celery import celery as app

    engine = os.environ.get("STT_ENGINE", 'nemo')
    load_engine_env(engine)
    stt = import_stt_module(engine)

    @app.task(name='transcribe_task')
    def transcribe_task(file_name: str, with_metadata: bool, language: Optional[str] = None):
        stt_utils = import_stt_module(engine, "utils")
        audio_dir = os.environ.get("AUDIO_DIR", "/opt/audio")
        file_path = os.path.join(audio_dir, file_name)
        try:
            file_content = stt_utils.load_audiofile(file_path)
        except Exception:
            import traceback
            raise Exception(f"{traceback.format_exc()}\nFailed to load resource {file_path}")
        try:
            result = stt.decode(file_content, stt.MODEL, with_metadata, language=language)
        except Exception:
            import traceback
            raise Exception(f"{traceback.format_exc()}\nFailed to decode {file_path}")
        return result

    service_name = os.environ.get("SERVICE_NAME", engine)
    concurrency = os.environ.get("CONCURRENCY", "1")
    worker_args = [
        'worker',
        '--loglevel=info',
        '-c', concurrency,
        '-Ofair',
        '-Q', service_name,
        '-n', f'{service_name}_worker@%h',
    ]

    # GPU: use solo pool to avoid CUDA context issues with prefork
    if stt.USE_GPU:
        worker_args.extend(['--pool=solo'])

    app.worker_main(argv=worker_args)
