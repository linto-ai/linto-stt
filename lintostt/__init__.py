import logging
import argparse
from dotenv import load_dotenv
from websocket.websocketserver import main as ws_server
from http_server_v2 import main as http_server
import uvicorn
import os

app = None

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)

def main():
    parser = argparse.ArgumentParser(
    prog='LinTO-STT',
    description='STT for LinTO with multiple backends')

    parser.add_argument('-m', '--mode', required=True, choices=['http', 'task', 'websocket'])
    parser.add_argument('-b', '--backend', required=False, choices=['nemo', 'whisper', 'kaldi'], default='nemo')
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
    # parser.add_argument('--model', required=False)
    # os.environ.get("STREAMING_PORT", 80)

    args = parser.parse_args()
    if args.mode == 'websocket':
        run_websocket_server(args.backend, args.host, args.port )

    elif args.mode == 'http':
        run_http_server(args.backend, args.host, args.port, args.workers)


def run_http_server(backend, host, port, workers):
    if backend == 'nemo':
        load_dotenv("nemo/.envdefault")
        load_dotenv(".env")
        from nemo_backend.stt.processing import MODEL, USE_GPU, decode, load_wave_buffer, warmup    

    app = http_server(MODEL, USE_GPU, decode, load_wave_buffer, warmup)
    uvicorn.run("main:app", host=host, port=port, workers=workers)


def run_websocket_server(backend, host, port):
    if backend == 'nemo':
        load_dotenv("nemo/.envdefault")
        load_dotenv(".env")
        from nemo_backend.stt.processing import MODEL
        from nemo_backend.stt.processing.streaming import wssDecode
    
    
    ws_server(host, port, wssDecode, MODEL)