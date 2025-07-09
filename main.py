import logging
import argparse
from dotenv import load_dotenv
from websocket.websocketserver import main as wsserver

def main():
    parser = argparse.ArgumentParser(
    prog='LinTO-STT',
    description='STT for LinTO with multiple backends')

    parser.add_argument('-m', '--mode', required=True, choices=['http', 'task', 'websocket'])
    parser.add_argument('-b', '--backend', required=False, choices=['nemo', 'whisper', 'kaldi'], default='nemo')
    parser.add_argument('-p', '--port', required=False, default=8080, type=int)
    # parser.add_argument('--model', required=False)
    # os.environ.get("STREAMING_PORT", 80)

    args = parser.parse_args()
    if args.mode == 'websocket':
        websocket_server(args.backend, args.port)

def websocket_server(backend, port):  
    if backend == 'nemo':
        load_dotenv("nemo/.envdefault")
        from nemo_backend.stt.processing import MODEL
        from nemo_backend.stt.processing.streaming import wssDecode
    
    load_dotenv(".env")
    wsserver(port, wssDecode, MODEL)

if __name__ == "__main__":
    main()