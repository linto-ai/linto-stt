import json
import re
from typing import Union

from simple_websocket.ws import Server as WSServer
from stt import logger
from vosk import KaldiRecognizer, Model
from websockets.legacy.server import WebSocketServerProtocol

from punctuation.recasepunc import apply_recasepunc

EOF_REGEX = re.compile(r' *\{.*"eof" *: *1.*\} *$')

async def wssDecode(ws: WebSocketServerProtocol, model: Model):
    """Async Decode function endpoint"""
    # Wait for config
    res = await ws.recv()

    model, punctuation_model = model

    # Parse config
    try:
        config = json.loads(res)["config"]
        sample_rate = config["sample_rate"]
    except Exception as e:
        logger.error("Failed to read stream configuration")
        await ws.close(reason="Failed to load configuration")
    # Recognizer
    try:
        recognizer = KaldiRecognizer(model, sample_rate)
    except Exception as e:
        logger.error("Failed to load recognizer")
        await ws.close(reason="Failed to load recognizer")

    # Wait for chunks
    while True:
        try:
            # Client data
            message = await ws.recv()
            if message is None or message == "":  # Timeout
                ws.close()
        except Exception as e:
            print("Connection closed by client: {}".format(str(e)))
            break

        # End frame
        if (isinstance(message, str) and re.match(EOF_REGEX, message)):
            ret = recognizer.FinalResult()
            ret = apply_recasepunc(punctuation_model, ret)
            await ws.send(json.dumps(ret))
            await ws.close(reason="End of stream")
            break

        # Audio chunk
        if recognizer.AcceptWaveform(message):
            ret = recognizer.Result()  # Result seems to not work properly
            ret = apply_recasepunc(punctuation_model, ret)
            await ws.send(ret)

        else:
            ret = recognizer.PartialResult()
            last_utterance = ret
            await ws.send(ret)


def ws_streaming(websocket_server: WSServer, model: Model):
    """Sync Decode function endpoint"""
    # Wait for config
    res = websocket_server.receive(timeout=10)

    model, punctuation_model = model

    # Timeout
    if res is None:
        pass

    # Parse config
    try:
        config = json.loads(res)["config"]
        sample_rate = config["sample_rate"]
    except Exception:
        logger.error("Failed to read stream configuration")
        websocket_server.close()

    # Recognizer
    try:
        recognizer = KaldiRecognizer(model, sample_rate)
    except Exception:
        logger.error("Failed to load recognizer")
        websocket_server.close()
    # Wait for chunks
    while True:
        try:
            # Client data
            message = websocket_server.receive(timeout=10)
            if message is None:  # Timeout
                websocket_server.close()
        except Exception:
            print("Connection closed by client")
            break
        # End frame
        if (isinstance(message, str) and re.match(EOF_REGEX, message)):
            ret = recognizer.FinalResult()
            ret = apply_recasepunc(punctuation_model, ret)
            websocket_server.send(json.dumps(re.sub("<unk> ", "", ret)))
            websocket_server.close()
            break
        # Audio chunk
        print("Received chunk")
        if recognizer.AcceptWaveform(message):
            ret = recognizer.Result()
            ret = apply_recasepunc(punctuation_model, ret)
            websocket_server.send(re.sub("<unk> ", "", ret))

        else:
            ret = recognizer.PartialResult()
            websocket_server.send(re.sub("<unk> ", "", ret))
