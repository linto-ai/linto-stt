import json
import re
from typing import Union

from websockets.legacy.server import WebSocketServerProtocol
from simple_websocket.ws import Server as WSServer
from vosk import KaldiRecognizer, Model

from stt import logger 

async def wssDecode(ws: WebSocketServerProtocol, model: Model):
    """ Async Decode function endpoint """
    # Wait for config
    res = await ws.recv()
    
    # Parse config
    try:
        config = json.loads(res)["config"]
        sample_rate = config["sample_rate"]
    except Exception as e :
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
            if message is None or message == "": # Timeout
                ws.close()
        except Exception as e:
            print("Connection closed by client: {}".format(str(e)))
            break
        
        # End frame
        if "eof" in str(message):
            ret = recognizer.FinalResult()
            await ws.send(json.dumps(ret))
            await ws.close(reason="End of stream")
            break

        # Audio chunk
        if recognizer.AcceptWaveform(message):
            ret = recognizer.Result() # Result seems to not work properly
            await ws.send(ret)
            
        else:
            ret = recognizer.PartialResult()
            last_utterance = ret
            await ws.send(ret)

def ws_streaming(ws: WSServer, model: Model):
    """ Sync Decode function endpoint"""
    # Wait for config
    res = ws.receive(timeout=10)

    # Timeout
    if res is None:
        pass

    # Parse config
    try:
        config = json.loads(res)["config"]
        sample_rate = config["sample_rate"]
    except Exception as e :
        logger.error("Failed to read stream configuration")
        ws.close()

    # Recognizer
    try:
        recognizer = KaldiRecognizer(model, sample_rate)
    except Exception as e:
        logger.error("Failed to load recognizer")
        ws.close()

    # Wait for chunks
    while True: 
        try:
            # Client data
            message = ws.receive(timeout=10)
            if message is None: # Timeout
                ws.close()
        except Exception:
            print("Connection closed by client")
            break
        # End frame
        if "eof" in str(message):
            ret = recognizer.FinalResult()
            ws.send(json.dumps(re.sub("<unk> ", "", ret)))
            ws.close()
            break    
        # Audio chunk
        print("Received chunk")
        if recognizer.AcceptWaveform(message):
            ret = recognizer.Result()
            ws.send(re.sub("<unk> ", "", ret))
            
        else:
            ret = recognizer.PartialResult()
            ws.send(re.sub("<unk> ", "", ret))