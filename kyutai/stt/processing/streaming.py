import asyncio
import json
import logging

import msgpack
import numpy as np
import websockets
from websockets.legacy.server import WebSocketServerProtocol

from . import KYUTAI_API_KEY, KYUTAI_URL, logger
from .utils import SAMPLE_RATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def forward_client(ws_client: WebSocketServerProtocol, ws_server):
    """Forward audio from LinTO client to Kyutai server"""
    # first message from client contains config
    res = await ws_client.recv()
    try:
        config = json.loads(res)["config"]
        sr = int(config.get("sample_rate", 16000))
    except Exception:
        logging.error("Invalid config from client")
        await ws_client.close(code=1003, reason="Invalid config")
        return

    await ws_server.send(
        msgpack.packb({"type": "Audio", "pcm": [0.0] * SAMPLE_RATE}, use_single_float=True)
    )
    while True:
        message = await ws_client.recv()
        if isinstance(message, str) and message.strip().startswith("{"):
            try:
                if json.loads(message).get("eof"):
                    break
            except Exception:
                continue
        audio = np.frombuffer(message, dtype=np.int16).astype(np.float32)
        if sr != SAMPLE_RATE:
            from scipy.signal import resample_poly

            gcd = np.gcd(sr, SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // gcd, sr // gcd)
        audio /= 32768.0
        await ws_server.send(
            msgpack.packb({"type": "Audio", "pcm": audio.tolist()}, use_single_float=True)
        )
    await ws_server.send(msgpack.packb({"type": "Marker", "id": 0}, use_single_float=True))
    for _ in range(10):
        await ws_server.send(
            msgpack.packb({"type": "Audio", "pcm": [0.0] * SAMPLE_RATE}, use_single_float=True)
        )


async def forward_server(ws_server, ws_client: WebSocketServerProtocol):
    transcript = []
    while True:
        try:
            message = await asyncio.wait_for(ws_server.recv(), timeout=2.0)
            data = msgpack.unpackb(message, raw=False)
            if data.get("type") == "Word":
                transcript.append(data["text"])
                await ws_client.send(json.dumps({"partial": " ".join(transcript)}))
        except asyncio.TimeoutError:
            if transcript:
                print(f"DEBUG: Timeout reached. Transcript is: {transcript}")
                if any(transcript[-1].endswith(p) for p in ".?!..."):
                    await ws_client.send(json.dumps({"text": " ".join(transcript)}))
                    transcript = []


async def wssDecode(ws: WebSocketServerProtocol, _model):
    url = f"{KYUTAI_URL}/api/asr-streaming"
    headers = {"kyutai-api-key": KYUTAI_API_KEY}
    logging.info(f"Attempting to connect to backend at {url}")
    send_task = None
    recv_task = None
    try:
        async with websockets.connect(url, additional_headers=headers) as ws_server:
            logging.info(f"Successfully connected to backend at {url}")
            send_task = asyncio.create_task(forward_client(ws, ws_server))
            recv_task = asyncio.create_task(forward_server(ws_server, ws))
            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
    except Exception as e:
        logging.error(f"Failed to connect or communicate with backend at {url}: {e}", exc_info=True)
    finally:
        if send_task:
            send_task.cancel()
        if recv_task:
            recv_task.cancel()
        logging.info("Connection closed.")
