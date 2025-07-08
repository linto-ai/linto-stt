import asyncio
import json
import logging
import os

import msgpack
import numpy as np
import websockets
from websockets.legacy.server import WebSocketServerProtocol

from . import KYUTAI_API_KEY, KYUTAI_URL
from .utils import SAMPLE_RATE

logger = logging.getLogger(__name__)


async def forward_client(ws_client: WebSocketServerProtocol, ws_server, connection_id: str):
    """Forward audio from LinTO client to Kyutai server"""
    # first message from client contains config
    res = await ws_client.recv()
    try:
        config = json.loads(res)["config"]
        sr = int(config.get("sample_rate", 16000))
    except Exception:
        logger.error(f"[{connection_id}] Invalid config from client")
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


async def forward_server(ws_server, ws_client: WebSocketServerProtocol, connection_id: str):
    transcript = []
    timer_task = None
    final_transcript_delay = float(os.environ.get("FINAL_TRANSCRIPT_DELAY", 1.5))
    log_transcripts = os.environ.get("LOG_TRANSCRIPTS", "false").lower() == "true"

    async def send_final_transcript():
        nonlocal transcript
        if transcript:
            full_text = " ".join(transcript)
            if log_transcripts:
                logger.info(f"[{connection_id}] Final transcript: {full_text}")
            await ws_client.send(json.dumps({"text": full_text}))
            transcript = []

    def on_timer_done(task: asyncio.Task):
        """Callback executed when the timer task is done."""
        if not task.cancelled():
            asyncio.create_task(send_final_transcript())

    try:
        while True:
            message = await ws_server.recv()
            data = msgpack.unpackb(message, raw=False)

            if data.get("type") == "Word":
                # A new word has arrived, cancel any pending final-transcript timer.
                if timer_task:
                    timer_task.cancel()

                transcript.append(data["text"])
                await ws_client.send(json.dumps({"partial": " ".join(transcript)}))

                # If the word ends a sentence, start a new timer.
                if any(data["text"].endswith(p) for p in [".", "?", "!", "..."]):
                    timer_task = asyncio.create_task(asyncio.sleep(final_transcript_delay))
                    timer_task.add_done_callback(on_timer_done)

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"[{connection_id}] Connection closed by peer.")
    finally:
        if timer_task:
            timer_task.cancel()
        # Send any remaining text as a final transcript
        if transcript:
            await send_final_transcript()


async def wssDecode(ws: WebSocketServerProtocol, _model, connection_id: str):
    url = f"{KYUTAI_URL}/api/asr-streaming"
    headers = {"kyutai-api-key": KYUTAI_API_KEY}
    logger.info(f"[{connection_id}] Attempting to connect to backend at {url}")
    send_task = None
    recv_task = None
    try:
        async with websockets.connect(url, additional_headers=headers) as ws_server:
            logger.info(f"[{connection_id}] Successfully connected to backend at {url}")
            send_task = asyncio.create_task(forward_client(ws, ws_server, connection_id))
            recv_task = asyncio.create_task(forward_server(ws_server, ws, connection_id))
            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
    except Exception as e:
        logger.error(
            f"[{connection_id}] Failed to connect or communicate with backend at {url}: {e}",
            exc_info=True
        )
    finally:
        if send_task:
            send_task.cancel()
        if recv_task:
            recv_task.cancel()
        logger.info(f"[{connection_id}] Connection closed.")
