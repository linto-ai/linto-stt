"""
This module is designed to implement file-based transcription with the Moshi server.
It connects to the server, sends a pre-recorded audio file, and returns the transcript.
/!\ Not tested /!\
"""
import asyncio

import msgpack
import numpy as np
import websockets

from . import KYUTAI_API_KEY, KYUTAI_URL, logger
from .utils import SAMPLE_RATE, resample_audio


async def _decode_audio(audio: np.ndarray) -> str:
    url = f"{KYUTAI_URL}/api/asr-streaming"
    headers = {"kyutai-api-key": KYUTAI_API_KEY}
    async with websockets.connect(url, extra_headers=headers) as ws:
        # Warm-up phase: send 2 seconds of silence
        await ws.send(
            msgpack.packb({"type": "Audio", "pcm": [0.0] * SAMPLE_RATE}, use_single_float=True)
        )
        await ws.send(
            msgpack.packb({"type": "Audio", "pcm": [0.0] * SAMPLE_RATE}, use_single_float=True)
        )

        # send audio in chunks
        frame = 1920
        for i in range(0, len(audio), frame):
            chunk = audio[i : i + frame]
            await ws.send(
                msgpack.packb({"type": "Audio", "pcm": chunk.tolist()}, use_single_float=True)
            )
        # send marker and some silence
        await ws.send(msgpack.packb({"type": "Marker", "id": 0}, use_single_float=True))
        for _ in range(10):
            await ws.send(
                msgpack.packb({"type": "Audio", "pcm": [0.0] * SAMPLE_RATE}, use_single_float=True)
            )
        transcript = []
        async for message in ws:
            data = msgpack.unpackb(message, raw=False)
            if data["type"] == "Word":
                transcript.append(data["text"])
            if data["type"] == "Marker":
                break
        return " ".join(transcript)


def decode_audio(audio):
    data, sr = audio
    data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    data = resample_audio(data, sr)
    data /= 32768.0
    text = asyncio.run(_decode_audio(data))
    return {"text": text.strip()}
