import asyncio
import json

import msgpack
import numpy as np
import websockets

from . import KYUTAI_API_KEY, KYUTAI_URL, logger
from .utils import SAMPLE_RATE


async def _decode_audio(audio: np.ndarray) -> str:
    url = f"{KYUTAI_URL}/api/asr-streaming"
    headers = {"kyutai-api-key": KYUTAI_API_KEY}
    async with websockets.connect(url, extra_headers=headers) as ws:
        # send initial second of silence
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
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly

        gcd = np.gcd(sr, SAMPLE_RATE)
        data = resample_poly(
            np.frombuffer(data, dtype=np.int16).astype(np.float32), SAMPLE_RATE // gcd, sr // gcd
        )
    else:
        data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    data /= 32768.0
    text = asyncio.run(_decode_audio(data))
    return {"text": text.strip()}