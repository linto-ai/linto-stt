import json
import logging

import requests

logger = logging.getLogger(__name__)


def transcribe_http(base_url: str, audio_path: str, language: str = None,
                    timeout: float = 300) -> str:
    """POST an audio file to /transcribe and return the transcription text.

    The first request can be slow: with lazy-loaded engines (e.g. NeMo on
    http+cpu, which skips startup warmup) it triggers the model load + first
    decode on CPU. The default read timeout therefore matches the server
    startup budget (--server-timeout, default 600s) rather than a short value.
    """
    url = f"{base_url}/transcribe"
    params = {}
    if language:
        params["language"] = language

    logger.info(
        f"Transcribing {audio_path.split('/')[-1]} via HTTP POST {url} "
        f"(the first request triggers the lazy model load, so it may be slow)..."
    )
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.split("/")[-1], f, "audio/wav")}
        resp = requests.post(
            url,
            files=files,
            headers={"accept": "application/json"},
            params=params,
            timeout=timeout,
        )
    logger.info(f"HTTP transcription response received ({resp.status_code})")

    if resp.status_code != 200:
        raise RuntimeError(f"Transcription failed ({resp.status_code}): {resp.text}")

    # FastAPI returns (json_string, status_code) as a JSON array,
    # or sometimes a raw JSON string.
    try:
        data = json.loads(resp.text)
        # Unwrap [body, status_code] tuple from FastAPI
        if isinstance(data, list) and len(data) == 2:
            data = data[0]
        # Body might be a JSON-encoded string
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, dict):
            return data.get("text", str(data))
        return str(data)
    except (json.JSONDecodeError, TypeError):
        return resp.text


def transcribe_websocket(ws_url: str, audio_path: str, language: str = None,
                         sample_rate: int = 16000, chunk_duration: float = 1.0,
                         timeout: float = 60) -> str:
    """Stream a WAV file over the WebSocket protocol and return the final text.

    Protocol (see linto_stt/.../streaming.py): send a `{"config": ...}` message,
    then raw 16-bit PCM audio chunks, then `{"eof": 1}` to flush. The server
    replies with `{"partial": "..."}` updates and `{"text": "..."}` finals; on
    `eof` it sends the assembled final and closes the connection.

    The server emits one `{"text": ...}` per committed segment (not a single
    cumulative one), so we accumulate every final and join them to reconstruct
    the whole transcription. Returned text has its whitespace collapsed.
    """
    import re
    import time
    import wave
    import websockets.sync.client as ws_client
    from websockets.exceptions import ConnectionClosed

    config = {"config": {"sample_rate": sample_rate}}
    if language:
        config["config"]["language"] = language

    # Read raw PCM frames (skip the WAV header) so we send only audio samples.
    with wave.open(audio_path, "rb") as w:
        audio_data = w.readframes(w.getnframes())

    # Chunk size in bytes (16-bit PCM = 2 bytes per sample).
    chunk_size = int(sample_rate * chunk_duration * 2)
    finals = []

    def _absorb(msg):
        data = json.loads(msg)
        if data.get("text"):
            logger.info(f"Streaming final segment: {data['text']!r}")
            finals.append(data["text"])

    logger.info(
        f"Streaming {audio_path.split('/')[-1]} to {ws_url} "
        f"(first audio triggers the lazy model load, so it may be slow)..."
    )
    with ws_client.connect(ws_url, open_timeout=timeout) as conn:
        conn.send(json.dumps(config))

        offset = 0
        while offset < len(audio_data):
            conn.send(audio_data[offset:offset + chunk_size])
            offset += chunk_size
            try:
                _absorb(conn.recv(timeout=0.1))
            except TimeoutError:
                pass

        # Signal end of stream so the server flushes the final and closes.
        conn.send(json.dumps({"eof": 1}))

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                _absorb(conn.recv(timeout=2.0))
            except TimeoutError:
                continue
            except ConnectionClosed:
                break

    logger.info(f"Streaming finished ({len(finals)} final segment(s) received)")
    return re.sub(r"\s+", " ", " ".join(finals)).strip()


def transcribe_celery(audio_filename: str, language: str = None,
                      broker_url: str = "redis://localhost:6379",
                      queue: str = "stt", timeout: float = 300) -> str:
    """Send a Celery transcription task and wait for the result.

    `queue` must match the worker's queue (its SERVICE_NAME, default "stt").
    Without it the task goes to the default "celery" queue, which the worker
    does not consume, and result.get() times out.

    `timeout` must cover the worker's lazy model load on the first task (NeMo
    loads on first use, not at startup), so it matches the server budget rather
    than a short value.
    """
    from celery import Celery

    app = Celery(
        "linto_stt",
        broker=f"{broker_url}/0",
        backend=f"{broker_url}/1",
    )

    args = [audio_filename, True]
    if language:
        args.append(language)
    else:
        args.append("fr")

    logger.info(
        f"Sending Celery 'transcribe_task' for {audio_filename} to {broker_url} "
        f"(first task triggers the lazy model load, so it may be slow)..."
    )
    result = app.send_task("transcribe_task", args=args, queue=queue)
    output = result.get(timeout=timeout)
    logger.info("Celery task result received")

    if isinstance(output, dict):
        return output.get("text", str(output))
    return str(output)
