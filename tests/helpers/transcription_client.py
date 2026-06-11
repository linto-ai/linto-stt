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

    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.split("/")[-1], f, "audio/wav")}
        resp = requests.post(
            url,
            files=files,
            headers={"accept": "application/json"},
            params=params,
            timeout=timeout,
        )

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
    """
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
    final_text = ""

    def _absorb(msg):
        nonlocal final_text
        data = json.loads(msg)
        if data.get("text"):
            final_text = data["text"]

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

    return final_text


def transcribe_celery(audio_filename: str, language: str = None,
                      broker_url: str = "redis://localhost:6379",
                      queue: str = "stt") -> str:
    """Send a Celery transcription task and wait for the result.

    `queue` must match the worker's queue (its SERVICE_NAME, default "stt").
    Without it the task goes to the default "celery" queue, which the worker
    does not consume, and result.get() times out.
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

    result = app.send_task("transcribe_task", args=args, queue=queue)
    output = result.get(timeout=120)

    if isinstance(output, dict):
        return output.get("text", str(output))
    return str(output)
