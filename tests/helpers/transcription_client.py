import json
import logging

import requests

logger = logging.getLogger(__name__)


def transcribe_http(base_url: str, audio_path: str, language: str = None) -> str:
    """POST an audio file to /transcribe and return the transcription text."""
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
            timeout=120,
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
                         sample_rate: int = 16000, chunk_duration: float = 1.0) -> str:
    """Send audio over WebSocket and return the final transcription."""
    import websockets.sync.client as ws_client

    # Build config message
    config = {"config": {"sample_rate": sample_rate}}
    if language:
        config["config"]["language"] = language

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    # Calculate chunk size in bytes (16-bit PCM = 2 bytes per sample)
    chunk_size = int(sample_rate * chunk_duration * 2)
    final_text = ""

    with ws_client.connect(ws_url) as conn:
        # Send config
        conn.send(json.dumps(config))

        # Send audio in chunks
        offset = 0
        while offset < len(audio_data):
            chunk = audio_data[offset:offset + chunk_size]
            conn.send(chunk)
            offset += chunk_size

            # Check for responses
            try:
                msg = conn.recv(timeout=0.1)
                data = json.loads(msg)
                if "text" in data:
                    final_text = data["text"]
            except TimeoutError:
                pass

        # Send empty bytes to signal end of stream
        conn.send(b"")

        # Collect remaining responses
        deadline = __import__("time").time() + 30
        while __import__("time").time() < deadline:
            try:
                msg = conn.recv(timeout=2.0)
                data = json.loads(msg)
                if "text" in data:
                    final_text = data["text"]
            except (TimeoutError, Exception):
                break

    return final_text


def transcribe_celery(audio_filename: str, language: str = None,
                      broker_url: str = "redis://localhost:6379") -> str:
    """Send a Celery transcription task and wait for the result."""
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

    result = app.send_task("transcribe_task", args=args)
    output = result.get(timeout=120)

    if isinstance(output, dict):
        return output.get("text", str(output))
    return str(output)
