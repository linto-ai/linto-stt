"""
This module acts as an adapter between a LinTO-like streaming client and
the Kyutai/Moshi ASR server. It handles WebSocket connections,
forwards audio streams, and relays transcription results back to the client.

The module now properly utilizes the Moshi server's semantic Voice Activity Detection (VAD)
to intelligently detect utterance boundaries, combining probabilistic VAD signals
with punctuation-based heuristics for robust end-of-utterance detection.
"""
import asyncio
import json
import logging
import os
from typing import List, Optional

import msgpack
import numpy as np
import websockets
from websockets.legacy.server import WebSocketServerProtocol

from . import KYUTAI_API_KEY, KYUTAI_URL
from .utils import SAMPLE_RATE, resample_audio

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SemanticVADTracker:
    """
    Tracks semantic VAD probabilities from Moshi server Step messages.

    The Moshi 1B model provides 4 extra heads with 6-dimensional outputs
    representing different semantic states. Based on the model architecture,
    these likely include end-of-utterance predictions.
    """

    def __init__(
        self,
        vad_threshold: float = 0.5,
        vad_history_size: int = 3,
        log_vad: bool = False
    ):
        """
        Initialize the VAD tracker.

        Args:
            vad_threshold: Probability threshold for detecting end of utterance (0.0-1.0)
            vad_history_size: Number of consecutive high VAD signals needed for confirmation
            log_vad: Whether to log VAD probabilities for debugging
        """
        self.vad_threshold = vad_threshold
        self.vad_history_size = vad_history_size
        self.log_vad = log_vad
        self.vad_history: List[float] = []
        self.last_step_idx: Optional[int] = None

    def update(self, step_idx: int, prs: List[float]) -> bool:
        """
        Process a Step message and determine if end-of-utterance is detected.

        Args:
            step_idx: The model step index
            prs: Probability distributions from extra heads

        Returns:
            True if end-of-utterance is detected, False otherwise
        """
        self.last_step_idx = step_idx
        logger.debug(f"Processing Step {step_idx} with {prs}")
        if not prs:
            return False

        # The first extra head's last dimension typically represents end-of-utterance probability
        # We'll use the maximum probability across all heads as a conservative approach
        max_vad_prob = prs[0] if prs else 0.0

        # if self.log_vad:
        #     logger.debug(
        #         f"Step {step_idx}: VAD probabilities = {prs[0][:3] if prs else []}, max = {max_vad_prob:.3f}")

        # Track VAD history for smoothing
        self.vad_history.append(max_vad_prob)
        if len(self.vad_history) > self.vad_history_size:
            self.vad_history.pop(0)

        # Detect end-of-utterance if recent VAD probabilities are consistently high
        if len(self.vad_history) >= self.vad_history_size:
            avg_vad = sum(self.vad_history) / len(self.vad_history)
            if avg_vad > self.vad_threshold:
                if self.log_vad:
                    logger.info(
                        f"Step {step_idx}: VAD triggered! avg={avg_vad:.3f}, threshold={self.vad_threshold}")
                return True

        return False

    def reset(self):
        """Reset VAD history."""
        self.vad_history.clear()


async def forward_client(ws_client: WebSocketServerProtocol, ws_server):
    """Forward audio from LinTO client to Kyutai server"""
    client_addr = ws_client.remote_address
    logger.info(f"[{client_addr}] forward_client started")

    # first message from client contains config
    try:
        res = await ws_client.recv()
    except websockets.exceptions.ConnectionClosed as e:
        logger.warning(
            f"[{client_addr}] Client disconnected before sending config: code={e.code} reason={e.reason}")
        return

    try:
        config = json.loads(res)["config"]
        sr = int(config.get("sample_rate", 16000))
        logger.info(
            f"[{client_addr}] Client config received: sample_rate={sr}")
    except Exception as e:
        logger.error(f"[{client_addr}] Invalid config from client: {e}")
        await ws_client.close(code=1003, reason="Invalid config")
        return

    # Warm-up phase: send a single silent packet to start the stream,
    # then wait for 2 seconds while discarding any incoming audio.
    await ws_server.send(
        msgpack.packb({"type": "Audio", "pcm": [0.0]}, use_single_float=True)
    )
    logger.info(f"[{client_addr}] Warm-up phase started")
    warmup_end_time = asyncio.get_event_loop().time() + 2.0

    while asyncio.get_event_loop().time() < warmup_end_time:
        try:
            # Discard incoming audio until warmup complete.
            _ = await asyncio.wait_for(ws_client.recv(), timeout=0.05)
        except asyncio.TimeoutError:
            # No message received, just wait
            await asyncio.sleep(0.05)
    logger.info(f"[{client_addr}] Warm-up phase ended")
    while True:
        message = await ws_client.recv()
        if isinstance(message, str) and message.strip().startswith("{"):
            try:
                if json.loads(message).get("eof"):
                    logger.debug(f"[{client_addr}] EOF received")
                    break
            except Exception:
                continue
        audio = np.frombuffer(message, dtype=np.int16).astype(np.float32)
        audio = resample_audio(audio, sr)
        audio /= 32768.0
        await ws_server.send(
            msgpack.packb({"type": "Audio", "pcm": audio.tolist()},
                          use_single_float=True)
        )
    await ws_server.send(msgpack.packb({"type": "Marker", "id": 0}, use_single_float=True))
    for _ in range(10):
        await ws_server.send(
            msgpack.packb({"type": "Audio", "pcm": [
                          0.0] * SAMPLE_RATE}, use_single_float=True)
        )


async def forward_server(ws_server, ws_client: WebSocketServerProtocol):
    """
    Forward transcription results from Kyutai server to LinTO client.

    This function now processes both Word and Step messages to utilize
    semantic VAD for intelligent utterance boundary detection.
    """
    transcript = []
    timer_task = None

    # Configuration from environment variables
    final_transcript_delay = float(
        os.environ.get("FINAL_TRANSCRIPT_DELAY", 1.5))
    log_transcripts = os.environ.get(
        "LOG_TRANSCRIPTS", "false").lower() == "true"

    # Semantic VAD configuration
    use_semantic_vad = os.environ.get(
        "USE_SEMANTIC_VAD", "true").lower() == "true"
    vad_threshold = float(os.environ.get("VAD_THRESHOLD", "0.5"))
    vad_history_size = int(os.environ.get("VAD_HISTORY_SIZE", "3"))
    log_vad = os.environ.get("LOG_VAD", "false").lower() == "true"
    # Delay after VAD trigger before sending final
    vad_delay = float(os.environ.get("VAD_DELAY", "0.3"))

    # Combine VAD with punctuation or use independently
    vad_require_punctuation = os.environ.get(
        "VAD_REQUIRE_PUNCTUATION", "true").lower() == "true"

    # Initialize VAD tracker
    vad_tracker = SemanticVADTracker(
        vad_threshold=vad_threshold,
        vad_history_size=vad_history_size,
        log_vad=log_vad
    ) if use_semantic_vad else None

    # Track state for VAD-based finalization
    has_punctuation = False
    vad_triggered = False

    logger.info(
        f"VAD Configuration: enabled={use_semantic_vad}, threshold={vad_threshold}, "
        f"history_size={vad_history_size}, require_punctuation={vad_require_punctuation}, "
        f"vad_delay={vad_delay}s, timer_delay={final_transcript_delay}s"
    )

    async def send_final_transcript(reason: str = "unknown"):
        """Send the final transcript and reset state."""
        nonlocal transcript, has_punctuation, vad_triggered
        if transcript:
            full_text = " ".join(transcript)
            if log_transcripts:
                logger.info(f"Final transcript ({reason}): {full_text}")
            await ws_client.send(json.dumps({"text": full_text}))
            transcript = []
            has_punctuation = False
            vad_triggered = False
            if vad_tracker:
                vad_tracker.reset()

    def on_timer_done(task: asyncio.Task):
        """Callback executed when the timer task is done."""
        if not task.cancelled():
            asyncio.create_task(send_final_transcript("timer"))

    try:
        while True:
            message = await ws_server.recv()
            data = msgpack.unpackb(message, raw=False)
            logger.debug(f"Received data: {data}")
            msg_type = data.get("type")

            if msg_type == "Step":
                # Process semantic VAD information
                if vad_tracker:
                    step_idx = data.get("step_idx", 0)
                    prs = data.get("prs", [])

                    # Check if VAD detects end of utterance
                    if vad_tracker.update(step_idx, prs):
                        vad_triggered = True

                        # Decide whether to finalize based on configuration
                        should_finalize = False

                        if vad_require_punctuation:
                            # Only finalize if we have both VAD signal and punctuation
                            if has_punctuation and transcript:
                                should_finalize = True
                                reason = "vad+punctuation"
                        else:
                            # Finalize on VAD signal alone if we have transcript
                            if transcript:
                                should_finalize = True
                                reason = "vad"

                        if should_finalize:
                            # Cancel any existing timer
                            if timer_task:
                                timer_task.cancel()

                            # Use shorter delay for VAD-triggered finals
                            timer_task = asyncio.create_task(
                                asyncio.sleep(vad_delay))
                            timer_task.add_done_callback(lambda t: asyncio.create_task(
                                send_final_transcript(reason)) if not t.cancelled() else None)

            elif msg_type == "Word":
                # A new word has arrived, cancel any pending final-transcript timer.
                if timer_task:
                    timer_task.cancel()
                    timer_task = None

                word_text = data["text"]
                transcript.append(word_text)
                await ws_client.send(json.dumps({"partial": " ".join(transcript)}))

                # Check if word has sentence-ending punctuation
                if any(word_text.endswith(p) for p in [".", "?", "!", "..."]):
                    has_punctuation = True

                    if use_semantic_vad:
                        # With VAD enabled: wait for VAD confirmation or fallback to timer
                        if vad_triggered:
                            # VAD already triggered, finalize quickly
                            timer_task = asyncio.create_task(
                                asyncio.sleep(vad_delay))
                            timer_task.add_done_callback(lambda t: asyncio.create_task(
                                send_final_transcript("vad+punctuation")) if not t.cancelled() else None)
                        else:
                            # Wait for VAD signal, but use timer as fallback
                            timer_task = asyncio.create_task(
                                asyncio.sleep(final_transcript_delay))
                            timer_task.add_done_callback(on_timer_done)
                    else:
                        # Without VAD: use original timer-based approach
                        timer_task = asyncio.create_task(
                            asyncio.sleep(final_transcript_delay))
                        timer_task.add_done_callback(on_timer_done)

            elif msg_type == "EndWord":
                # EndWord provides timing information, currently not used
                pass

            elif msg_type == "Marker":
                # Marker indicates server has processed client's marker
                pass

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Connection closed by peer.")
    except Exception as e:
        logger.exception("Unexpected error occurred")
        # logger.error(f"Unexpected error: {type(e).__name__}: {e}")
        raise  # Re-raise pour ne pas masquer l'erreur
    finally:
        if timer_task:
            timer_task.cancel()
        # Send any remaining text as a final transcript
        if transcript:
            await send_final_transcript("connection_closed")


async def wssDecode(ws: WebSocketServerProtocol, _model):
    url = f"{KYUTAI_URL}/api/asr-streaming"
    headers = {"kyutai-api-key": KYUTAI_API_KEY}
    logger.info(f"Attempting to connect to backend at {url}")
    send_task = None
    recv_task = None
    try:
        async with websockets.connect(url, additional_headers=headers) as ws_server:
            logger.info(f"Successfully connected to backend at {url}")
            send_task = asyncio.create_task(forward_client(ws, ws_server))
            recv_task = asyncio.create_task(forward_server(ws_server, ws))
            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
    except Exception as e:
        logger.error(
            f"Failed to connect or communicate with backend at {url}: {e}",
            exc_info=True
        )
    finally:
        if send_task:
            send_task.cancel()
        if recv_task:
            recv_task.cancel()
        logger.info(f"Connection closed.")
