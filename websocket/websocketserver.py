import asyncio
import logging
import os
import uuid

import websockets

from stt.processing import MODEL
from stt.processing.streaming import wssDecode

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)


async def _fun_wrapper(ws):
    """Wrap wssDecode function to add STT Model reference"""
    connection_id = str(uuid.uuid4())
    logger.info(f"[{connection_id}] Client connected: {ws.remote_address}")
    try:
        await wssDecode(ws, MODEL, connection_id=connection_id)
    except Exception as e:
        logger.error(f"[{connection_id}] Error in connection handler: {e}", exc_info=True)
    finally:
        logger.info(f"[{connection_id}] Client disconnected: {ws.remote_address}")


async def main():
    """Launch the websocket server"""
    serving_port = int(os.environ.get("STREAMING_PORT", 8001))
    logger.info(f"Starting websocket server on port {serving_port}")
    stop = asyncio.Future()
    async with websockets.serve(
        _fun_wrapper, "0.0.0.0", serving_port, ping_interval=None, ping_timeout=None
    ):
        await stop
    logger.info("Websocket server stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down gracefully.")
