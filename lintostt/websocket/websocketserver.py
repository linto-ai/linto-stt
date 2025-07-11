import asyncio
import os

import websockets


def main(host, serving_port: int, wssDecode, MODEL):
    

    async def _fun_wrapper(ws):
        """Wrap wssDecode function to add STT Model reference"""
        return await wssDecode(ws, MODEL)


    async def WSServer(host, port: int):
        """Launch the websocket server"""
        async with websockets.serve(_fun_wrapper, host, port, ping_interval=None, ping_timeout=None):
            await asyncio.Future()

    asyncio.run(WSServer(host, serving_port))
