import argparse
import asyncio
import websockets

async def _linstt_streaming(port):
    ws_api = f"ws://localhost:{port}"
    async with websockets.connect(ws_api, ping_interval=None, ping_timeout=None) as websocket:
        await websocket.close()
        exit(0)
    exit(1)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test the streaming STT")
    parser.add_argument("port", type=int, default=80, help="Port to connect to")
    args = parser.parse_args()

    asyncio.run(_linstt_streaming(args.port))