async def wssDecode(ws, model):
    """WebSocket streaming is not supported for the hf engine (the server
    refuses to start in websocket mode, see linto_stt.main; this stub only keeps
    the engine contract).

    Streaming would need vLLM's realtime API, which is not wired here (the vllm
    backend is used for offline transcription only). Run the engine in `http` or
    `task` mode instead.
    """
    raise NotImplementedError(
        "WebSocket streaming is not supported for the hf engine "
        "(use http or task mode)."
    )
