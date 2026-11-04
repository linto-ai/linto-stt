async def wssDecode(ws, model):
    """WebSocket streaming is not supported for the transformers engine.

    Qwen3-ASR streaming is only available through the vLLM backend, which this
    engine does not use. Run the engine in `http` or `task` mode instead.
    """
    raise NotImplementedError(
        "WebSocket streaming is not supported for the transformers engine "
        "(use http or task mode)."
    )
