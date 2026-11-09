import torch

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "auto": "auto",
}


def load_asr_model(model_name, device, dtype="bfloat16", attn_implementation=None):
    """Load a transformers-native ASR processor + model, by checkpoint model type.

    - ``qwen3_asr`` (e.g. Qwen/Qwen3-ASR-1.7B-hf): an LLM-style model loaded via
      Qwen3ASRForConditionalGeneration (apply_transcription_request protocol).
    - any other type (e.g. ``cohere_asr``): a Whisper-like speech-to-text
      encoder-decoder, loaded via AutoModelForSpeechSeq2Seq (built-in support,
      no remote code).

    The backend (see backends.py) auto-detects which protocol to use from the
    processor, so new transformers-ASR families can be added here without changing
    the engine contract.
    """
    from transformers import AutoConfig, AutoProcessor

    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type", "")
    torch_dtype = _DTYPES.get(dtype, torch.bfloat16)
    device_map = "auto" if "cuda" in str(device) else str(device)

    processor = AutoProcessor.from_pretrained(model_name)

    if model_type == "qwen3_asr":
        # Qwen3ASRForConditionalGeneration is the concrete class; fall back to the
        # generic multimodal Auto mapping if this transformers only exposes that.
        try:
            from transformers import Qwen3ASRForConditionalGeneration as _ModelCls
        except ImportError:
            from transformers import AutoModelForMultimodalLM as _ModelCls
        kwargs = {"torch_dtype": torch_dtype, "device_map": device_map}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        model = _ModelCls.from_pretrained(model_name, **kwargs).eval()
    else:
        from transformers import AutoModelForSpeechSeq2Seq

        kwargs = {"dtype": torch_dtype, "device_map": device_map}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **kwargs).eval()
    return processor, model


def load_asr_backend(backend, model_name, device, dtype="bfloat16", attn_implementation=None):
    """Build the ASR backend selected by `BACKEND`.

    - ``transformers``: in-process Hugging Face transformers (Qwen3-ASR + Cohere).
    - ``vllm``: a managed ``vllm serve`` subprocess, over HTTP (Qwen3-ASR + Cohere).
    - ``vllm_offline``: in-process ``vllm.LLM`` (Qwen3-ASR only), for benchmarking
      against the ``vllm`` serve backend.

    Returns an object exposing ``transcribe(audio, language, prompt) ->
    (text, detected_language)``. The forced aligner (word timestamps) is loaded
    separately and is the same regardless of backend.
    """
    from .backends import TransformersASRBackend, VllmASRBackend, VllmServeBackend

    if backend == "transformers":
        processor, model = load_asr_model(model_name, device, dtype, attn_implementation)
        return TransformersASRBackend(processor, model)
    if backend == "vllm":
        return VllmServeBackend(model_name, device, dtype=dtype)
    if backend == "vllm_offline":
        return VllmASRBackend(model_name, device, dtype=dtype)
    raise ValueError(
        f"Unknown BACKEND={backend!r} "
        "(expected 'transformers', 'vllm' or 'vllm_offline')"
    )


