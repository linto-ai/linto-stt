"""Swappable ASR backends for the transformers engine.

The engine runs one pipeline (chunking, forced-alignment word timestamps,
language/hotword handling, response assembly) around a pluggable ASR backend
selected by the ``BACKEND`` environment variable:

- ``transformers`` (default): in-process Hugging Face ``transformers``.
- ``vllm``: in-process vLLM (``vllm.LLM``), for higher throughput.

A backend only has to turn one audio segment into ``(text, detected_language)``;
word timestamps (the forced aligner) and everything else stay backend-agnostic,
so backends can be benchmarked against each other by flipping ``BACKEND``.
"""
import inspect
import os

import torch

from linto_stt.engines.transformers.stt import MAX_NEW_TOKENS, logger

from .utils import SAMPLE_RATE


def detect_capabilities(processor):
    """Detect a model's ASR capabilities from its *processor's* transcription
    signature — not from the model name — so it is robust across checkpoints:

    - ``hotwords``: the transcription call takes a context/hotword ``prompt``.
    - ``language_spec``: it takes a ``language`` argument at all.
    - ``language_auto``: ``language`` is optional (auto-detection) rather than
      required.

    Qwen3-ASR exposes ``apply_transcription_request(audio, language=None,
    prompt=None)``; a seq2seq model (e.g. Cohere) exposes
    ``__call__(audio, language, ...)`` with no ``prompt`` and a required
    ``language``.
    """
    fn = getattr(processor, "apply_transcription_request", None) or processor.__call__
    params = inspect.signature(fn).parameters
    has_language = "language" in params
    language_optional = (
        has_language and params["language"].default is not inspect.Parameter.empty
    )
    return {
        "hotwords": "prompt" in params,
        "language_spec": has_language,
        "language_auto": (not has_language) or language_optional,
    }

# vLLM tuning (env). gpu_memory_utilization is the FRACTION of total VRAM vLLM
# reserves (model + KV cache); keep it modest on a shared GPU. max_model_len caps
# the context window: the model advertises a huge default (65536) whose KV cache
# does not fit a modest memory share, and transcription needs far less, so we cap
# it (raise it for very long audio if VRAM allows).
VLLM_GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.6"))
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384"))
# Optional ABSOLUTE KV-cache budget in bytes. When set, it pins the KV cache size
# directly (vLLM's kv_cache_memory_bytes) instead of deriving it from the
# gpu_memory_utilization fraction — preferable on unified-memory / shared GPUs
# where a relative share is awkward. gpu_memory_utilization still bounds the total
# reservation (weights + activations + KV).
VLLM_KV_CACHE_MEMORY_BYTES = os.environ.get("VLLM_KV_CACHE_MEMORY_BYTES") or None
# Skip torch.compile + CUDA-graph capture. Uses less VRAM and starts almost
# instantly (no ~minute-long compile), at some inference-throughput cost. Useful
# on memory-tight GPUs; off by default so the graphs give best throughput.
VLLM_ENFORCE_EAGER = os.environ.get("VLLM_ENFORCE_EAGER", "").strip().lower() in (
    "1", "true", "yes",
)


class TransformersASRBackend:
    """In-process Hugging Face transformers ASR, for any transformers-native ASR.

    Two model families are supported, auto-detected from the processor:

    - **Qwen3-ASR** (processor exposes ``apply_transcription_request``): an LLM-
      style model. The context ``prompt`` carries hotwords and the language is
      forced via an assistant prefill; long audio is chunked by the engine.
    - **Seq2seq ASR** (e.g. ``cohere_asr``): a Whisper-like encoder-decoder.
      ``processor(audio, language=...)`` -> ``generate`` -> ``processor.decode``.
      It has no hotword/prompt mechanism, requires an explicit language, and
      chunks long audio itself (``handles_long_audio``).
    """

    name = "transformers"

    def __init__(self, processor, model):
        self.processor = processor
        self.model = model
        # Qwen3-ASR exposes apply_transcription_request (chat + hotword prompt);
        # other transformers ASR (e.g. cohere_asr) use the plain seq2seq API.
        self._is_qwen = hasattr(processor, "apply_transcription_request")
        # Seq2seq models (Cohere) split long audio internally, so the engine must
        # NOT also chunk them.
        self.handles_long_audio = not self._is_qwen
        caps = detect_capabilities(processor)
        self.supports_hotwords = caps["hotwords"]
        self.supports_language_spec = caps["language_spec"]
        self.supports_language_auto = caps["language_auto"]

    def transcribe(self, audio, language=None, prompt=None):
        """Transcribe one 16 kHz mono float32 segment -> (text, detected_language)."""
        if self._is_qwen:
            return self._transcribe_qwen(audio, language, prompt)
        return self._transcribe_seq2seq(audio, language)

    def _transcribe_qwen(self, audio, language, prompt):
        request_kwargs = {}
        if language:
            request_kwargs["language"] = language
        if prompt:
            request_kwargs["prompt"] = prompt

        inputs = self.processor.apply_transcription_request(audio=audio, **request_kwargs)
        inputs = inputs.to(self.model.device, self.model.dtype)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        parsed = self.processor.decode(
            output_ids[:, input_len:], return_format="parsed"
        )[0]
        text = (parsed.get("transcription") or "").strip()
        return text, parsed.get("language")

    def _transcribe_seq2seq(self, audio, language):
        # Seq2seq ASR (e.g. Cohere): language is required and there is no hotword
        # mechanism; both are validated upstream (decode) from the capabilities.
        inputs = self.processor(
            audio=audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", language=language
        )
        # The feature extractor splits long audio into chunks; decode reassembles
        # them using audio_chunk_index.
        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs = inputs.to(self.model.device, self.model.dtype)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        text = self.processor.decode(
            output_ids,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=language,
        )[0].strip()
        return text, language


class VllmASRBackend:
    """In-process vLLM ASR (``vllm.LLM``), loaded lazily.

    vLLM uses the non-``-hf`` checkpoint layout (``Qwen/Qwen3-ASR-1.7B``), so an
    ``-hf`` ``MODEL`` is mapped to its non-``-hf`` id. Transcription builds the
    chat/transcription prompt with the HF processor's chat template (system =
    context/hotwords prompt, user = audio, optional assistant prefill
    ``language <Name><asr_text>`` to force the language) and feeds it to
    ``LLM.generate`` with the raw audio as ``multi_modal_data``. The model emits
    ``language <Name><asr_text><transcription>``, which we parse.

    GPU only. Word timestamps still come from the transformers forced aligner.
    """

    name = "vllm"
    handles_long_audio = False  # Qwen via vLLM: the engine chunks long audio
    supports_hotwords = True  # vLLM backend is Qwen3-ASR only (guarded below)
    supports_language_auto = True
    supports_language_spec = True

    def __init__(self, model_name, device, dtype="bfloat16", **kwargs):
        if "cuda" not in str(device):
            raise RuntimeError(
                "The vLLM backend requires a CUDA device (set DEVICE=cuda)."
            )
        # vLLM wants the non-`-hf` checkpoint (the `-hf` repo targets transformers).
        self.vllm_model = model_name.removesuffix("-hf")

        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams
        self.processor = AutoProcessor.from_pretrained(self.vllm_model)
        # The vLLM path builds Qwen3-ASR's chat/transcription prompt, so it only
        # fits that family. Other transformers ASR (e.g. the Cohere seq2seq model,
        # which vLLM runs as an encoder-decoder) need a different prompt API that
        # is not wired here — fail clearly rather than emit a wrong prompt.
        if not hasattr(self.processor, "apply_transcription_request"):
            raise NotImplementedError(
                f"The vLLM backend currently supports only the Qwen3-ASR family; "
                f"{model_name!r} is not supported. Use BACKEND=transformers for it."
            )

        llm_kwargs = {
            "model": self.vllm_model,
            "runner": "generate",
            "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
            "dtype": dtype,
            "trust_remote_code": True,
        }
        if VLLM_MAX_MODEL_LEN:
            llm_kwargs["max_model_len"] = int(VLLM_MAX_MODEL_LEN)
        if VLLM_KV_CACHE_MEMORY_BYTES:
            llm_kwargs["kv_cache_memory_bytes"] = int(VLLM_KV_CACHE_MEMORY_BYTES)
        if VLLM_ENFORCE_EAGER:
            llm_kwargs["enforce_eager"] = True
        logger.info(
            f"Loading vLLM model {self.vllm_model} (gpu_mem_util="
            f"{VLLM_GPU_MEMORY_UTILIZATION}"
            + (f", kv_cache_memory_bytes={VLLM_KV_CACHE_MEMORY_BYTES}"
               if VLLM_KV_CACHE_MEMORY_BYTES else "")
            + ")..."
        )
        self.llm = LLM(**llm_kwargs)

    def transcribe(self, audio, language=None, prompt=None):
        from .utils import language_code_to_name

        messages = []
        if prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": prompt}]})
        messages.append({"role": "user", "content": [{"type": "audio", "audio": audio}]})
        prompt_str = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        forced_name = language_code_to_name(language) or language if language else None
        if forced_name:
            # Force the language by prefilling the transcription header so the
            # model emits only the transcript. The chat template does not keep a
            # trailing assistant turn, so append the header to the prompt string
            # directly (vs continue_final_message, which the template rejects).
            prompt_str += f"language {forced_name}<asr_text>"

        sp = self._SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
        out = self.llm.generate(
            {"prompt": prompt_str, "multi_modal_data": {"audio": (audio, 16000)}},
            sp,
        )
        text = out[0].outputs[0].text
        detected = forced_name
        # When the language is not forced, the model emits the full
        # "language <Name><asr_text><transcription>" string; split it out.
        if "<asr_text>" in text:
            head, text = text.split("<asr_text>", 1)
            head = head.strip()
            if head.startswith("language"):
                detected = head[len("language"):].strip() or detected
        return text.strip(), detected
