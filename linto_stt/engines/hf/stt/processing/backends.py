"""Swappable ASR backends for the hf engine.

The engine runs one pipeline (chunking, forced-alignment word timestamps,
language/hotword handling, response assembly) around a pluggable ASR backend
selected by the ``BACKEND`` environment variable:

- ``vllm`` (default): a managed ``vllm serve`` subprocess reached over HTTP, for
  higher throughput; serves any vLLM-servable ASR model (CUDA GPU only).
- ``transformers``: in-process Hugging Face ``transformers`` (runs on CPU too).
- ``vllm_offline``: in-process vLLM (``vllm.LLM``), Qwen3-ASR only (experimental).

A backend only has to turn one audio segment into ``(text, detected_language)``;
word timestamps (the forced aligner) and everything else stay backend-agnostic,
so backends can be benchmarked against each other by flipping ``BACKEND``.
"""
import inspect
import os

import torch

from linto_stt.engines.hf.stt import MAX_NEW_TOKENS, logger

from .utils import SAMPLE_RATE


def detect_capabilities(processor):
    """Detect a model's ASR capabilities from its *processor's* transcription
    signature (not from the model name), so it is robust across checkpoints:

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
# gpu_memory_utilization fraction; preferable on unified-memory / shared GPUs
# where a relative share is awkward. gpu_memory_utilization still bounds the total
# reservation (weights + activations + KV).
VLLM_KV_CACHE_MEMORY_BYTES = os.environ.get("VLLM_KV_CACHE_MEMORY_BYTES") or None
# Skip torch.compile + CUDA-graph capture. Uses less VRAM and starts almost
# instantly (no ~minute-long compile), at some inference-throughput cost. Useful
# on memory-tight GPUs; off by default so the graphs give best throughput.
VLLM_ENFORCE_EAGER = os.environ.get("VLLM_ENFORCE_EAGER", "").strip().lower() in (
    "1", "true", "yes",
)
# --- vLLM "serve" backend (BACKEND=vllm): a managed `vllm serve` subprocess that
# the engine talks to over HTTP (/v1/audio/transcriptions). Unlike the in-process
# offline backend, this one runs vLLM in its own process (no transformers-registry
# pollution) and works for every vLLM-servable ASR model (Qwen3-ASR and Cohere).
VLLM_SERVE_HOST = os.environ.get("VLLM_SERVE_HOST", "127.0.0.1")
# Port for the managed `vllm serve`. Unset (the default) picks a free port per
# server at startup, so a leftover/orphaned server never blocks a new one; set it
# to pin a fixed port.
VLLM_SERVE_PORT = os.environ.get("VLLM_SERVE_PORT") or None
# How long to wait for the subprocess to become healthy (model load + warmup).
VLLM_SERVE_STARTUP_TIMEOUT = int(os.environ.get("VLLM_SERVE_STARTUP_TIMEOUT", "900"))
# Per-request HTTP timeout (long audio is chunked server-side, so keep generous).
VLLM_SERVE_REQUEST_TIMEOUT = int(os.environ.get("VLLM_SERVE_REQUEST_TIMEOUT", "600"))
# Extra raw args appended verbatim to the `vllm serve` command line.
VLLM_SERVE_EXTRA_ARGS = os.environ.get("VLLM_SERVE_EXTRA_ARGS", "").split()


def _find_free_port(host="127.0.0.1"):
    """Ask the OS for a currently-free TCP port on `host`."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _model_derived_max_len(model_name):
    """Best-effort read of a model's own maximum sequence length from its config.

    Used to cap ``--max-model-len`` at ``min(our cap, the model's own max)``: a
    model may advertise a huge context (Qwen3-ASR: 65536) whose KV cache does not
    fit a modest GPU share, while another caps low (Cohere: 1024) and rejects any
    larger value. Scans the common length keys on the top config and its text /
    decoder sub-configs; returns the largest found (ignoring the tiny audio-tower
    value), or None when it cannot tell.
    """
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:  # noqa: BLE001 -- unknown config -> fall back to our cap
        return None
    keys = (
        "max_position_embeddings", "n_positions", "max_seq_len", "seq_length",
        "max_sequence_length", "max_seq_length", "model_max_length",
        "max_target_positions",  # Whisper's decoder limit (448); not max_source_positions
    )
    sub_names = ("text_config", "thinker_config", "decoder_config", "llm_config")
    configs = [config] + [
        getattr(config, s) for s in sub_names if getattr(config, s, None) is not None
    ]
    values = [
        v
        for cfg in configs
        for k in keys
        if isinstance((v := getattr(cfg, k, None)), int) and 0 < v < 10_000_000
    ]
    return max(values) if values else None


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
    """In-process vLLM ASR (``vllm.LLM``), loaded lazily. Selected by
    ``BACKEND=vllm_offline`` (the HTTP ``vllm serve`` backend below is ``vllm``).

    Qwen3-ASR only: it builds that model's decoder-only prompt, which does not fit
    an encoder-decoder model such as Cohere (use ``BACKEND=vllm`` or
    ``transformers`` for those).

    vLLM uses the non-``-hf`` checkpoint layout (``Qwen/Qwen3-ASR-1.7B``), so an
    ``-hf`` ``MODEL`` is mapped to its non-``-hf`` id. Transcription builds the
    chat/transcription prompt with the HF processor's chat template (system =
    context/hotwords prompt, user = audio, optional assistant prefill
    ``language <Name><asr_text>`` to force the language) and feeds it to
    ``LLM.generate`` with the raw audio as ``multi_modal_data``. The model emits
    ``language <Name><asr_text><transcription>``, which we parse.

    GPU only. Word timestamps still come from the transformers forced aligner.
    """

    name = "vllm_offline"
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
        # is not wired here, so fail clearly rather than emit a wrong prompt.
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


class VllmServeBackend:
    """vLLM ASR served by a managed ``vllm serve`` subprocess, over HTTP.

    Selected by ``BACKEND=vllm``. On load it starts ``vllm serve <model>`` as a
    child process (one per worker, inside the container) and talks to it through
    the OpenAI-compatible ``/v1/audio/transcriptions`` endpoint. Unlike the
    in-process offline backend (``BACKEND=vllm_offline``), vLLM runs in its own
    process, so it does not patch the parent's transformers registry and it works
    for every vLLM-servable ASR family, including encoder-decoder models such as
    Cohere. vLLM chunks long audio server-side, so the engine does not re-chunk.

    GPU only. Word timestamps still come from the transformers forced aligner.
    Per-request capabilities (hotwords / language) are detected from the HF
    processor, as for the transformers backend, and passed through as the OpenAI
    ``prompt`` (context/hotwords) and ``language`` (forced language) fields.
    """

    name = "vllm"
    handles_long_audio = True  # vLLM serve chunks long audio server-side

    def __init__(self, model_name, device, dtype="bfloat16", **kwargs):
        if "cuda" not in str(device):
            raise RuntimeError(
                "The vLLM backend requires a CUDA device (set DEVICE=cuda)."
            )
        self.model_name = model_name
        # vLLM wants the non-`-hf` checkpoint for Qwen (the `-hf` repo targets
        # transformers); a no-op for models without that suffix (e.g. Cohere).
        self.served_model = model_name.removesuffix("-hf")

        # Capabilities from the HF processor (name-independent, same rule as the
        # transformers backend). Loading the processor is cheap and does not pull
        # the model weights.
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name)
        caps = detect_capabilities(processor)
        self.supports_hotwords = caps["hotwords"]
        self.supports_language_spec = caps["language_spec"]
        self.supports_language_auto = caps["language_auto"]

        self.host = VLLM_SERVE_HOST
        # Pin the port only if the user set VLLM_SERVE_PORT; otherwise pick a free
        # one, so an orphaned/leftover server can never block a new one.
        self.port = int(VLLM_SERVE_PORT) if VLLM_SERVE_PORT else _find_free_port(self.host)
        self.base_url = f"http://{self.host}:{self.port}"
        self._proc = None
        self._log = None
        self._start_server(dtype)

    def _build_command(self, dtype):
        import shutil
        import sys

        vllm_bin = shutil.which("vllm") or os.path.join(
            os.path.dirname(sys.executable), "vllm"
        )
        cmd = [
            vllm_bin, "serve", self.served_model,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(VLLM_GPU_MEMORY_UTILIZATION),
            "--dtype", dtype,
            "--trust-remote-code",
        ]
        # Cap the context at min(our cap, the model's own max): Qwen3-ASR's 65536
        # default needs a KV cache too big for a modest GPU share, while Cohere
        # caps at 1024 and rejects anything larger. VLLM_MAX_MODEL_LEN is the cap
        # (16384 by default); the model's own max is never exceeded.
        derived = _model_derived_max_len(self.model_name)
        effective = min(VLLM_MAX_MODEL_LEN, derived) if derived else VLLM_MAX_MODEL_LEN
        cmd += ["--max-model-len", str(effective)]
        if VLLM_KV_CACHE_MEMORY_BYTES:
            cmd += ["--kv-cache-memory-bytes", str(int(VLLM_KV_CACHE_MEMORY_BYTES))]
        if VLLM_ENFORCE_EAGER:
            cmd += ["--enforce-eager"]
        cmd += VLLM_SERVE_EXTRA_ARGS
        return cmd

    def _start_server(self, dtype):
        import atexit
        import subprocess
        import tempfile

        # Fail (with a clear message) if the port stays occupied, instead of
        # silently binding to a stale/orphaned server and getting confusing 404s
        # from a mismatched process. Tolerate a brief window where a just-killed
        # previous server is still releasing the port.
        self._wait_port_free(timeout=15)
        cmd = self._build_command(dtype)
        log_path = os.path.join(
            tempfile.gettempdir(), f"vllm_serve_{self.port}.log"
        )
        logger.info(
            f"Starting vLLM server for {self.served_model} on {self.base_url} "
            f"(logs: {log_path}): {' '.join(cmd)}"
        )
        self._log = open(log_path, "wb")
        # vLLM would JIT-compile FlashInfer's top-k/top-p sampler at startup, which
        # needs the CUDA toolkit (nvcc) that a slim runtime image does not ship. We
        # decode greedily (temperature=0), so the native PyTorch sampler is
        # equivalent -- disable the FlashInfer sampler to drop the nvcc dependency.
        # Overridable: set VLLM_USE_FLASHINFER_SAMPLER yourself to keep it.
        env = {**os.environ}
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        # Keep the subprocess in this process group (NO new session): when the
        # parent or its group is killed -- a container stop, or the test harness
        # signalling the whole group -- vLLM and its EngineCore workers are reaped
        # too, instead of orphaning a server that keeps holding the port and GPU.
        self._proc = subprocess.Popen(
            cmd, stdout=self._log, stderr=subprocess.STDOUT, env=env
        )
        atexit.register(self.shutdown)
        self._wait_until_ready(log_path)

    def _port_in_use(self):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex((self.host, self.port)) == 0

    def _wait_port_free(self, timeout=15):
        import time

        deadline = time.time() + timeout
        while self._port_in_use():
            if time.time() >= deadline:
                raise RuntimeError(
                    f"Port {self.port} on {self.host} is still in use after "
                    f"{timeout}s (an orphaned vLLM server may still be running). "
                    f"Free it, or set a different VLLM_SERVE_PORT."
                )
            time.sleep(1)

    def _wait_until_ready(self, log_path):
        import time

        import requests

        deadline = time.time() + VLLM_SERVE_STARTUP_TIMEOUT
        health = f"{self.base_url}/health"
        while time.time() < deadline:
            if self._proc.poll() is not None:
                returncode = self._proc.returncode
                self.shutdown()  # release the log handle, drop the atexit hook
                raise RuntimeError(
                    f"vLLM server exited (code {returncode}) during "
                    f"startup. Last log lines:\n{self._tail_log(log_path)}"
                )
            try:
                if requests.get(health, timeout=2).status_code == 200:
                    logger.info(f"vLLM server ready on {self.base_url}")
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        self.shutdown()
        raise RuntimeError(
            f"vLLM server did not become ready within {VLLM_SERVE_STARTUP_TIMEOUT}s. "
            f"Last log lines:\n{self._tail_log(log_path)}"
        )

    @staticmethod
    def _tail_log(log_path, n=20):
        try:
            with open(log_path, "r", errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return "(no log)"

    def transcribe(self, audio, language=None, prompt=None):
        import io

        import requests
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buf.seek(0)

        data = {"model": self.served_model, "temperature": "0", "response_format": "json"}
        if language:
            data["language"] = language
        if prompt:
            # Qwen3-ASR uses the OpenAI `prompt` field as its context/hotwords
            # prompt; models without a prompt mechanism ignore it (hotwords are
            # rejected upstream for them anyway, from the detected capabilities).
            data["prompt"] = prompt

        resp = requests.post(
            f"{self.base_url}/v1/audio/transcriptions",
            data=data,
            files={"file": ("audio.wav", buf, "audio/wav")},
            timeout=VLLM_SERVE_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        text = (resp.json().get("text") or "").strip()
        # The transcription endpoint does not return the auto-detected language;
        # report the forced one when given, else None (word timestamps and the
        # response language fall back accordingly).
        return text, language

    def shutdown(self):
        """Stop the vLLM server (idempotent; runs at exit). SIGTERM lets vLLM reap
        its own EngineCore workers; escalate to SIGKILL if it does not exit."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001 -- escalate to SIGKILL on any wait issue
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:  # noqa: BLE001
                    pass
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None
        import atexit

        atexit.unregister(self.shutdown)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:  # noqa: BLE001
            pass
