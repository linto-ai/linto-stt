import os
import subprocess
import time
import warnings
from pathlib import Path

import pytest

from helpers.env_setup import build_env_dict
from helpers.server_runner import UVServerRunner, DockerServerRunner, find_free_port


# Error signatures that mean "this host can't run CUDA inside a Docker container"
# (common on Docker Desktop / WSL2, where native GPU works but container GPU
# passthrough doesn't). When a GPU Docker test hits one of these, we skip it with
# a clear reason instead of failing — the host driver itself is fine.
_DOCKER_GPU_UNAVAILABLE_SIGNATURES = (
    "CUDA driver version is insufficient",
    "CUDA failed with error",
    "no CUDA-capable device is detected",
    "could not select device driver",   # `docker run --gpus all` unsupported
    "nvidia-container-cli",             # NVIDIA Container Toolkit missing/broken
)


def _docker_gpu_unavailable(message: str) -> bool:
    return any(sig in message for sig in _DOCKER_GPU_UNAVAILABLE_SIGNATURES)


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--engine", action="store", default=None,
                     choices=["nemo", "whisper", "kaldi"],
                     help="Only run tests for this engine")
    parser.addoption("--device", action="store", default="cpu",
                     choices=["cpu", "cuda"],
                     help="Target device (skip GPU tests when cpu)")
    parser.addoption("--docker-only", action="store_true", default=False,
                     help="Only run Docker-based tests")
    parser.addoption("--uv-only", action="store_true", default=False,
                     help="Only run UV-based tests")
    parser.addoption("--server-timeout", action="store", type=int, default=300,
                     help="Timeout in seconds for server startup (failures usually "
                          "surface much sooner via process-exit / fatal-log checks)")
    parser.addoption("--kaldi-am-path", action="store", default=None,
                     help="Path to Kaldi acoustic model")
    parser.addoption("--kaldi-lm-path", action="store", default=None,
                     help="Path to Kaldi language model")


# ---------------------------------------------------------------------------
# Collection filtering
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    engine_filter = config.getoption("--engine")
    device = config.getoption("--device")
    docker_only = config.getoption("--docker-only")
    uv_only = config.getoption("--uv-only")

    deselected = []

    for item in items[:]:
        markers = {m.name for m in item.iter_markers()}

        # Filter by engine
        if engine_filter:
            engine_markers = markers & {"nemo", "whisper", "kaldi"}
            if engine_markers and engine_filter not in engine_markers:
                deselected.append(item)
                continue

        # Skip GPU tests on CPU
        if device == "cpu" and "gpu" in markers:
            item.add_marker(pytest.mark.skip(reason="GPU test skipped (--device=cpu)"))

        # Docker/UV filtering
        if docker_only and "docker" not in markers:
            deselected.append(item)
            continue
        if uv_only and "docker" in markers:
            deselected.append(item)
            continue

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [i for i in items if i not in deselected]


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return str(Path(__file__).resolve().parent.parent)


@pytest.fixture(scope="session")
def test_audio_bonjour():
    """Return path to the bonjour.wav test file."""
    path = Path(__file__).resolve().parent / "bonjour.wav"
    assert path.exists(), f"Test audio not found: {path}"
    return path


@pytest.fixture(scope="session")
def server_timeout(request):
    return request.config.getoption("--server-timeout")


# ---------------------------------------------------------------------------
# Indirect server fixtures (parametrized per-test)
# ---------------------------------------------------------------------------

@pytest.fixture
def uv_server(request, project_root, server_timeout):
    """Launch a UV server. Parametrize indirectly with a dict:
    {"engine": str, "mode": str, "port": int, "env_overrides": dict}
    """
    params = request.param
    engine = params["engine"]
    mode = params.get("mode", "http")
    port = params.get("port", 0)
    env_overrides = params.get("env_overrides", {})

    env_dict = build_env_dict(project_root, engine, env_overrides)
    runner = UVServerRunner(
        project_root=project_root,
        engine=engine,
        mode=mode,
        port=port,
        env_dict=env_dict,
        timeout=server_timeout,
    )
    try:
        url = runner.start()
        yield {"url": url, "runner": runner, "engine": engine, "mode": mode,
               "timeout": server_timeout}
    finally:
        # Stop even if start() raised (e.g. healthcheck timeout), otherwise the
        # subprocess leaks.
        runner.stop()


@pytest.fixture
def docker_server(request, project_root, server_timeout):
    """Launch a Docker server. Parametrize indirectly with a dict:
    {"engine": str, "mode": str, "port": int, "env_overrides": dict,
     "dockerfile": str, "use_gpu": bool, "volumes": dict}
    """
    params = request.param
    engine = params["engine"]
    mode = params.get("mode", "http")
    port = params.get("port", 0)
    env_overrides = params.get("env_overrides", {})
    dockerfile = params.get("dockerfile", "Dockerfile")
    use_gpu = params.get("use_gpu", False)
    volumes = params.get("volumes", {})

    # Reuse the host's HuggingFace cache so the container doesn't re-download
    # the model (e.g. NeMo's 2.4 GB parakeet) on every run — that download is
    # what makes uncached Docker model tests blow past the timeouts.
    hf_cache = os.path.expanduser(
        os.environ.get("HF_HOME", "~/.cache/huggingface"))
    if os.path.isdir(hf_cache):
        # Mount *outside* the runtime user's home: the entrypoint does a
        # `chown -R` on the home, and recursing into a multi-GB bind mount would
        # be painfully slow. HF_HOME points the cache at this mount instead.
        volumes.setdefault(hf_cache, "/opt/hf_cache")
        env_overrides.setdefault("HF_HOME", "/opt/hf_cache")
        # Run the container as the host user. The entrypoint defaults to
        # uid 33 (www-data) otherwise, which can't read the mounted cache (owned
        # by the host user) — so it would re-download despite the mount.
        env_overrides.setdefault("USER_ID", str(os.getuid()))
        env_overrides.setdefault("GROUP_ID", str(os.getgid()))

    # For task mode, mount test dir as /opt/audio
    if mode == "task":
        test_dir = str(Path(__file__).resolve().parent)
        volumes.setdefault(test_dir, "/opt/audio")

    # Kaldi needs AM/LM model volumes
    if engine == "kaldi":
        am_path = request.config.getoption("--kaldi-am-path")
        lm_path = request.config.getoption("--kaldi-lm-path")
        if am_path:
            volumes.setdefault(am_path, "/opt/AM")
        if lm_path:
            volumes.setdefault(lm_path, "/opt/LM")

    env_dict = build_env_dict(project_root, engine, env_overrides)
    runner = DockerServerRunner(
        project_root=project_root,
        engine=engine,
        mode=mode,
        port=port,
        env_dict=env_dict,
        timeout=server_timeout,
        dockerfile=dockerfile,
        use_gpu=use_gpu,
        volumes=volumes,
    )
    try:
        try:
            url = runner.start()
        except RuntimeError as exc:
            # GPU Docker tests can't run if this host can't pass CUDA into a
            # container — skip (not fail), since the native GPU path works.
            # Also emit a warning so the skip is visible in the run summary
            # (skip reasons are otherwise only shown with -rs/-ra).
            if use_gpu and _docker_gpu_unavailable(str(exc)):
                warnings.warn(
                    f"GPU Docker test skipped: CUDA is not usable inside a "
                    f"container on this host: {exc}",
                    stacklevel=2,
                )
                pytest.skip(f"Docker GPU not usable in this environment: {exc}")
            raise
        yield {"url": url, "runner": runner, "engine": engine, "mode": mode,
               "timeout": server_timeout}
    finally:
        # Stop even if start() raised (e.g. healthcheck timeout), otherwise the
        # container leaks (left running with --rm but never stopped).
        runner.stop()


@pytest.fixture(scope="session")
def redis_server():
    """Launch a Redis container for Celery tests. Shared across the session."""
    container_name = "test_redis_pytest"
    redis_image = "redis/redis-stack-server:latest"

    # Make sure the image is present, pulling it up-front (blocking) if needed.
    # We pull here rather than relying on an on-the-fly pull during `docker run`:
    # that pull (~1 min) could still be running when the Celery worker's broker
    # wait (wait-for-it.sh, 20s) starts, so Redis wouldn't be reachable in time.
    # Pulling before anything starts guarantees the image is ready.
    if subprocess.run(
        ["docker", "image", "inspect", redis_image],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode != 0:
        print(f"Pulling {redis_image} (one-time, may take ~1 min)...")
        pull = subprocess.run(
            ["docker", "pull", redis_image],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if pull.returncode != 0:
            pytest.fail(
                f"Could not pull Redis image '{redis_image}' "
                f"(pull it manually with: docker pull {redis_image}):\n"
                f"{pull.stdout.decode(errors='replace')}",
                pytrace=False,
            )

    # Stop any existing instance
    subprocess.run(
        ["docker", "stop", container_name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1)

    proc = subprocess.Popen(
        [
            "docker", "run", "--rm",
            "-p", "6379:6379",
            "--name", container_name,
            redis_image,
            "redis-server", "/etc/redis-stack.conf",
            "--protected-mode", "no",
            "--bind", "0.0.0.0",
            "--loglevel", "debug",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)
    if proc.poll() is not None:
        raise RuntimeError("Redis container failed to start")

    yield "redis://localhost:6379"

    subprocess.run(
        ["docker", "stop", container_name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
