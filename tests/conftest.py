import os
import subprocess
import time
from pathlib import Path

import pytest

from helpers.env_setup import build_env_dict
from helpers.server_runner import UVServerRunner, DockerServerRunner, find_free_port


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
    parser.addoption("--server-timeout", action="store", type=int, default=600,
                     help="Timeout in seconds for server startup")
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
    url = runner.start()
    yield {"url": url, "runner": runner, "engine": engine, "mode": mode}
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
    url = runner.start()
    yield {"url": url, "runner": runner, "engine": engine, "mode": mode}
    runner.stop()


@pytest.fixture(scope="session")
def redis_server():
    """Launch a Redis container for Celery tests. Shared across the session."""
    container_name = "test_redis_pytest"
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
            "redis/redis-stack-server:latest",
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
