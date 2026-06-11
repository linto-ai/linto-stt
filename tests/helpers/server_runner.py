import os
import signal
import socket
import subprocess
import tempfile
import time
import logging

import requests

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    """Bind to port 0 to let the OS assign a free port, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Markers in server logs that mean startup has fatally failed — used to fail a
# health-check fast instead of waiting out the whole timeout.
_FATAL_LOG_MARKERS = (
    "Traceback (most recent call last)",
    "Failed to load transcription model",
    "Error loading ASGI app",
    "Application startup failed",
    "Address already in use",
)


def _format_logs(get_logs, max_chars: int = 4000) -> str:
    """Return a trimmed server-log blob to attach to an error message."""
    if not get_logs:
        return ""
    logs = (get_logs() or "").strip()
    if not logs:
        return ""
    if len(logs) > max_chars:
        logs = "...(truncated)...\n" + logs[-max_chars:]
    return f"\n--- server logs ---\n{logs}"


def _poll_healthcheck(url, timeout, process_or_container=None, get_logs=None,
                      fatal_markers=()) -> None:
    """Poll a URL until it returns 2xx/4xx; fail fast if the server exits or logs
    a fatal error; otherwise time out. Server logs are attached to all errors."""
    deadline = time.monotonic() + timeout
    interval = 1.0
    last_error = None
    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code in (200, 400, 426):
                logger.info(f"Server ready at {url}")
                return
        except requests.ConnectionError as e:
            last_error = e
        # Fast-fail if the process/container exited before becoming ready.
        if process_or_container is not None and hasattr(process_or_container, "poll"):
            if process_or_container.poll() is not None:
                raise RuntimeError(
                    f"Server exited with code {process_or_container.returncode} "
                    f"before becoming ready.{_format_logs(get_logs)}"
                )
        # Fast-fail if the server logged a fatal error (don't wait out the timeout).
        if get_logs and fatal_markers:
            logs = get_logs()
            if any(m in logs for m in fatal_markers):
                raise RuntimeError(
                    f"Server failed during startup (matched a fatal log marker)."
                    f"{_format_logs(get_logs)}"
                )
        time.sleep(interval)
    raise TimeoutError(
        f"Server at {url} not ready after {timeout}s (last error: {last_error})."
        f"{_format_logs(get_logs)}"
    )


def _wait_for_container_log(container_name: str, marker: str, timeout: float,
                           process=None) -> None:
    """Poll `docker logs` until `marker` appears, the process dies, or timeout.

    Used for Celery task mode, which runs only a worker (no HTTP endpoint to
    poll), so readiness is detected from the worker's startup log instead.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        logs = subprocess.run(
            ["docker", "logs", container_name],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ).stdout.decode(errors="replace")
        if marker in logs:
            logger.info(f"Worker ready in {container_name} (found {marker!r})")
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Container process exited with code {process.returncode}\n{logs}"
            )
        time.sleep(1.0)
    raise TimeoutError(
        f"Container {container_name} did not log {marker!r} within {timeout}s"
    )


class UVServerRunner:
    """Launch a linto-stt server via `uv run main.py` as a subprocess."""

    def __init__(self, project_root: str, engine: str, mode: str, port: int,
                 env_dict: dict, timeout: float = 600):
        self.project_root = project_root
        self.engine = engine
        self.mode = mode
        self.port = port if port != 0 else find_free_port()
        self.env_dict = env_dict
        self.timeout = timeout
        self.process = None
        self._log_file = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _read_logs(self) -> str:
        if not self._log_file:
            return ""
        try:
            with open(self._log_file.name, "r", errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    def start(self) -> str:
        """Start the server and wait until healthy. Returns base URL."""
        # Build subprocess environment: inherit current env + overlay our vars
        env = os.environ.copy()
        env.update(self.env_dict)
        env["STT_ENGINE"] = self.engine

        cmd = [
            "uv", "run", "main.py",
            "-m", self.mode,
            "-e", self.engine,
            "-p", str(self.port),
            "-i", "127.0.0.1",
        ]
        logger.info(f"Starting UV server: {' '.join(cmd)}")
        self._log_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False, prefix="linto_uv_"
        )
        self.process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            env=env,
            # Write logs to a file, NOT a PIPE: an undrained PIPE buffer fills up
            # during the verbose NeMo/uvicorn startup and blocks the server on its
            # next write (it then never binds the port). The file also lets us
            # surface the logs when start-up fails.
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            # New session/process group: `uv run` spawns a child `python main.py`,
            # so we must signal the whole group to stop the server (see stop()).
            start_new_session=True,
        )

        # Wait for healthcheck
        if self.mode == "http":
            healthcheck_url = f"{self.base_url}/healthcheck"
        elif self.mode == "websocket":
            healthcheck_url = self.base_url
        else:
            # task mode (Celery) - no HTTP endpoint to poll
            time.sleep(5)
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Celery server exited with code {self.process.returncode}."
                    f"{_format_logs(self._read_logs)}"
                )
            return self.base_url

        _poll_healthcheck(healthcheck_url, self.timeout, self.process,
                          get_logs=self._read_logs, fatal_markers=_FATAL_LOG_MARKERS)
        return self.base_url

    def stop(self):
        """Terminate the server process group.

        `uv run` spawns a child `python main.py` in the same process group.
        Killing only the `uv run` parent leaves that child (which holds the
        loaded model) running, so we signal the whole group. Otherwise leaked
        servers accumulate across tests and starve the CPU.
        """
        if self.process is None:
            return
        try:
            pgid = os.getpgid(self.process.pid)
        except ProcessLookupError:
            self.process = None
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=5)
        except ProcessLookupError:
            pass
        self.process = None
        if self._log_file:
            try:
                self._log_file.close()
            except OSError:
                pass
            if os.path.exists(self._log_file.name):
                os.unlink(self._log_file.name)
            self._log_file = None


class DockerServerRunner:
    """Build and run a linto-stt Docker container."""

    _built_images: dict = {}  # (dockerfile, use_gpu) -> image tag

    def __init__(self, project_root: str, engine: str, mode: str, port: int,
                 env_dict: dict, timeout: float = 600,
                 dockerfile: str = "Dockerfile", use_gpu: bool = False,
                 volumes: dict = None):
        self.project_root = project_root
        self.engine = engine
        self.mode = mode
        self.port = port if port != 0 else find_free_port()
        self.env_dict = env_dict
        self.timeout = timeout
        self.dockerfile = dockerfile
        self.use_gpu = use_gpu
        self.volumes = volumes or {}
        self.container_name = f"test_{engine}_{mode}_{self.port}"
        self._env_file = None
        self._process = None
        self._log_file = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _read_logs(self) -> str:
        """Return `docker run` output (the docker CLI's own messages plus the
        container's stdout/stderr), captured to a file to avoid a PIPE deadlock."""
        if not self._log_file:
            return ""
        try:
            with open(self._log_file.name, "r", errors="replace") as f:
                return f.read()
        except OSError:
            return ""

    def _build_image(self) -> str:
        """Build the Docker image if not already built. Returns image tag."""
        cache_key = (self.dockerfile, self.use_gpu)
        if cache_key in DockerServerRunner._built_images:
            return DockerServerRunner._built_images[cache_key]

        # GPU and CPU builds are different images (the GPU build installs the
        # CUDA runtime libs), so they get distinct tags / cache entries.
        tag = f"linto-stt-test:{self.engine}{'-gpu' if self.use_gpu else ''}"
        cmd = [
            "docker", "build", ".",
            "-f", self.dockerfile,
            "--build-arg", f"STT_ENGINE={self.engine}",
        ]
        if self.use_gpu:
            # GPU=1 installs cuBLAS/cuDNN that ctranslate2 needs for CUDA.
            cmd.extend(["--build-arg", "GPU=1"])
        cmd.extend(["-t", tag])
        logger.info(f"Building Docker image: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Docker build failed:\n{result.stderr.decode()}"
            )
        DockerServerRunner._built_images[cache_key] = tag
        return tag

    def start(self) -> str:
        """Build image, run container, wait for healthy. Returns base URL."""
        image_tag = self._build_image()

        # Write env to a temp file
        self._env_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, prefix="linto_test_"
        )
        env_with_mode = dict(self.env_dict)
        env_with_mode["SERVICE_MODE"] = self.mode
        env_with_mode["STT_ENGINE"] = self.engine
        for k, v in env_with_mode.items():
            self._env_file.write(f"{k}={v}\n")
        self._env_file.close()

        cmd = [
            "docker", "run", "--rm",
            "-p", f"{self.port}:80",
            "--name", self.container_name,
            "--env-file", self._env_file.name,
        ]

        if self.use_gpu:
            cmd.extend(["--gpus", "all"])

        for host_path, container_path in self.volumes.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])

        cmd.append(image_tag)

        logger.info(f"Starting Docker container: {' '.join(cmd)}")
        # Log to a file, NOT a PIPE: an undrained PIPE buffer fills up during the
        # verbose container startup and blocks it. The file captures both the
        # docker CLI's own errors (e.g. "could not select device driver") and the
        # container's stdout/stderr, and lets us surface them on failure.
        self._log_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False, prefix="linto_docker_"
        )
        self._process = subprocess.Popen(
            cmd, cwd=self.project_root,
            stdout=self._log_file, stderr=subprocess.STDOUT,
        )
        time.sleep(2)

        if self._process.poll() is not None:
            raise RuntimeError(
                f"Docker container exited immediately.{_format_logs(self._read_logs)}"
            )

        # Wait until ready
        if self.mode == "task":
            # Celery worker only — no HTTP endpoint. Wait for the worker's
            # "ready." startup log instead of polling /healthcheck (which would
            # never respond and hang until timeout).
            _wait_for_container_log(
                self.container_name, "ready.", self.timeout, self._process)
        else:
            healthcheck_url = f"{self.base_url}/healthcheck" if self.mode == "http" else self.base_url
            _poll_healthcheck(healthcheck_url, self.timeout, self._process,
                              get_logs=self._read_logs, fatal_markers=_FATAL_LOG_MARKERS)
        return self.base_url

    def stop(self):
        """Stop and remove the container."""
        subprocess.run(
            ["docker", "stop", self.container_name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Clean up env file
        if self._env_file and os.path.exists(self._env_file.name):
            os.unlink(self._env_file.name)
            self._env_file = None
        # Clean up log file
        if self._log_file:
            try:
                self._log_file.close()
            except OSError:
                pass
            if os.path.exists(self._log_file.name):
                os.unlink(self._log_file.name)
            self._log_file = None
