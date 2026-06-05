import os
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


def _poll_healthcheck(url: str, timeout: float, process_or_container=None) -> None:
    """Poll a URL until it returns 2xx/4xx or timeout is reached."""
    deadline = time.monotonic() + timeout
    interval = 1.0
    last_error = None
    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code in (200, 400, 426):
                logger.info(f"Server ready at {url} after {timeout - (deadline - time.monotonic()):.0f}s")
                return
        except requests.ConnectionError as e:
            last_error = e
        # Check if process died
        if process_or_container is not None and hasattr(process_or_container, "poll"):
            if process_or_container.poll() is not None:
                stdout = process_or_container.stdout.read().decode() if process_or_container.stdout else ""
                stderr = process_or_container.stderr.read().decode() if process_or_container.stderr else ""
                raise RuntimeError(
                    f"Server process exited with code {process_or_container.returncode}\n"
                    f"stdout: {stdout}\nstderr: {stderr}"
                )
        time.sleep(interval)
    raise TimeoutError(f"Server at {url} not ready after {timeout}s (last error: {last_error})")


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

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

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
        self.process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
                stdout = self.process.stdout.read().decode() if self.process.stdout else ""
                stderr = self.process.stderr.read().decode() if self.process.stderr else ""
                raise RuntimeError(
                    f"Celery server exited with code {self.process.returncode}\n"
                    f"stdout: {stdout}\nstderr: {stderr}"
                )
            return self.base_url

        _poll_healthcheck(healthcheck_url, self.timeout, self.process)
        return self.base_url

    def stop(self):
        """Terminate the server process."""
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.process = None


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

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

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
        self._process = subprocess.Popen(
            cmd, cwd=self.project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(2)

        if self._process.poll() is not None:
            stdout = self._process.stdout.read().decode() if self._process.stdout else ""
            stderr = self._process.stderr.read().decode() if self._process.stderr else ""
            raise RuntimeError(
                f"Docker container exited immediately:\n{stdout}\n{stderr}"
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
            _poll_healthcheck(healthcheck_url, self.timeout, self._process)
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
