import unittest
import os
import time
import subprocess
import requests
import re
import warnings
from ddt import ddt
from pathlib import Path
from automated_utils import AUTOMATEDTESTDIR, TESTDIR, SERVER_STARTING_TIMEOUT, get_file_regex, parse_env_variables

def finalize_tests():
    subprocess.run(["docker", "stop", "test_container"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "stop", "test_redis"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)    

@ddt
class TestRunner(unittest.TestCase):

    built_images = []
    redis_launched = False

    def echo_success(self, message):
        print('\033[0;32m' + u'\u2714' + '\033[0m ' + message)

    def echo_failure(self, message):
        print('\033[0;31m' + u'\u2716' + '\033[0m ' + message)

    def echo_note(self, message):
        print(u'\u231B' + ' ' + message)

    def echo_command(self, message):
        print(f"$ {message}")

    def report_failure(self, message, expect_failure=False):
        if not expect_failure:
            self.echo_failure(message)
        self.cleanup()
        if not expect_failure:
            self.fail(message)
        return message

    def report_success(self):
        self.echo_success("Test passed.")
        self.cleanup()

    def cleanup(self):
        # Check if the container is running
        p = subprocess.Popen(["docker", "ps", "-a"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if b"test_container" in out:        
            self.echo_command("docker stop test_container")
            subprocess.run(["docker", "stop", "test_container"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.2) # Without this, the following tests can fail (The container name "/test_container" is already in use)

    def process_output(self, p):
        l = p.communicate()[0].decode('utf-8').replace('\n', '\n\t')
        e = p.communicate()[1].decode('utf-8').replace('\n', '\n\t')
        return f" \u2192 Log Message:\n\t{l}\n \u2192 Error Message:\n\t{e}"


    def check_http_server_availability(self, server, pid, streaming=False):
        total_wait_time = SERVER_STARTING_TIMEOUT  # 10 minutes in seconds
        retry_interval = 1    # Interval between attempts (in seconds)
        elapsed_time = 0

        while elapsed_time < total_wait_time:
            try:
                if streaming:
                    response = requests.get(server)
                else:
                    response = requests.head(server)
                if response.status_code == 200 or response.status_code == 400 or (streaming and response.status_code == 426):
                    self.echo_note(f"Server: {server} is available after {elapsed_time} sec.")
                    return
            except requests.ConnectionError:
                pass
            if pid.poll() is not None:
                return f"The server container has stopped for an unexpected reason.\n{self.process_output(pid)}"

            time.sleep(retry_interval)
            elapsed_time += retry_interval

        return f"Server: {server} is not available after {total_wait_time} seconds, server launching must have failed.\n{self.process_output(pid)}"

    def launch_redis(self):
        if TestRunner.redis_launched:
            return
        cmd = "docker run --rm -p 6379:6379 --name test_redis redis/redis-stack-server:latest redis-server /etc/redis-stack.conf --protected-mode no --bind 0.0.0.0 --loglevel debug"
        self.echo_command(cmd)
        p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)
        if p.poll() is not None:
            self.cleanup()
            return f"Redis server failed to start.\n{self.process_output(p)}", None
        TestRunner.redis_launched = True

    def build_and_run_container(self, serving, docker_image, env_variables, use_local_cache):
        self.echo_note(f"* Docker image: {docker_image}")
        self.echo_note(f"* Options.....: {env_variables}")
        build_args = ""
        for i, env in enumerate(env_variables.split()):
            if i>0 and env_variables.split()[i-1] =="-v":
                build_args += f"-v {env} "
            elif env=="-v":
                continue
            else:
                build_args += f"--env {env} "
        build_args += f"--env SERVICE_MODE={serving} "
        if use_local_cache:
            home = str(Path.home())
            build_args += f"-v {home}/.cache:/root/.cache "

        if serving == "task":
            self.launch_redis()
            build_args += f"-v {TESTDIR}/:/opt/audio "

        tag = f"test_{os.path.basename(docker_image)}"
        if tag not in TestRunner.built_images:
            # Only build images that have not been built yet
            cmd = f'docker build . -f {docker_image} -t linto-stt-test:{tag}'
            self.echo_command(cmd)
            start_time = time.time()
            p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            p.wait()
            end_time = time.time()
            if p.poll() != 0:
                self.cleanup()
                return f"Docker build failed.\n{self.process_output(p)}", None
            self.echo_note(f"Docker image has been successfully built in {end_time - start_time:.0f} sec.")
            TestRunner.built_images.append(tag)

        cmd=f"docker run --rm -p 8080:80 --name test_container --env-file {AUTOMATEDTESTDIR}/.env --gpus all {build_args} linto-stt-test:{tag}"
        self.echo_command(cmd)
        def launch_docker_run(command):
            p = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(1)
            if p.poll() is not None:
                error_log = self.process_output(p)
                if "Conflict. The container name " in error_log:
                    self.echo_note(f"Previous conatiner was not stopped yet, retry docker run")
                    self.cleanup()
                    launch_docker_run(command)
                else:
                    return f"Docker container failed to start.\n{self.process_output(p)}", None
            return None, p
        error, p = launch_docker_run(cmd)
        return error, p

    def transcribe(self, command, regex, test_file, error_message, success_message, timeout=None):
        start = time.time()
        res = subprocess.run(command, shell=True, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        end = time.time()
        if res.returncode != 0:
            raise FileNotFoundError(f"Error: {res.stderr.decode('utf-8')}")
        res = res.stdout.decode('utf-8')
        if not re.search(regex, res):
            message = f"{error_message}: The string '{res}' is not matching the regex ({regex}), the server didn't transcribe correctly."
            return self.report_failure(message)
        self.echo_note(f"{success_message} has transcribed {test_file} in {end - start:.0f} sec.")
        return

    def run_test(self, docker_image="whisper/Dockerfile.ctranslate2", serving="http", env_variables="", test_file=f"{TESTDIR}/bonjour.wav", language=None, use_local_cache=True, expect_failure=False):
        warnings.simplefilter("ignore", ResourceWarning)
        regex = get_file_regex(test_file, parse_env_variables(env_variables).get("LANGUAGE", "fr") if language is None else language)
        r, pid = self.build_and_run_container(serving, docker_image, env_variables, use_local_cache)
        if r:
            return self.report_failure(r, expect_failure=expect_failure)
        if serving == "http":
            r=self.check_http_server_availability("http://localhost:8080/healthcheck", pid)
            if r:
                return self.report_failure(r, expect_failure=expect_failure)
            cmd = f'curl -X POST "http://localhost:8080/transcribe" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@{test_file};type=audio/wav"'
            if language:
                cmd += f' -F "language={language}"'
            self.echo_command(cmd)
            r = self.transcribe(cmd, regex, test_file, "Error transcription", "HTTP route 'transcribe'")
        elif serving == "websocket":
            r=self.check_http_server_availability("http://localhost:8080", pid, streaming=True)
            if r:
                return self.report_failure(r, expect_failure=expect_failure)
            cmd = f"python3 {TESTDIR}/test_streaming.py --audio_file {test_file} -v --stream_duration 1 --stream_wait 0.0"
            if language:
                cmd += f" --language {language}"
            self.echo_command(cmd)
            r = self.transcribe(cmd, regex, test_file, "Error streaming", "HTTP route 'streaming'")
        elif serving == "task":
            # you can be stuck here if the server crashed bc the task will be in the queue forever
            cmd = f"python3 {TESTDIR}/test_celery.py --audio_file {os.path.basename(test_file)}"
            if language:
                cmd += f" --language {language}"
            self.echo_command(cmd)
            r = self.transcribe(cmd, regex, test_file, "Error task", "TASK route", timeout=60)
        else:
            raise RuntimeError(f"Unknown serving mode: {serving}")
        if r:
            return self.report_failure(r, expect_failure=expect_failure)
        if not expect_failure:
            self.report_success()
        return ""

    def setUp(self):
        # Print an empty line because unittest prints the name of the test first, without a newline
        print()
        print("-"*70)

    def tearDown(self):
        print("-"*70)
