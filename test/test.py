import unittest
import os
import time
import subprocess
import requests
import argparse
import re
from ddt import ddt, data, idata
import signal
import sys



class TestContainer():
    def __init__(self, show_failed_tests=True):
        self.show_failed_tests = show_failed_tests
        self.cleanup()

    def echo_success(self, message):
        print('\033[0;32m' + u'\u2714' + '\033[0m ' + message)

    def echo_failure(self, message):
        print('\033[0;31m' + u'\u2716' + '\033[0m ' + message)

    def echo_note(self, message):
        print(u'\u231B' + ' ' + message)

    def echo_command(self, message):
        print(f"$ {message}")

    def test_failed(self, message):
        if self.show_failed_tests:
            self.echo_failure(message)
            self.cleanup()

    def test_succeeded(self):
        self.echo_success(f"Test passed.")
        self.cleanup()

    def cleanup(self):
        try:
            os.remove("build_finished")
        except FileNotFoundError:
            pass
        subprocess.run(["docker", "stop", "test_redis"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["docker", "stop", "test_container"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def process_output(self, p):
        l = p.communicate()[0].decode('utf-8').replace('\n', '\n\t')
        e = p.communicate()[1].decode('utf-8').replace('\n', '\n\t')
        return f" \u2192 Log Message:\n\t{l}\n \u2192 Error Message:\n\t{e}"


    def check_http_server_availability(self, server, pid):
        total_wait_time = 60  # 10 minutes in seconds
        retry_interval = 1    # Interval between attempts (in seconds)
        elapsed_time = 0

        while elapsed_time < total_wait_time:
            try:
                response = requests.head(server)
                if response.status_code == 200:
                    self.echo_note(f"Server: {server} is available after {elapsed_time} sec.")
                    return
            except requests.ConnectionError:
                pass
            if pid.poll() is not None:
                return f"The server container has stopped for an unexpected reason.\n{self.process_output(pid)}"

            time.sleep(retry_interval)
            elapsed_time += retry_interval

        return f"Server: {server} is not available after {total_wait_time} seconds, server launching must have failed."
        
    def build_and_run_container(self, serving, docker_image, use_local_cache, env_variables):
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
        if use_local_cache > 0:
            from pathlib import Path
            home = str(Path.home())
            build_args += f"-v {home}/.cache:/root/.cache "

        if serving == "task":
            build_args += "-v {}/:/opt/audio ".format(os.getcwd())
            CMD = "docker run --rm -p 6379:6379 --name test_redis redis/redis-stack-server:latest redis-server /etc/redis-stack.conf --protected-mode no --bind 0.0.0.0 --loglevel debug"
            self.echo_command(CMD)
            p = subprocess.Popen(CMD.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.poll() is not None:
                return f"Redis server failed to start.\n{self.process_output(p)}", None
            time.sleep(2)
        tag = f"test_{os.path.basename(docker_image)}"
        CMD = f'docker build . -f {docker_image} -t linto-stt-test:{tag}'
        self.echo_command(CMD)
        start_time = time.time()
        p = subprocess.Popen(CMD.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
        end_time = time.time()
        if p.poll() != 0:
            return f"Docker build failed.\n{self.process_output(p)}", None
        self.echo_note(f"Docker image has been successfully built in {end_time - start_time:.0f} sec.")
        CMD=f"docker run --rm -p 8080:80 --name test_container --env-file test/.env --gpus all {build_args} linto-stt-test:{tag}"
        self.echo_command(CMD)
        p = subprocess.Popen(CMD.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.poll() is not None:
            return f"Docker container failed to start.\n{self.process_output(p)}", None
        return None, p

    def transcribe(self, command, regex, test_file, error_message, success_message, timeout=None):
        start = time.time()
        res = subprocess.run(command, shell=True, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        end = time.time()
        if res.returncode != 0:
            raise FileNotFoundError(f"Error: {res.stderr.decode('utf-8')}")
        res = res.stdout.decode('utf-8')
        if not re.search(regex, res):
            message = f"{error_message}: The string '{res}' is not matching the regex ({regex}), the server didn't transcribe correctly."
            self.test_failed(message)
            return message
        self.echo_note(f"{success_message} has transcribed {test_file} in {end - start:.0f} sec.")
        return

    def run_test(self, serving, test_file, docker_image, use_local_cache, env_variables):
        import warnings
        warnings.simplefilter("ignore", ResourceWarning)
        regex = ""
        if test_file == "test/bonjour.wav":
            regex = re.compile("[b|B]onjour")
        r, pid=self.build_and_run_container(serving, docker_image, use_local_cache, env_variables)
        if r!=None:
            self.test_failed(r)
            return r
        if serving == "http":
            r=self.check_http_server_availability("http://localhost:8080/healthcheck", pid)
            if r!=None:
                self.test_failed(r)
                return r
            CMD = f'curl -X POST "http://localhost:8080/transcribe" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@{test_file};type=audio/wav"'
            self.echo_command(CMD)
            r = self.transcribe(CMD, regex, test_file, "Error transcription", "HTTP route 'transcribe'")
            if r!=None:
                return r
            CMD = f'python3 test/test_streaming.py --audio_file {test_file}'
            self.echo_command(CMD)
            r = self.transcribe(CMD, regex, test_file, "Error streaming", "HTTP route 'streaming'")
            if r!=None:
                return r
        elif serving == "task":
            # you can be stuck here if the server crashed bc the task will be in the queue forever
            CMD = f"python3 test/test_celery.py {test_file}"
            self.echo_command(CMD)
            r = self.transcribe(CMD, regex, test_file, "Error task", "TASK route", timeout=60)
            if r!=None:
                return r
        self.test_succeeded()
        return True


def generate_whisper_test_setups():
    dockerfiles = ["whisper/Dockerfile.ctranslate2", "whisper/Dockerfile.ctranslate2.cpu",
                       "whisper/Dockerfile.torch", "whisper/Dockerfile.torch.cpu"]

    use_local_caches = [1]  # Add 0 for additional cache usage

    servings = ["task", "http"]

    vads = ["NONE", "false", "auditok", "silero"]
    devices = ["NONE", "cpu", "cuda"]
    models = ["tiny"]

    for use_local_cache in use_local_caches:
        for dockerfile in dockerfiles:
            for device in devices:
                for vad in vads:
                    for model in models:
                        for serving in servings:
                            # try:
                            if dockerfile.endswith("cpu") and device != "cpu":
                                continue
                            envs = ""
                            if vad != "NONE":
                                envs += f"VAD={vad} "
                            if device != "NONE":
                                envs += f"DEVICE={device} "
                            envs += f"MODEL={model}"
    
                            yield serving, "test/bonjour.wav", dockerfile, use_local_cache, envs

def generate_kaldi_test_setups():
    dockerfiles = ["kaldi/Dockerfile"]

    use_local_caches = [1]  # Add 0 for additional cache usage

    servings = ["task", "http"]
    
    for use_local_cache in use_local_caches:
        for dockerfile in dockerfiles:
            for serving in servings:
                envs = ""
                yield serving, "test/bonjour.wav", dockerfile, use_local_cache, envs

def copy_env_file(env_file, key_words_to_remove):
    with open(env_file, "r") as f:
        lines = f.readlines()
    with open("test/.env", "w") as f:
        for line in lines:
            if not any([word in line for word in key_words_to_remove]):
                f.write(line)

@ddt
class TestRunner(unittest.TestCase):
    
    @idata(generate_kaldi_test_setups())
    def test_kaldi_integration(self, setup):
        print()
        if AM_PATH is None or LM_PATH is None:
            self.fail("AM or LM path not provided. Skipping kaldi test.")
        if not os.path.exists(AM_PATH) or not os.path.exists(LM_PATH):
            self.fail(f"AM or LM path not found: {AM_PATH} or {LM_PATH}")
        copy_env_file("kaldi/.envdefault", ["SERVICE_MODE"])
        serving, test_file, dockerfile, use_local_cache, envs = setup
        envs += f"-v {AM_PATH}:/opt/AM -v {LM_PATH}:/opt/LM"
        testobject = TestContainer()
        test_result = testobject.run_test(serving, test_file, dockerfile, use_local_cache, envs)
        if test_result!=True:
            self.fail(test_result)
            
            
    @idata(generate_whisper_test_setups())
    def test_whisper_integration(self, setup):
        print()
        copy_env_file("whisper/.envdefault", ["VAD", "DEVICE", "MODEL", "SERVICE_MODE"])
        serving, test_file, dockerfile, use_local_cache, envs = setup
        testobject = TestContainer()
        test_result = testobject.run_test(serving, test_file, dockerfile, use_local_cache, envs)
        if test_result!=True:
            self.fail(test_result)
            
    def test_whisper_curl_not_existing_file(self):
        print()
        copy_env_file("whisper/.envdefault", ["VAD", "DEVICE", "MODEL", "SERVICE_MODE"])
        serving = "http"
        test_file = "notexisting"
        dockerfile = "whisper/Dockerfile.ctranslate2"
        use_local_cache = 1
        envs = "MODEL=tiny "
        testobject = TestContainer()
        with self.assertRaises(FileNotFoundError):
            testobject.run_test(serving, test_file, dockerfile, use_local_cache, envs)
            
    def test_cuda_on_cpu_dockerfile(self):
        print()
        copy_env_file("whisper/.envdefault", ["VAD", "DEVICE", "MODEL", "SERVICE_MODE"])
        serving = "http"
        test_file = "test/bonjour.wav"
        dockerfile = "whisper/Dockerfile.ctranslate2.cpu"
        use_local_cache = 1
        envs = "MODEL=tiny  DEVICE=cuda"
        testobject = TestContainer(show_failed_tests=False)
        self.assertIn("The server container has stopped for an unexpected reason.", testobject.run_test(serving, test_file, dockerfile, use_local_cache, envs))
        
    def test_model_whisper(self):
        print()
        copy_env_file("whisper/.envdefault", ["VAD", "DEVICE", "MODEL", "SERVICE_MODE"])
        serving = "http"
        test_file = "test/bonjour.wav"
        dockerfile = "whisper/Dockerfile.ctranslate2"
        use_local_cache = 1
        envs = "MODEL=small"
        testobject = TestContainer()
        test_result = testobject.run_test(serving, test_file, dockerfile, use_local_cache, envs)
        if test_result!=True:
            self.fail(test_result)

    def test_vad_whisper(self):
        print()
        copy_env_file("whisper/.envdefault", ["VAD", "DEVICE", "MODEL", "SERVICE_MODE"])
        serving = "http"
        test_file = "test/bonjour.wav"
        dockerfile = "whisper/Dockerfile.ctranslate2"
        use_local_cache = 1
        envs = "VAD=whatever"
        testobject = TestContainer(show_failed_tests=False)
        self.assertIn("The server container has stopped for an unexpected reason.", testobject.run_test(serving, test_file, dockerfile, use_local_cache, envs))


AM_PATH = None
LM_PATH = None

if __name__ == '__main__':
    from configparser import ConfigParser
    config = ConfigParser()

    config.read('test/test_config.ini')
    
    AM_PATH = config.get('kaldi', 'AM_PATH')
    LM_PATH = config.get('kaldi', 'LM_PATH')
    
    unittest.main(verbosity=2)
