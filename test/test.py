import unittest
import os
import time
import subprocess
import requests
import re
from ddt import ddt, idata
from pathlib import Path
import warnings

TESTDIR = os.path.dirname(os.path.realpath(__file__))
ROOTDIR = os.path.dirname(TESTDIR)
os.chdir(ROOTDIR)
TESTDIR = os.path.basename(TESTDIR)



def generate_whisper_test_setups():
    dockerfiles = ["whisper/Dockerfile.ctranslate2", "whisper/Dockerfile.ctranslate2.cpu",
                       "whisper/Dockerfile.torch", "whisper/Dockerfile.torch.cpu"]

    servings = ["http", "task"]

    vads = ["NONE", "false", "auditok", "silero"]
    devices = ["NONE", "cpu", "cuda"]
    models = ["tiny"]

    for dockerfile in dockerfiles:
        for device in devices:
            for vad in vads:
                for model in models:
                    for serving in servings:
                        # try:
                        if dockerfile.endswith("cpu") and device != "cpu":
                            continue
                        env_variables = ""
                        if vad != "NONE":
                            env_variables += f"VAD={vad} "
                        if device != "NONE":
                            env_variables += f"DEVICE={device} "
                        env_variables += f"MODEL={model}"

                        yield dockerfile, serving, env_variables

def generate_kaldi_test_setups():
    dockerfiles = ["kaldi/Dockerfile"]

    servings = ["http", "task"]
    
    for dockerfile in dockerfiles:
        for serving in servings:
            env_variables = ""
            yield dockerfile, serving, env_variables

def copy_env_file(env_file, env_variables=""):
    env_variables = env_variables.split()
    env_variables.append("SERVICE_MODE=")
    with open(env_file, "r") as f:
        lines = f.readlines()
    with open(f"{TESTDIR}/.env", "w") as f:
        for line in lines:
            if not any([line.startswith(b.split("=")[0] + "=") for b in env_variables]):
                f.write(line)

@ddt
class TestRunner(unittest.TestCase):

    built_images = []

    def __init__(self, *args, **kwargs):
        super(TestRunner, self).__init__(*args, **kwargs)
        self.cleanup()

    def echo_success(self, message):
        print('\033[0;32m' + u'\u2714' + '\033[0m ' + message)

    def echo_failure(self, message):
        print('\033[0;31m' + u'\u2716' + '\033[0m ' + message)

    def echo_note(self, message):
        print(u'\u231B' + ' ' + message)

    def echo_command(self, message):
        print(f"$ {message}")

    def report_failure(self, message, expect_failure=True):
        if expect_failure:
            self.echo_failure(message)
        self.cleanup()
        if expect_failure:
            self.fail(message)
        return message

    def report_success(self):
        self.echo_success("Test passed.")
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
        total_wait_time = SERVER_STARTING_TIMEOUT  # 10 minutes in seconds
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

        return f"Server: {server} is not available after {total_wait_time} seconds, server launching must have failed.\n{self.process_output(pid)}"
        
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
            # Launch redis
            build_args += "-v {}/:/opt/audio ".format(os.getcwd())
            cmd = "docker run --rm -p 6379:6379 --name test_redis redis/redis-stack-server:latest redis-server /etc/redis-stack.conf --protected-mode no --bind 0.0.0.0 --loglevel debug"
            self.echo_command(cmd)
            p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.poll() is not None:
                self.cleanup()
                return f"Redis server failed to start.\n{self.process_output(p)}", None
            time.sleep(2)

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

        cmd=f"docker run --rm -p 8080:80 --name test_container --env-file {TESTDIR}/.env --gpus all {build_args} linto-stt-test:{tag}"
        self.echo_command(cmd)
        p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
            return self.report_failure(message)
        self.echo_note(f"{success_message} has transcribed {test_file} in {end - start:.0f} sec.")
        return

    def run_test(self, docker_image="whisper/Dockerfile.ctranslate2", serving="http", env_variables="", test_file=f"{TESTDIR}/bonjour.wav", use_local_cache=True, expect_failure=False):
        warnings.simplefilter("ignore", ResourceWarning)
        regex = ""
        if os.path.basename(test_file) == "bonjour.wav":
            regex = re.compile("[bB]onjour")
        r, pid = self.build_and_run_container(serving, docker_image, env_variables, use_local_cache)
        if r:
            return self.report_failure(r, expect_failure=expect_failure)
        if serving == "http":
            r=self.check_http_server_availability("http://localhost:8080/healthcheck", pid)
            if r:
                return self.report_failure(r, expect_failure=expect_failure)
            cmd = f'curl -X POST "http://localhost:8080/transcribe" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@{test_file};type=audio/wav"'
            self.echo_command(cmd)
            r = self.transcribe(cmd, regex, test_file, "Error transcription", "HTTP route 'transcribe'")
            if r:
                return self.report_failure(r, expect_failure=expect_failure)
            cmd = f"python3 {TESTDIR}/test_streaming.py --audio_file {test_file}"
            self.echo_command(cmd)
            r = self.transcribe(cmd, regex, test_file, "Error streaming", "HTTP route 'streaming'")
        elif serving == "task":
            # you can be stuck here if the server crashed bc the task will be in the queue forever
            cmd = f"python3 {TESTDIR}/test_celery.py {test_file}"
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

    @idata(generate_kaldi_test_setups())
    def test_01_kaldi_integration(self, setup):
        dockerfile, serving, env_variables = setup
        if AM_PATH is None or LM_PATH is None or AM_PATH=="" or LM_PATH=="":
            self.fail("AM or LM path not provided. Skipping kaldi test.")
        if not os.path.exists(AM_PATH) or not os.path.exists(LM_PATH):
            self.fail(f"AM or LM path not found: {AM_PATH} or {LM_PATH}")
        copy_env_file("kaldi/.envdefault")
        env_variables += f"-v {AM_PATH}:/opt/AM -v {LM_PATH}:/opt/LM"
        self.run_test(dockerfile, serving=serving, env_variables=env_variables)
            
            
    @idata(generate_whisper_test_setups())
    def test_03_whisper_integration(self, setup):
        dockerfile, serving, env_variables = setup
        copy_env_file("whisper/.envdefault", env_variables)
        self.run_test(dockerfile, serving=serving, env_variables=env_variables)
            
    def test_02_whisper_failures_not_existing_file(self):
        env_variables = "MODEL=tiny"
        copy_env_file("whisper/.envdefault", env_variables)
        with self.assertRaises(FileNotFoundError):
            self.run_test(test_file="notexisting", env_variables=env_variables, expect_failure=False)
        self.cleanup()
            
    def test_02_whisper_failures_cuda_on_cpu_dockerfile(self):
        env_variables = "MODEL=tiny  DEVICE=cuda"
        dockerfile = "whisper/Dockerfile.ctranslate2.cpu"
        copy_env_file("whisper/.envdefault", env_variables)
        self.assertIn("cannot open shared object file", self.run_test(dockerfile, env_variables=env_variables, expect_failure=False))

    def test_02_whisper_failures_wrong_vad(self):
        env_variables = "VAD=whatever MODEL=tiny"
        copy_env_file("whisper/.envdefault", env_variables)
        self.assertIn("Got unexpected VAD method whatever", self.run_test(env_variables=env_variables, expect_failure=False))

    def test_04_model_whisper(self):
        env_variables = "MODEL=small"
        copy_env_file("whisper/.envdefault", env_variables)
        self.run_test(env_variables=env_variables)


AM_PATH = None
LM_PATH = None
SERVER_STARTING_TIMEOUT = 60

if __name__ == '__main__':
    from configparser import ConfigParser
    config = ConfigParser()

    config.read(f"{TESTDIR}/test_config.ini")
    
    SERVER_STARTING_TIMEOUT = int(config.get('server', 'STARTING_TIMEOUT')) if config.get('server', 'STARTING_TIMEOUT')!="" else SERVER_STARTING_TIMEOUT
    
    AM_PATH = config.get('kaldi', 'AM_PATH')
    LM_PATH = config.get('kaldi', 'LM_PATH')
    
    unittest.main(verbosity=2)
