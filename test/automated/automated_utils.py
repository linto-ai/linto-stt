import os
import re
from configparser import ConfigParser

AUTOMATEDTESTDIR = os.path.dirname(os.path.realpath(__file__))
TESTDIR = os.path.dirname(AUTOMATEDTESTDIR)
ROOTDIR = os.path.dirname(TESTDIR)
os.chdir(ROOTDIR)

config = ConfigParser()
config.read(f"{AUTOMATEDTESTDIR}/test_config.ini")

SERVER_STARTING_TIMEOUT = int(config.get('server', 'STARTING_TIMEOUT')) if config.get('server', 'STARTING_TIMEOUT')!="" else 60

def copy_env_file(env_file, env_variables=""):
    env_variables = env_variables.split()
    env_variables.append("SERVICE_MODE=")
    with open(env_file, "r") as f:
        lines = f.readlines()
    with open(f"{AUTOMATEDTESTDIR}/.env", "w") as f:
        for line in lines:
            if not any([line.startswith(b.split("=")[0] + "=") for b in env_variables]):
                f.write(line)

def parse_env_variables(env_variables):
    # make a dict
    env_variables = env_variables.split()
    env = {}
    for env_variable in env_variables:
        key, value = env_variable.split("=")
        env[key] = value
    return env

def get_file_regex(test_file, language=None):
    if not language:
        raise ValueError("Language must be set")
    if os.path.basename(test_file) == "bonjour.wav":
        if language == "ru":
            return re.compile("Б")
        else :
            return re.compile("[bB]onjour")
    elif "notexisting":
        return re.compile("")
    raise ValueError(f"Unknown test file {test_file}")