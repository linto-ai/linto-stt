import unittest
import os

from core import TestRunner, finalize_tests
from automated_utils import config, copy_env_file, TESTDIR
from ddt import ddt, idata


def generate_kaldi_test_setups():
    dockerfiles = ["kaldi/Dockerfile"]

    servings = ["http", "task"]

    for dockerfile in dockerfiles:
        for serving in servings:
            env_variables = ""
            yield dockerfile, serving, env_variables


@ddt
class KaldiTestRunner(TestRunner):

    @idata(generate_kaldi_test_setups())
    def test_01_integration(self, setup):
        dockerfile, serving, env_variables = setup
        if AM_PATH is None or LM_PATH is None or AM_PATH == "" or LM_PATH == "":
            self.fail("AM or LM path not provided. Skipping kaldi test.")
        if not os.path.exists(AM_PATH) or not os.path.exists(LM_PATH):
            self.fail(f"AM or LM path not found: {AM_PATH} or {LM_PATH}")
        copy_env_file("kaldi/.envdefault")
        env_variables += f"-v {AM_PATH}:/opt/AM -v {LM_PATH}:/opt/LM"
        self.run_test(dockerfile, serving=serving, env_variables=env_variables)


AM_PATH = None
LM_PATH = None

if __name__ == "__main__":
    AM_PATH = config.get("kaldi", "AM_PATH")
    LM_PATH = config.get("kaldi", "LM_PATH")

    try:
        unittest.main(verbosity=2)
    finally:
        finalize_tests()
