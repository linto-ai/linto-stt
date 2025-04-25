import unittest
from ddt import ddt, idata
from core import TestRunner, finalize_tests
from automated_utils import config, copy_env_file, TESTDIR


def generate_nemo_test_setups(
    device="cpu", vads=[None], models=['nvidia/stt_fr_fastconformer_hybrid_large_pc'], architectures=['hybrid_bpe']
):
    dockerfiles = [
        "nemo/Dockerfile",
    ]
    servings = ["http", "task"]

    for dockerfile in dockerfiles:
        for vad in vads:
            for model, architecture in zip(models, architectures):
                for serving in servings:
                    env_variables = ""
                    if vad:
                        env_variables += f"VAD={vad} "
                    if device:
                        env_variables += f"DEVICE={device} "
                    env_variables += f"MODEL={model} "
                    env_variables += f"ARCHITECTURE={architecture}"
                    yield dockerfile, serving, env_variables


@ddt
class WhisperTestRunner(TestRunner):

    @idata(generate_nemo_test_setups(device="cpu", vads=["false"]))
    def test_04_integration_cpu(self, setup):
        dockerfile, serving, env_variables = setup
        copy_env_file("nemo/.envdefault", env_variables)
        self.run_test(dockerfile, serving=serving, env_variables=env_variables)

    @idata(generate_nemo_test_setups(device="cuda", vads=[None, "false", "auditok", "silero"]))
    def test_03_integration_cuda(self, setup):
        dockerfile, serving, env_variables = setup
        copy_env_file("nemo/.envdefault", env_variables)
        self.run_test(dockerfile, serving=serving, env_variables=env_variables)

    def test_02_ctc_bpe(self):
        env_variables = "MODEL=nvidia/stt_fr_conformer_ctc_large ARCHITECTURE=ctc_bpe"
        copy_env_file("nemo/.envdefault", env_variables)
        self.run_test(docker_image="nemo/Dockerfile", env_variables=env_variables)

    def test_01_ctc_bpe_websocket(self):
        env_variables = "MODEL=nvidia/stt_fr_conformer_ctc_large ARCHITECTURE=ctc_bpe"
        copy_env_file("nemo/.envdefault", env_variables)
        self.run_test(docker_image="nemo/Dockerfile", serving="websocket", env_variables=env_variables)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        finalize_tests()
