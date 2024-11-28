import unittest
from ddt import ddt, idata
from core import TestRunner, finalize_tests
from automated_utils import config, copy_env_file, TESTDIR


def generate_whisper_test_setups(
    device="cpu", vads=[None, "false", "auditok", "silero"]
):
    # reduce the number of tests because it takes multiples hours
    if device == "cpu":
        dockerfiles = [
            "whisper/Dockerfile.ctranslate2.cpu",
            "whisper/Dockerfile.torch.cpu",
        ]
    elif device == "cuda":
        dockerfiles = [
            "whisper/Dockerfile.ctranslate2",
            "whisper/Dockerfile.torch",
        ]
    else:
        dockerfiles = [
            "whisper/Dockerfile.ctranslate2",
            "whisper/Dockerfile.ctranslate2.cpu",
            "whisper/Dockerfile.torch",
            "whisper/Dockerfile.torch.cpu",
        ]

    servings = ["http", "task"]

    models = ["tiny"]

    for dockerfile in dockerfiles:
        for vad in vads:
            for model in models:
                for serving in servings:
                    env_variables = ""
                    if vad:
                        env_variables += f"VAD={vad} "
                    if device:
                        env_variables += f"DEVICE={device} "
                    env_variables += f"MODEL={model}"

                    yield dockerfile, serving, env_variables


@ddt
class WhisperTestRunner(TestRunner):

    @idata(generate_whisper_test_setups(device="cpu"))
    def test_04_integration_cpu(self, setup):
        dockerfile, serving, env_variables = setup
        copy_env_file("whisper/.envdefault", env_variables)
        self.run_test(dockerfile, serving=serving, env_variables=env_variables)

    @idata(generate_whisper_test_setups(device="cuda", vads=[None, "silero"]))
    def test_05_integration_cuda(self, setup):
        dockerfile, serving, env_variables = setup
        copy_env_file("whisper/.envdefault", env_variables)
        self.run_test(dockerfile, serving=serving, env_variables=env_variables)

    @idata(generate_whisper_test_setups(device=None, vads=[None]))
    def test_06_integration_nodevice(self, setup):
        dockerfile, serving, env_variables = setup
        copy_env_file("whisper/.envdefault", env_variables)
        self.run_test(dockerfile, serving=serving, env_variables=env_variables)

    def test_02_failures_cuda_on_cpu_dockerfile(self):
        env_variables = "MODEL=tiny  DEVICE=cuda"
        dockerfile = "whisper/Dockerfile.ctranslate2.cpu"
        copy_env_file("whisper/.envdefault", env_variables)
        self.assertIn(
            "cannot open shared object file",
            self.run_test(dockerfile, env_variables=env_variables, expect_failure=True),
        )

    def test_02_failure_not_existing_file(self):
        env_variables = "MODEL=tiny"
        copy_env_file("whisper/.envdefault", env_variables)
        with self.assertRaises(FileNotFoundError):
            self.run_test(
                test_file="notexisting",
                env_variables=env_variables,
                expect_failure=True,
            )
        self.cleanup()

    def test_02_failure_wrong_vad(self):
        env_variables = "VAD=whatever MODEL=tiny"
        copy_env_file("whisper/.envdefault", env_variables)
        self.assertIn(
            "Got unexpected VAD method whatever",
            self.run_test(env_variables=env_variables, expect_failure=True),
        )

    def test_03_model(self):
        env_variables = "MODEL=small"
        copy_env_file("whisper/.envdefault", env_variables)
        self.run_test(env_variables=env_variables)
        
    def test_01_failure_wrong_language(self):
        env_variables = "MODEL=tiny LANGUAGE=whatever"
        copy_env_file("whisper/.envdefault", env_variables)
        self.assertIn(
            "ValueError: Language \'whatever\' is not available",
            self.run_test(env_variables=env_variables, expect_failure=True),
        )

    def test_01_nolanguage(self):
        env_variables = "MODEL=tiny LANGUAGE=*"
        copy_env_file("whisper/.envdefault", env_variables)
        self.run_test(env_variables=env_variables)
    
    def test_01_russian(self):
        env_variables = "MODEL=tiny LANGUAGE=ru"
        copy_env_file("whisper/.envdefault", env_variables)
        self.run_test(env_variables=env_variables)
        
    def test_01_language_over_config(self):
        env_variables = "MODEL=tiny LANGUAGE=ru"
        copy_env_file("whisper/.envdefault", env_variables)
        self.run_test(env_variables=env_variables, language="fr")

    def test_01_russian_celery(self):
        env_variables = "MODEL=tiny LANGUAGE=ru"
        copy_env_file("whisper/.envdefault", env_variables)
        self.run_test(serving="task", env_variables=env_variables)
        
    def test_01_language_over_config_celery(self):
        env_variables = "MODEL=tiny LANGUAGE=ru"
        copy_env_file("whisper/.envdefault", env_variables)
        self.run_test(serving="task", env_variables=env_variables, language="fr")



if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        finalize_tests()
