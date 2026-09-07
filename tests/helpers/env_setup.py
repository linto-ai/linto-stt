import os
import re

from dotenv import dotenv_values


def read_envdefault(project_root: str, engine: str) -> dict:
    """Parse an engine's .envdefault file into a dict using python-dotenv."""
    env_path = os.path.join(project_root, "linto_stt", "engines", engine, ".envdefault")
    if not os.path.exists(env_path):
        return {}
    # dotenv_values handles inline comments, quoting, etc.
    return {k: v for k, v in dotenv_values(env_path).items() if v is not None}


def build_env_dict(project_root: str, engine: str, overrides: dict = None) -> dict:
    """Merge .envdefault with overrides, removing SERVICE_MODE (set by CLI)."""
    env = read_envdefault(project_root, engine)
    env.pop("SERVICE_MODE", None)
    if overrides:
        # Drop None-valued overrides: a None means "leave unset, use the engine
        # default" (e.g. VAD=None in the config generators). Keeping it would
        # both shadow the .envdefault value and crash subprocess.Popen, which
        # cannot encode a None env value.
        env.update({k: v for k, v in overrides.items() if v is not None})
    return env


def get_expected_regex(test_file: str, language: str = None) -> re.Pattern:
    """Return a compiled regex for expected transcription output given an audio file."""
    if language is None:
        raise ValueError("Language must be set")
    basename = os.path.basename(test_file)
    if basename == "bonjour.wav":
        if language == "ru":
            return re.compile("Б")
        return re.compile("[bB]onjou")
    elif basename == "Hotel20sec.wav":
        return re.compile("[bB]onjour [mM]onsieur")
    raise ValueError(f"Unknown test file {test_file}")
