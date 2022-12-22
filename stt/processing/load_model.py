import whisper

import os
import requests
import huggingface_hub
import speechbrain as sb

def load_whisper_model(model_type_or_file, device = "cpu", download_root = "/opt"):

    model = whisper.load_model(model_type_or_file, device = device, download_root = os.path.join(download_root, "whisper"))

    model.eval()
    model.requires_grad_(False)
    return model

def load_speechbrain_model(source, device = "cpu", download_root = "/opt"):
    
    if os.path.isdir(source):
        yaml_file = os.path.join(source, "hyperparams.yaml")
        assert os.path.isfile(yaml_file), f"Hyperparams file {yaml_file} not found"
    else:
        try:
            yaml_file = huggingface_hub.hf_hub_download(repo_id=source, filename="hyperparams.yaml", cache_dir = os.path.join(download_root, "huggingface/hub"))
        except requests.exceptions.HTTPError:
            yaml_file = None
    overrides = make_yaml_overrides(yaml_file, {"save_path": os.path.join(download_root, "speechbrain")})

    savedir = os.path.join(download_root, "speechbrain")
    try:
        model = sb.pretrained.EncoderASR.from_hparams(source = source, run_opts= {"device": device}, savedir = savedir, overrides = overrides)
    except ValueError:
        model = sb.pretrained.EncoderDecoderASR.from_hparams(source = source, run_opts= {"device": device}, savedir = savedir, overrides = overrides)

    model.train(False)
    model.requires_grad_(False)
    return model


def make_yaml_overrides(yaml_file, key_values):
    """
    return a dictionary of overrides to be used with speechbrain (hyperyaml files)
    yaml_file: path to yaml file
    key_values: dict of key values to override
    """
    if yaml_file is None: return None

    override = {}
    with open(yaml_file, "r") as f:
        parent = None
        for line in f:
            if line.strip() == "":
                parent = None
            elif line == line.lstrip():
                if ":" in line:
                    parent = line.split(":")[0].strip()
                    if parent in key_values:
                        override[parent] = key_values[parent]
            elif ":" in line:
                child = line.strip().split(":")[0].strip()
                if child in key_values:
                    override[parent] = override.get(parent, {}) | {child: key_values[child]}
    return override
