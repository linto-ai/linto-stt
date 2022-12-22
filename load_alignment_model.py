import os 
import urllib.request
import zipfile

import huggingface_hub
import speechbrain as sb
import requests


def load_alignment_model(name, download_root = "/opt"):
    if name.startswith("linSTT"):
        destdir = os.path.join(download_root, name)
        if not os.path.exists(destdir):
            # Download model
            url = f"https://dl.linto.ai/downloads/model-distribution/acoustic-models/fr-FR/{name}.zip"
            destzip = destdir+".zip"
            if not os.path.exists(destzip):
                print("Downloading", url, "into", destdir)
                os.makedirs(download_root, exist_ok=True)
                urllib.request.urlretrieve(url, destzip)
            with zipfile.ZipFile(destzip, 'r') as z:
                os.makedirs(destdir, exist_ok=True)
                z.extractall(destdir)
            assert os.path.isdir(destdir)
            os.remove(destzip)
    else:
        destdir = name
    load_speechbrain_model(destdir, download_root = download_root)

def load_speechbrain_model(source, device = None, download_root = "/opt"):
    
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
        model = sb.pretrained.EncoderASR.from_hparams(source = source, savedir = savedir, overrides = overrides)
    except ValueError:
        model = sb.pretrained.EncoderDecoderASR.from_hparams(source = source, savedir = savedir, overrides = overrides)
    return model

def make_yaml_overrides(yaml_file, key_values):
    """
    return a dictionary of overrides to be used with speechbrain
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


if __name__ == "__main__":

    import sys
    assert len(sys.argv) in [1, 2], f"Usage: {sys.argv[0]} <model_type_or_file>"
    load_alignment_model(sys.argv[1] if len(sys.argv) > 1 else "linSTT_speechbrain_fr-FR_v1.0.0")
