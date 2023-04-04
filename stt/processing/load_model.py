import whisper_timestamped as whisper

import os
import requests
import huggingface_hub
import speechbrain as sb
import transformers
import torchaudio

import time
from stt import logger

# Sources:
# * https://github.com/m-bain/whisperX (in whisperx/transcribe.py)
# * https://pytorch.org/audio/stable/pipelines.html
# * https://huggingface.co/jonatasgrosman

ALIGNMENT_MODELS = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    # "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    # "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    # "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    # "it": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
    # "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "vi": "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
}


def get_alignment_model(alignment_model_name, language, force = False):
    if alignment_model_name in ["wav2vec", "wav2vec2"]:
        if language is None:
            # Will load alignment model on the fly depending on detected language
            return {}
        elif language in ALIGNMENT_MODELS:
            return ALIGNMENT_MODELS[language]
        elif force:
            raise ValueError(f"No wav2vec alignment model for language '{language}'.")
        else:
            logger.warn(f"No wav2vec alignment model for language '{language}'. Fallback to English.")
            return ALIGNMENT_MODELS["en"]
    elif alignment_model_name in whisper.tokenizer.LANGUAGES.keys():
        return get_alignment_model("wav2vec", alignment_model_name, force = True)
    return alignment_model_name


def load_whisper_model(model_type_or_file, device="cpu", download_root="/opt"):

    start = time.time()

    model = whisper.load_model(model_type_or_file, device=device,
                               download_root=os.path.join(download_root, "whisper"))

    model.eval()
    model.requires_grad_(False)

    logger.info("Whisper Model loaded. (t={}s)".format(time.time() - start))

    return model


def load_alignment_model(source, device="cpu", download_root="/opt"):

    start = time.time()

    if source in torchaudio.pipelines.__all__:
        model = load_torchaudio_model(source, device=device, download_root=download_root)
    else:
        try:
            model = load_transformers_model(source, device=device, download_root=download_root)
        except Exception as err1:
            try:
                model = load_speechbrain_model(source, device=device, download_root=download_root)
            except Exception as err2:
                raise Exception(
                    f"Failed to load alignment model:\n<<< transformers <<<\n{str(err1)}\n<<< speechbrain <<<\n{str(err2)}") from err2

    logger.info(f"Alignment Model of type {get_model_type(model)} loaded. (t={time.time() - start}s)")

    return model


def load_speechbrain_model(source, device="cpu", download_root="/opt"):

    if os.path.isdir(source):
        yaml_file = os.path.join(source, "hyperparams.yaml")
        assert os.path.isfile(
            yaml_file), f"Hyperparams file {yaml_file} not found"
    else:
        try:
            yaml_file = huggingface_hub.hf_hub_download(
                repo_id=source, filename="hyperparams.yaml", cache_dir=os.path.join(download_root, "huggingface/hub"))
        except requests.exceptions.HTTPError:
            yaml_file = None
    overrides = make_yaml_overrides(
        yaml_file, {"save_path": os.path.join(download_root, "speechbrain")})

    savedir = os.path.join(download_root, "speechbrain")
    try:
        model = sb.pretrained.EncoderASR.from_hparams(
            source=source, run_opts={"device": device}, savedir=savedir, overrides=overrides)
    except ValueError:
        model = sb.pretrained.EncoderDecoderASR.from_hparams(
            source=source, run_opts={"device": device}, savedir=savedir, overrides=overrides)

    model.train(False)
    model.requires_grad_(False)
    return model


def load_transformers_model(source, device="cpu", download_root="/opt"):

    model = transformers.Wav2Vec2ForCTC.from_pretrained(source).to(device)
    processor = transformers.Wav2Vec2Processor.from_pretrained(source)

    model.eval()
    model.requires_grad_(False)
    return model, processor


def load_torchaudio_model(source, device="cpu", download_root="/opt"):

    bundle = torchaudio.pipelines.__dict__[source]
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()

    model.eval()
    model.requires_grad_(False)
    return model, labels


def get_model_type(model):
    if not isinstance(model, tuple):
        return "speechbrain"
    assert len(model) == 2, "Invalid model type"
    if isinstance(model[0], transformers.Wav2Vec2ForCTC):
        return "transformers"
    return "torchaudio"


def make_yaml_overrides(yaml_file, key_values):
    """
    return a dictionary of overrides to be used with speechbrain (hyperyaml files)
    yaml_file: path to yaml file
    key_values: dict of key values to override
    """
    if yaml_file is None:
        return None

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
                    override[parent] = override.get(parent, {}) | {
                        child: key_values[child]}
    return override
