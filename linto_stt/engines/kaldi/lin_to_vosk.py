#!/usr/bin/env python3
import os

LANGUAGE_MODEL_PATH = "/opt/LM"
ACOUSTIC_MODEL_PATH = "/opt/AM"
TARGET_PATH = "/opt/model"


def write_word_boundary(phones_path: str, target: str):
    """Kaldi's phones/word_boundary.int from phones.txt: a phone named with a
    word-position suffix (_B, _E, _I, _S) is begin, end, internal or singleton;
    any other phone (silence, noise) is nonword; <eps> and disambiguation
    symbols (#...) are left out."""
    positions = {"B": "begin", "E": "end", "I": "internal", "S": "singleton"}
    with open(phones_path) as f, open(target, "w") as out:
        for line in f:
            fields = line.split()
            if len(fields) != 2:
                continue
            name, index = fields
            if index == "0" or name.startswith("#"):
                continue
            suffix = name.rsplit("_", 1)[-1] if "_" in name else ""
            out.write(f"{index} {positions.get(suffix, 'nonword')}\n")


def lin_to_vosk_format(am_path: str, lm_path: str, target_path: str):
    if os.path.exists(target_path):
        print(
            "Target model folder already exist, assuming model has already been converted. Skipping..."
        )
        return
    os.mkdir(target_path)
    # Create directory structure
    print("Create directory structure")
    for subfolder in ["am", "conf", "graph", "ivector", "rescore"]:
        os.mkdir(os.path.join(target_path, subfolder))

    # Populate am directory
    # final.mdl
    print("Populate am directory")
    for f in ["final.mdl"]:
        print(f)
        os.symlink(os.path.join(am_path, f), os.path.join(target_path, "am", f))

    # Populate conf directory
    print("Populate conf directory")
    print("mfcc.conf")
    os.symlink(
        os.path.join(am_path, "conf", "mfcc.conf"),
        os.path.join(target_path, "conf", "mfcc.conf"),
    )

    print("model.conf")
    with open(os.path.join(target_path, "conf", "model.conf"), "w") as f:
        f.write("--min-active=200\n")
        f.write("--max-active=7000\n")
        f.write("--beam=13.0\n")
        f.write("--lattice-beam=6.0\n")
        f.write("--acoustic-scale=1.0\n")
        f.write("--frame-subsampling-factor=3\n")
        f.write("--endpoint.silence-phones=1:2:3:4:5:6:7:8:9:10\n")
        f.write("--endpoint.rule2.min-trailing-silence=0.5\n")
        f.write("--endpoint.rule3.min-trailing-silence=1.0\n")
        f.write("--endpoint.rule4.min-trailing-silence=2.0\n")

    # Populate graph directory
    print("Populate graph directory")
    for f in ["HCLG.fst", "words.txt"]:
        print(f)
        os.symlink(os.path.join(lm_path, f), os.path.join(target_path, "graph", f))

    print("phones.txt")
    os.symlink(
        os.path.join(am_path, "phones.txt"),
        os.path.join(target_path, "graph", "phones.txt"),
    )

    # Populate graph/phones directory
    os.mkdir(os.path.join(target_path, "graph", "phones"))

    print("Populate graph/phones directory")

    # Vosk needs it to place words in time (else every word is at 0.0). Some
    # decoding graphs do not ship it: derive it from the position-dependent
    # phones, as Kaldi's prepare_lang.sh does.
    print("word_boundary.int")
    word_boundary = os.path.join(lm_path, "word_boundary.int")
    target_word_boundary = os.path.join(target_path, "graph", "phones", "word_boundary.int")
    if os.path.exists(word_boundary):
        os.symlink(word_boundary, target_word_boundary)
    else:
        write_word_boundary(os.path.join(am_path, "phones.txt"), target_word_boundary)

    # Populate ivector directory
    print("Populate graph/phones directory")
    for f in [
        "final.dubm",
        "final.ie",
        "final.mat",
        "global_cmvn.stats",
        "online_cmvn.conf",
    ]:
        print(f)
        os.symlink(
            os.path.join(am_path, "ivector_extractor", f),
            os.path.join(target_path, "ivector", f),
        )

    print("splice.conf")
    with open(os.path.join(am_path, "ivector_extractor", "splice_opts"), "r") as in_f:
        with open(os.path.join(target_path, "ivector", "splice.conf"), "w") as out_f:
            for param in in_f.read().split(" "):
                out_f.write(f"{param}\n")

    # Populate rescore
    # ?


if __name__ == "__main__":
    lin_to_vosk_format(ACOUSTIC_MODEL_PATH, LANGUAGE_MODEL_PATH, TARGET_PATH)
