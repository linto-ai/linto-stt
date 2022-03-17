#!/usr/bin/env python3
import os
import re
import configparser

LANGUAGE_MODEL_PATH="/opt/LM"
ACOUSTIC_MODEL_PATH="/opt/AM"
TARGET_PATH="/opt/model"

def lin_to_vosk_format(am_path: str, lm_path: str, target_path: str):
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
        os.symlink(os.path.join(am_path, f),
                os.path.join(target_path, "am", f))

    # Populate conf directory
    print("Populate conf directory")
    print("mfcc.conf")
    os.symlink(os.path.join(am_path, "conf", "mfcc.conf"),
            os.path.join(target_path, "conf", "mfcc.conf"))
    
    print("model.conf")
    with open(os.path.join(target_path, "conf", "model.conf"), 'w') as f:
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
        os.symlink(os.path.join(lm_path, f),
                os.path.join(target_path, "graph", f))

    print("phones.txt")
    os.symlink(os.path.join(am_path, "phones.txt"),
                os.path.join(target_path, "graph", "phones.txt"))
    
    # Populate graph/phones directory
    os.mkdir(os.path.join(target_path, "graph", "phones"))
    
    print("Populate graph/phones directory")
    
    print("word_boundary.int")
    os.symlink(os.path.join(lm_path, "word_boundary.int"), 
               os.path.join(target_path, "graph", "phones", "word_boundary.int"))
    
    # Populate ivector directory
    print("Populate graph/phones directory")
    for f in ["final.dubm",  "final.ie",  "final.mat",  "global_cmvn.stats",  "online_cmvn.conf"]:
        print(f)
        os.symlink(os.path.join(am_path, "ivector_extractor", f),
                   os.path.join(target_path, "ivector", f))
    
    print("splice.conf")
    with open(os.path.join(am_path, "ivector_extractor", "splice_opts"), 'r') as in_f:
        with open(os.path.join(target_path, "ivector", "splice.conf"), 'w') as out_f:
            for param in in_f.read().split(" "):
                out_f.write(f"{param}\n")

    # Populate rescore
    # ?

if __name__ == "__main__":
    lin_to_vosk_format(ACOUSTIC_MODEL_PATH, LANGUAGE_MODEL_PATH, TARGET_PATH)

