import os
import re
import configparser

from vosk import Model

def prepare(am_path:str, lm_path:str, config_path:str):
    ''' Prepare folder and configuration files needed for the model usage '''

    if not os.path.isdir(config_path):
        os.mkdir(config_path)

    # load decoder parameters from "decode.cfg"
    decoder_settings = configparser.ConfigParser()
    if not os.path.exists(am_path+'/decode.cfg'):
        raise FileNotFoundError("decode.cfg file is missing")

    decoder_settings.read(am_path+'/decode.cfg')

    # Prepare "online.conf"
    with open(am_path+"/conf/online.conf") as f:
        values = f.readlines()
        with open(config_path+"/online.conf", 'w') as f:
            for i in values:
                f.write(i)
            f.write("--ivector-extraction-config=" +
                    config_path+"/ivector_extractor.conf\n")
            f.write("--mfcc-config=" + os.path.join(am_path, "conf/mfcc.conf") + "\n")
            f.write("--beam=" + decoder_settings.get('decoder_params', 'beam') + "\n")
            f.write("--lattice-beam=" + decoder_settings.get('decoder_params', 'lattice_beam')+"\n")
            f.write("--acoustic-scale=" + decoder_settings.get('decoder_params', 'acwt') + "\n")
            f.write("--min-active=" + decoder_settings.get('decoder_params', 'min_active') + "\n")
            f.write("--max-active=" + decoder_settings.get('decoder_params', 'max_active') + "\n")
            f.write("--frame-subsampling-factor=" + decoder_settings.get('decoder_params', 'frame_subsampling_factor') + "\n")

    # Prepare "ivector_extractor.conf"
    with open(am_path+"/conf/ivector_extractor.conf") as f:
        values = f.readlines()
        with open(config_path+"/ivector_extractor.conf", 'w') as f:
            for i in values:
                f.write(i)
            f.write("--splice-config="+am_path+"/conf/splice.conf\n")
            f.write("--cmvn-config="+am_path +
                    "/conf/online_cmvn.conf\n")
            f.write("--lda-matrix="+am_path +
                    "/ivector_extractor/final.mat\n")
            f.write("--global-cmvn-stats="+am_path +
                    "/ivector_extractor/global_cmvn.stats\n")
            f.write("--diag-ubm="+am_path +
                    "/ivector_extractor/final.dubm\n")
            f.write("--ivector-extractor="+am_path +
                    "/ivector_extractor/final.ie")

    # Prepare "word_boundary.int" if not exist
    if not os.path.exists(lm_path+"/word_boundary.int") and os.path.exists(am_path+"/phones.txt"):
        print("Create word_boundary.int based on phones.txt")
        with open(am_path+"/phones.txt", 'r') as f:
            phones = f.readlines()

        with open(lm_path+"/word_boundary.int", "w") as f:
            for phone in phones:
                phone = phone.strip()
                phone = re.sub('^<eps> .*', '', phone)
                phone = re.sub('^#\d+ .*', '', phone)
                if phone != '':
                    id = phone.split(' ')[1]
                    if '_I ' in phone:
                        f.write(id+" internal\n")
                    elif '_B ' in phone:
                        f.write(id+" begin\n")
                    elif '_E ' in phone:
                        f.write(id+" end\n")
                    elif '_S ' in phone:
                        f.write(id+" singleton\n")
                    else:
                        f.write(id+" nonword\n")

def loadModel(am_path: str, lm_path: str, config_path: str) -> Model:
    """ Load STT model """
    print("MODEL" , os.path.join(config_path, "online.conf"))
    return Model(am_path,lm_path, config_path)
