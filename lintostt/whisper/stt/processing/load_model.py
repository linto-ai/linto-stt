import os
import shutil
import subprocess
import sys
import time

from stt import USE_CTRANSLATE2, logger

if USE_CTRANSLATE2:
    import faster_whisper
else:
    import whisper_timestamped as whisper


def load_whisper_model(model_type_or_file, device="cpu", download_root=None):
    start = time.time()

    logger.info("Loading Whisper model {}...".format(model_type_or_file))

    default_cache_root = os.path.join(os.path.expanduser("~"), ".cache")
    if download_root is None:
        download_root = default_cache_root

    if USE_CTRANSLATE2:
        if not os.path.isdir(model_type_or_file):
            # Note: There is no good way to set the root cache directory
            #       with the current version of faster_whisper:
            #       if "download_root" is specified to faster_whisper.WhisperModel
            #       (or "output_dir" in faster_whisper.utils.download_model),
            #       then files are downloaded directly in it without symbolic links
            #       to the cache directory. So it's different from the behavior
            #       of the huggingface_hub.
            #       So we try to create a symbolic link to the cache directory that will be used by HuggingFace...
            if not os.path.exists(download_root):
                if not os.path.exists(default_cache_root):
                    os.makedirs(download_root)
                    if default_cache_root != download_root:
                        os.symlink(download_root, default_cache_root)
                else:
                    os.symlink(default_cache_root, download_root)
            elif not os.path.exists(default_cache_root):
                os.symlink(download_root, default_cache_root)

        if device == "cpu":
            compute_types = ["int8", "float32"]
        else:
            compute_types = ["int8", "int8_float16", "float16", "float32"]

        device_index = 0
        if device.startswith("cuda:"):
            device_index = [int(dev) for dev in device[5:].split(",")]
            device = "cuda"

        if not os.path.isfile(os.path.join(model_type_or_file, "model.bin")) and model_type_or_file not in faster_whisper.utils.available_models():
            # Convert transformer model

            output_dir = os.path.join(
                download_root,
                f"ctranslate2/converters/transformers--{model_type_or_file.replace('/', '--')}",
            )
            logger.info(f"CTranslate2 model in {output_dir}")
            if not os.path.isdir(output_dir):

                check_torch_installed()

                from transformers.utils import cached_file
                import json

                kwargs = dict(cache_dir=download_root, use_auth_token=None, revision=None)
                delete_hf_path = False
                if not os.path.isdir(model_type_or_file):
                    model_path = None
                    hf_path = None
                    for candidate in ["pytorch_model.bin", "model.safetensors", "whisper.ckpt", "pytorch_model.bin.index.json", "model.safetensors.index.json"]:
                        try:
                            hf_path = model_path = cached_file(model_type_or_file, candidate, **kwargs)
                        except OSError:
                            continue
                        if candidate.endswith("index.json"):
                            index_file = model_path
                            mapping = json.load(open(index_file))
                            assert "weight_map" in mapping
                            assert isinstance(mapping["weight_map"], dict)
                            model_path = list(set(mapping["weight_map"].values()))
                            folder = os.path.dirname(index_file)
                            model_path = [os.path.join(folder, p) for p in model_path]
                        break
                    if model_path is None:
                        raise RuntimeError(f"Could not find model {model_type_or_file} from HuggingFace nor local folders.")
                    hf_path = os.path.dirname(os.path.dirname(os.path.dirname(hf_path)))
                    delete_hf_path = not os.path.exists(hf_path)
                else:
                    hf_path = None
                    for candidate in ["pytorch_model.bin", "model.safetensors", "whisper.ckpt", "pytorch_model.bin.index.json", "model.safetensors.index.json"]:
                        model_path = os.path.join(model_type_or_file, candidate)
                        if os.path.exists(model_path):
                            hf_path = model_path
                            break
                    if hf_path is None:
                        raise RuntimeError(f"Could not find pytorch_model.bin in {model_type_or_file}")

                # from ctranslate2.converters.transformers import TransformersConverter
                # converter = TransformersConverter(
                #     model_type_or_file,
                #     activation_scales=None, # Path to the pre-computed activation scales, see https://github.com/mit-han-lab/smoothquant
                #     copy_files=[], # Note: "tokenizer.json" does not always exist, we will copy it separately
                #     load_as_float16=False,
                #     revision=None,
                #     low_cpu_mem_usage=False,
                #     trust_remote_code=False,
                # )

                try:
                    # converter.convert(
                    #     output_dir,
                    #     force=False
                    # )
                    cmd = [
                        "ct2-transformers-converter",
                        "--model",
                        model_type_or_file,
                        "--output_dir",
                        os.path.realpath(output_dir),
                        "--quantization",
                        "float16",
                    ]

                    logger.info(f"Converting {model_type_or_file} to {output_dir} with:\n{' '.join(cmd)}")
                    subprocess.check_call(cmd)
                except Exception as err:
                    shutil.rmtree(output_dir, ignore_errors=True)
                    raise err

                finally:
                    if delete_hf_path:
                        logger.info(f"Deleting {hf_path}")
                        shutil.rmtree(hf_path, ignore_errors=True)

                assert os.path.isdir(output_dir), f"Failed to build {output_dir}"

            model_type_or_file = output_dir

        model = None
        for i, compute_type in enumerate(compute_types):
            try:
                model = faster_whisper.WhisperModel(
                    model_type_or_file,
                    device=device,
                    device_index=device_index,
                    compute_type=compute_type,
                    # cpu_threads=0,  # Can be controled with OMP_NUM_THREADS
                    # num_workers=1,
                    download_root=os.path.join(download_root, f"huggingface/hub"),
                )
                logger.info(f"Whisper model loaded with compute_type={compute_type}. (t={time.time() - start}s)")
                break
            except ValueError as err:
                logger.info(
                    "WARNING: failed to load model with compute_type={}".format(compute_type)
                )
                # On some old GPU we may have the error
                # "ValueError: Requested int8_float16 compute type,
                # but the target device or backend do not support efficient int8_float16 computation."
                if i == len(compute_types) - 1:
                    raise err

    else:
        model = whisper.load_model(
            model_type_or_file,
            device=device,
            download_root=download_root,
        )
        model.eval()
        model.requires_grad_(False)

    logger.info("Whisper model loaded. (t={}s)".format(time.time() - start))

    return model


def check_torch_installed():
    try:
        import torch
    except ImportError:
        # Install transformers with torch
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers[torch]>=4.23"])

        # # Re-load ctranslate2
        # import importlib
        # import ctranslate2
        # importlib.reload(ctranslate2)
        # importlib.reload(ctranslate2.converters.transformers)

    # import torch

