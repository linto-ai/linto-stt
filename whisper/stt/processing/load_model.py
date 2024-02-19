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

        if not os.path.isfile(os.path.join(model_type_or_file, "model.bin")) and not max(
            [
                model_type_or_file.startswith(prefix)
                for prefix in ["tiny", "base", "small", "medium", "large"]
            ]
        ):
            # Convert transformer model

            output_dir = os.path.join(
                download_root,
                f"ctranslate2/converters/transformers--{model_type_or_file.replace('/', '--')}",
            )
            logger.info(f"CTranslate2 model in {output_dir}")
            if not os.path.isdir(output_dir):
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

                check_torch_installed()

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

                    subprocess.check_call(
                        [
                            "ct2-transformers-converter",
                            "--model",
                            model_type_or_file,
                            "--output_dir",
                            os.path.realpath(output_dir),
                            "--quantization",
                            "float16",
                        ]
                    )
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
                    # download_root=os.path.join(download_root, f"huggingface/hub/models--guillaumekln--faster-whisper-{model_type_or_file}"),
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
        extension = (
            os.path.splitext(model_type_or_file)[-1] if os.path.isfile(model_type_or_file) else None
        )

        if model_type_or_file in whisper.available_models() or extension == ".pt":
            model = whisper.load_model(
                model_type_or_file,
                device=device,
                download_root=os.path.join(download_root, "whisper"),
            )

        else:
            # Convert HuggingFace model
            import torch

            peft_folder = None

            if extension in [".ckpt", ".bin"]:
                model_path = model_type_or_file
            else:
                # Search for the cached file (download if necessary)
                if os.path.isdir(model_type_or_file):
                    for root, _, files in os.walk(model_type_or_file):
                        if "adapter_config.json" in files:
                            peft_folder = root
                            break
                try:
                    import transformers
                except ImportError:
                    raise ImportError(
                        f"If you are trying to download a HuggingFace model with {model_type_or_file}, please install first the transformers library"
                    )
                from transformers.utils import cached_file

                try:
                    model_path = cached_file(
                        model_type_or_file,
                        "pytorch_model.bin",
                        cache_dir=download_root,
                        use_auth_token=None,
                        revision=None,
                    )
                except Exception as e:
                    try:
                        if isinstance(e, OSError):
                            model_path = cached_file(
                                model_type_or_file,
                                "whisper.ckpt",
                                cache_dir=download_root,
                                use_auth_token=None,
                                revision=None,
                            )
                        else:
                            raise e
                    except:
                        if peft_folder is None:
                            raise RuntimeError(
                                f"Original error: {e}\nCould not find model {model_type_or_file} from HuggingFace nor local folders."
                            )

            # Load HF Model
            if peft_folder is not None:
                import transformers
                from peft import PeftConfig, PeftModel

                peft_config = PeftConfig.from_pretrained(peft_folder)
                base_model = peft_config.base_model_name_or_path

                model = transformers.WhisperForConditionalGeneration.from_pretrained(base_model)
                model = PeftModel.from_pretrained(model, peft_folder)
                hf_state_dict = model.state_dict()
                del model
            else:
                hf_state_dict = torch.load(model_path, map_location="cpu")

            # Rename layers
            for key in list(hf_state_dict.keys()):
                new_key = hf_to_whisper_states(key)
                if new_key is None:
                    hf_state_dict.pop(key)
                elif new_key != key:
                    hf_state_dict[new_key] = hf_state_dict.pop(key)

            # Init Whisper Model and replace model weights
            dims = whisper.model.ModelDimensions(**states_to_dim(hf_state_dict))
            if "proj_out.weight" in hf_state_dict:
                hf_state_dict["decoder.proj_out.weight"] = hf_state_dict.pop("proj_out.weight")
                print("WARNING: Using untied projection layer")
                whisper_model = WhisperUntied(dims)
            else:
                whisper_model = whisper.model.Whisper(dims)
            whisper_model.load_state_dict(hf_state_dict)
            del hf_state_dict
            whisper_model = whisper_model.to(device)
            return whisper_model

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


# Credit: https://github.com/openai/whisper/discussions/830
def hf_to_whisper_states(text):
    import re

    # From Speechbrain
    if text == "_mel_filters":
        return None

    # From PEFT
    if "default" in text:
        # print(f"WARNING: Ignoring {text}")
        return None
    if text.startswith("base_model.model."):
        text = text[len("base_model.model.") :]

    text = re.sub(".layers.", ".blocks.", text)
    text = re.sub(".self_attn.", ".attn.", text)
    text = re.sub(".q_proj.", ".query.", text)
    text = re.sub(".k_proj.", ".key.", text)
    text = re.sub(".v_proj.", ".value.", text)
    text = re.sub(".out_proj.", ".out.", text)
    text = re.sub(".fc1.", ".mlp.0.", text)
    text = re.sub(".fc2.", ".mlp.2.", text)
    text = re.sub(".fc3.", ".mlp.3.", text)
    text = re.sub(".fc3.", ".mlp.3.", text)
    text = re.sub(".encoder_attn.", ".cross_attn.", text)
    text = re.sub(".cross_attn.ln.", ".cross_attn_ln.", text)
    text = re.sub(".embed_positions.weight", ".positional_embedding", text)
    text = re.sub(".embed_tokens.", ".token_embedding.", text)
    text = re.sub("model.", "", text)
    text = re.sub("attn.layer_norm.", "attn_ln.", text)
    text = re.sub(".final_layer_norm.", ".mlp_ln.", text)
    text = re.sub("encoder.layer_norm.", "encoder.ln_post.", text)
    text = re.sub("decoder.layer_norm.", "decoder.ln.", text)
    return text


def states_to_dim(state_dict):
    n_audio_state = len(state_dict["encoder.ln_post.bias"])
    n_text_state = len(state_dict["decoder.ln.bias"])
    return {
        "n_mels": state_dict["encoder.conv1.weight"].shape[1],  # 80
        "n_vocab": state_dict["decoder.token_embedding.weight"].shape[0],  # 51864 / 51865
        "n_audio_ctx": state_dict["encoder.positional_embedding"].shape[0],  # 1500
        "n_audio_state": n_audio_state,  # 384 / 512 / 768 / 1024 / 1280
        "n_audio_head": n_audio_state // 64,  # 6 / 8 / 12 / 16 / 20
        "n_audio_layer": len(
            set([".".join(k.split(".")[:3]) for k in state_dict.keys() if "encoder.blocks." in k])
        ),  # 4 / 6 / 12 / 24 / 32
        "n_text_ctx": state_dict["decoder.positional_embedding"].shape[0],  # 448
        "n_text_state": n_text_state,  # 384 / 512 / 768 / 1024 / 1280
        "n_text_head": n_text_state // 64,  # 6 / 8 / 12 / 16 / 20
        "n_text_layer": len(
            set([".".join(k.split(".")[:3]) for k in state_dict.keys() if "decoder.blocks." in k])
        ),  # 4 / 6 / 12 / 24 / 32
    }


if not USE_CTRANSLATE2:

    class TextDecoderUntied(whisper.model.TextDecoder):
        """
        Same as TextDecoder but with untied weights
        """

        def __init__(self, *args, **kwargs):
            import torch

            super().__init__(*args, **kwargs)

            n_vocab, n_state = self.token_embedding.weight.shape

            self.proj_out = torch.nn.Linear(n_state, n_vocab, bias=False)

        def forward(self, x, xa, kv_cache=None):
            offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
            x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
            x = x.to(xa.dtype)

            for block in self.blocks:
                x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

            x = self.ln(x)

            # logits = self.proj_out(x).float()
            # logits = (x @ torch.transpose(self.proj_out.weight.to(x.dtype), 0, 1)).float()
            logits = self.proj_out.to(x.dtype)(x).float()

            return logits

    class WhisperUntied(whisper.model.Whisper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.decoder = TextDecoderUntied(
                self.dims.n_vocab,
                self.dims.n_text_ctx,
                self.dims.n_text_state,
                self.dims.n_text_head,
                self.dims.n_text_layer,
            )
