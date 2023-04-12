import os
import time

from stt import logger, USE_CTRANSLATE2

if USE_CTRANSLATE2:
    import faster_whisper as whisper
else:
    import whisper_timestamped as whisper

def load_whisper_model(model_type_or_file, device="cpu", download_root="/opt"):

    start = time.time()

    if USE_CTRANSLATE2:
        if not os.path.isdir(model_type_or_file):
            # To specify the cache directory
            model_type_or_file = whisper.utils.download_model(
                model_type_or_file,
                output_dir=os.path.join(download_root, "huggingface/hub")
            )
        model = whisper.WhisperModel(model_type_or_file,
                                     device=device,
                                     compute_type="default",
                                     cpu_threads=0, # Can be controled with OMP_NUM_THREADS
                                     num_workers=1,
                                     )

    else:
        model = whisper.load_model(
            model_type_or_file, device=device,
            download_root=os.path.join(download_root, "whisper")
        )
        model.eval()
        model.requires_grad_(False)

    logger.info("Whisper Model loaded. (t={}s)".format(time.time() - start))

    return model