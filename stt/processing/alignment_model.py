import math
import torch
import torch.nn.utils.rnn as rnn_utils

from stt import logger
from .load_model import get_model_type

import whisper

################################################################################
# Get list of labes (and blank_id) from model


def get_vocab(model):
    type = get_model_type(model)
    if type == "speechbrain":
        labels, blank_id = get_vocab_speechbrain(model)
    elif type == "transformers":
        labels, blank_id = get_vocab_transformers(model)
    else:
        labels, blank_id = get_vocab_torchaudio(model)
    assert isinstance(labels, list) and min(
        [isinstance(l, str) for l in labels]), "labels must be a list of strings"
    return norm_labels(labels, blank_id), blank_id


def get_vocab_speechbrain(model):
    tokenizer = model.tokenizer
    # Is this general enough?
    labels = [{'': " ", ' ⁇ ': "<pad>"}.get(i, i) for i in tokenizer.decode(
        [[i] for i in range(tokenizer.get_piece_size())])]
    blank_id = labels.index("<pad>")
    return labels, blank_id


def get_vocab_torchaudio(model_and_labels):
    _, labels = model_and_labels
    labels = list(labels)
    # WTF : blank_id = labels.index("-") ...? Is it general enough?
    blank_id = 0
    return labels, blank_id


def get_vocab_transformers(model_and_processor):
    _, processor = model_and_processor
    labels_dict = dict((v, k)
                       for k, v in processor.tokenizer.get_vocab().items())
    labels = [labels_dict[i] for i in range(len(labels_dict))]
    blank_id = labels.index("<pad>")
    return labels, blank_id


def norm_labels(labels, blank_id):
    labels[blank_id] = ""
    return [l if l != "|" else " " for l in labels]

################################################################################
# Compute log-probabilities from model


# The following limit is to handle the corner Case of too long audio segment (which is better to split it to avoid memory overflow).
# But it is 2240400 / 16000 Hz ~ 140 seconds, which should not happen for segments detected by Whisper (usually one sentence).
# Also note that Whisper works with 30 seconds segment, so there is chance that this limit is never reached.
MAX_LEN = 2240400


def compute_logprobas(model, audios, max_len=MAX_LEN):

    # Single audio
    if not isinstance(audios, list):
        audios = [audios]
        logits = compute_logprobas(model, audios, max_len=max_len)
        return logits[0]

    # Batch of audios (can occur when max_len is reached)
    assert len(audios) > 0, "audios must be a non-empty list"

    type = get_model_type(model)
    if type == "speechbrain":
        logits = compute_logits_speechbrain(model, audios, max_len)
    elif type == "transformers":
        logits = compute_logits_transformers(model, audios, max_len)
    else:
        logits = compute_logits_torchaudio(model, audios, max_len)

    return torch.log_softmax(logits, dim=-1)


def compute_logits_speechbrain(model, audios, max_len):
    if not isinstance(audios[0], torch.Tensor):
        audios = [torch.from_numpy(a) for a in audios]
    if max([len(a) for a in audios]) > max_len:
        # Split audios into chunks of max_len
        batch_size = len(audios)
        chunks = []
        i_audio = []
        for a in audios:
            chunks.extend([a[i:min(i+max_len, len(a))]
                          for i in range(0, len(a), max_len)])
            i_audio.append(len(chunks))
            if len(chunks) > 1:
                logger.warning(
                    "Audio too long, splitting into {} chunks for alignment".format(len(chunks)))
        # Decode chunks of audio and concatenate results
        log_probas = [[] for i in range(len(audios))]
        for i in range(0, len(chunks), batch_size):
            chunk = chunks[i:min(i+batch_size, len(chunks))]
            log_probas_tmp = compute_logits_speechbrain(model, chunk)
            for j in range(i, i+len(chunk)):
                k = 0
                while j >= i_audio[k]:
                    k += 1
                log_probas[k].append(log_probas_tmp[j-i])
        log_probas = [torch.cat(p, dim=0) for p in log_probas]
        log_probas, wav_lens = pack_sequences(log_probas, device=model.device)
    else:
        batch, wav_lens = pack_sequences(audios, device=model.device)
        log_probas = model.forward(batch, wav_lens)

    return log_probas.cpu().detach()


def pack_sequences(tensors, device="cpu"):
    if len(tensors) == 1:
        return tensors[0].unsqueeze(0).to(device), torch.Tensor([1.]).to(device)
    tensor = rnn_utils.pad_sequence(tensors, batch_first=True)
    wav_lens = [len(x) for x in tensors]
    maxwav_lens = max(wav_lens)
    wav_lens = torch.Tensor([l/maxwav_lens for l in wav_lens])
    return tensor.to(device), wav_lens.to(device)


def compute_logits_transformers(model_and_processor, audios, max_len):

    model, processor = model_and_processor

    # can be different from processor.feature_extractor.sampling_rate
    sample_rate = whisper.audio.SAMPLE_RATE
    device = model.device

    audios = [audio.numpy() for audio in audios]
    processed_batch = processor(audios, sampling_rate=sample_rate)

    padded_batch = processor.pad(
        processed_batch,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors="pt",
    )

    l = padded_batch.input_values.shape[1]

    with torch.inference_mode():
        if l > max_len:
            # Split batch in smaller chunks
            logger.warning(
                "Audio too long, splitting into {} chunks for alignment".format(math.ceil(l / max_len)))
            logits = []
            for i in range(0, l, max_len):
                j = min(i + max_len, l)
                logits.append(model(padded_batch.input_values[:, i:j].to(device),
                                    attention_mask=padded_batch.attention_mask[:, i:j].to(device)).logits)
            logits = torch.cat(logits, dim=1)
        else:
            logits = model(padded_batch.input_values.to(device),
                           attention_mask=padded_batch.attention_mask.to(device)).logits

    return logits.cpu().detach()


def compute_logits_torchaudio(model_and_labels, audios, max_len):
    # TODO: factorize with compute_logits_transformers, and add support for batch of audios

    model, _ = model_and_labels

    all_logits = []

    with torch.inference_mode():
        for audio in audios:
            l = len(audio)
            if l > max_len:
                # Split audio in smaller chunks
                logger.warning(
                    "Audio too long, splitting into {} chunks for alignment".format(math.ceil(l / max_len)))
                logits = []
                for i in range(0, l, max_len):
                    j = min(i + max_len, l)
                    logits.append(model(audio[i:j].unsqueeze(0))[0])
                logits = torch.cat(logits, dim=1)
            else:
                logits, _ = model(audio.unsqueeze(0))

            all_logits.append(logits.cpu().detach())

    assert len(all_logits) == 1  # TODO: support batch of audios

    return all_logits[0]
