import math
import torch
import torch.nn.utils.rnn as rnn_utils

from stt import logger


def speechbrain_get_vocab(model):
    tokenizer = model.tokenizer
    labels = [{'': " ", ' ⁇ ': "<pad>"}.get(i, i).lower() for i in tokenizer.decode(
        [[i] for i in range(tokenizer.get_piece_size())])]
    blank_id = labels.index("<pad>")
    return labels, blank_id


# The following limit is to handle the corner Case of too long audio segment (which is better to split it to avoid memory overflow).
# But it is 2240400 / 16000 Hz ~ 140 seconds, which should not happen for segments detected by Whisper (usually one sentence).
# Also note that Whisper works with 30 seconds segment, so there is chance that this limit is never reached.
MAX_LEN = 2240400


def speechbrain_compute_log_probas(model, audios, max_len=MAX_LEN):
    # Single audio
    if not isinstance(audios, list):
        audios = [audios]
        log_probas = speechbrain_compute_log_probas(
            model, audios, max_len=max_len)
        return log_probas[0]

    # Batch of audios (can occur when max_len is reached)
    assert len(audios) > 0, "audios must be a non-empty list"
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
            log_probas_tmp = speechbrain_compute_log_probas(model, chunk)
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

    log_probas = torch.log_softmax(log_probas, dim=-1)
    return log_probas


def pack_sequences(tensors, device="cpu"):
    if len(tensors) == 1:
        return tensors[0].unsqueeze(0).to(device), torch.Tensor([1.]).to(device)
    tensor = rnn_utils.pad_sequence(tensors, batch_first=True)
    wav_lens = [len(x) for x in tensors]
    maxwav_lens = max(wav_lens)
    wav_lens = torch.Tensor([l/maxwav_lens for l in wav_lens])
    return tensor.to(device), wav_lens.to(device)
