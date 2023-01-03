"""
source: https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
"""
from dataclasses import dataclass
import torch

from stt import logger
from .alignment_model import compute_logprobas, get_vocab
from .utils import flatten
from .text_normalize import transliterate


def compute_alignment(audio, transcript, model):
    """ Compute the alignment of the audio and a transcript, for a given model that returns log-probabilities on the charset defined the transcript."""

    emission = compute_logprobas(model, audio)
    labels, blank_id = get_vocab(model)
    labels = labels[:emission.shape[1]]
    dictionary = {c: i for i, c in enumerate(labels)}

    default = labels.index("-") if "-" in labels else None
    tokens = [loose_get_char_index(dictionary, c, default) for c in transcript]
    tokens = flatten(tokens)

    num_emissions = emission.shape[0]
    num_repetitions = count_repetitions(tokens)
    if len(tokens) + num_repetitions > num_emissions:
        # It will be impossible to find a path...
        # It can happen when Whisper is lost in a loop (ex: "Ha ha ha ha ...")
        logger.warn(
            f"Got too many characters from Whisper. Shrinking to the first characters.")
        tokens = tokens[:num_emissions]
        num_repetitions = count_repetitions(tokens)
        while len(tokens) + num_repetitions > num_emissions:
            tokens = tokens[:-1]
            num_repetitions = count_repetitions(tokens)

    # Make sure transcript has the same length as tokens (it could be different just because of transliteration "œ" -> "oe")
    transcript = "".join([labels[i][0] for i in tokens])

    trellis = get_trellis(emission, tokens, blank_id=blank_id)

    path = backtrack(trellis, emission, tokens, blank_id=blank_id)

    segments = merge_repeats(transcript, path)

    word_segments = merge_words(segments)

    return labels, emission, trellis, segments, word_segments


def count_repetitions(tokens):
    return sum([a == b for a, b in zip(tokens[1:], tokens[:-1])])


def loose_get_char_index(dictionary, c, default=None):
    i = dictionary.get(c, None)
    if i is None:
        # Try with alternative versions of the character
        tc = transliterate(c)
        other_char = list(
            set([c.lower(), c.upper(), tc, tc.lower(), tc.upper()]))
        for c2 in other_char:
            i = dictionary.get(c2, None)
            if i is not None:
                i = [i]
                break
        # Some transliterated versions may correspond to multiple characters
        if i is None:
            for c2 in other_char:
                if len(c2) > 1:
                    candidate = [dictionary[c3]
                                 for c3 in c2 if c3 in dictionary]
                    if len(candidate) > 0 and (i is None or len(candidate) > len(i)):
                        i = candidate
        # If still not found
        if i is None:
            logger.warn("Character not correctly handled by alignment model: '" +
                        "' / '".join(list(set([c] + other_char))) + "'")
            i = [default] if default is not None else []
    else:
        i = [i]
    return i


def get_trellis(emission, tokens, blank_id=0, use_max=False):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1)).to(emission.device)
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            torch.maximum(trellis[t, 1:] + emission[t, tokens],
                          # Score for changing to the next token
                          trellis[t, :-1] + emission[t, tokens])
        ) if use_max else torch.logaddexp(
            trellis[t, 1:] + emission[t, blank_id],
            torch.logaddexp(trellis[t, 1:] + emission[t, tokens],
                            trellis[t, :-1] + emission[t, tokens])
        )
    return trellis


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1]
                        if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        logger.warn(f"Failed to align {len(tokens)} tokens")
        return path
    return path[::-1]


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(transcript, path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def merge_words(segments, separator=" "):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / \
                    sum(seg.length for seg in segs)
                words.append(
                    Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words
