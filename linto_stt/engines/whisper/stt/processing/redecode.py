"""Decode again the time ranges where Whisper may have hallucinated.

On some windows Whisper writes a subtitle credit or a video outro learned from
its training data ("Sous-titrage Société Radio-Canada", "Merci d'avoir regardé
cette vidéo") instead of the speech, with a high confidence: the temperature
fallback does not retry it, and the rest of the 30 s window is skipped, so the
speech is lost. Whether it happens depends on where the window starts: the same
speech is usually transcribed when the window starts a second later.

After the normal decoding, a time range is suspicious when, inside the speech found
by the VAD, it has no word for at least HALLUCINATION_REDECODE_MIN_GAP seconds
(the words of known hallucinated phrases, see hallucinations.py, not counting as
words), or when it holds a known hallucinated phrase (from the end of the real
word before it to the start of the real word after it, as such a phrase is often
glued to real speech). Each suspicious time range is decoded again as a separate excerpt, in the
language of the whole audio, with these attempts in order: starting at the
time range, with then without the hotwords (they make such hallucinations more
frequent); starting 1 s later, with then without the hotwords; starting 1 s
earlier (2 s later at the start of the audio). Each excerpt runs CONTEXT_AFTER
seconds past the time range (only the words inside the time range are kept). An
attempt is clean when it gives no known hallucinated phrase and not mostly the
same words repeated. The first clean attempt with a plausible amount of words
(MIN_WORDS_PER_SECOND per second of speech in the time range, at least one)
replaces the words of the time range. If none reaches it and the time range held
a known phrase, any clean attempt is better than the phrase: the one with the
most words replaces it, or else one that gives no word at all in the time range
while transcribing the speech after it (there is no speech there: the phrase is
removed). Otherwise the time range keeps what the normal decoding wrote.

Without VAD, the whole audio counts as speech, so pauses and music would look like
lost speech: only the time ranges holding a known phrase are then decoded again.

Applies to offline decoding (not streaming) and needs word timestamps: nothing is
done for plain-text HTTP requests (Accept: text/plain), which have none.
"""
import dataclasses

import regex as re

from linto_stt.engines.whisper.stt import HALLUCINATION_REDECODE_MIN_GAP, logger

from .hallucinations import find_hallucinated_words
from .utils import SAMPLE_RATE

MIN_WORDS_PER_SECOND = 0.5
CONTEXT_AFTER = 5.0  # seconds of audio decoded after a time range (its words are not kept)


def _words(segments):
    return [w for s in segments for w in (s.words or [])]


def _hallucinated(segments):
    """Indexes (in _words(segments)) of the words of known hallucinated phrases.
    faster-whisper splits some words in pieces ("Radio", "-Canada"): a piece
    without a leading space is glued to the previous word before matching (a
    simpler rule than the gluing of the output, enough for the phrases); each
    segment's word range is given to detect the whole-segment fragments."""
    groups, ranges, i = [], [], 0
    for segment in segments:
        first = len(groups)
        for w in segment.words or []:
            if len(groups) > first and w.word and not w.word[0].isspace():
                groups[-1].append(i)
            else:
                groups.append([i])
            i += 1
        if len(groups) > first:
            ranges.append((first, len(groups) - 1))
    words = _words(segments)
    matched = find_hallucinated_words(["".join(words[j].word for j in g) for g in groups], ranges)
    return {j for g in matched for j in groups[g]}


def _time_ranges(words, hallucinated, speech, min_gap):
    """Suspicious time ranges: inside the speech, (1) every time range of at least
    min_gap seconds without a real word (not looked for if min_gap is None), and
    (2) every detected hallucinated
    phrase, from the end of the real word before it to the start of the real word
    after it (it may be glued to real speech). Overlapping time ranges are merged.
    Returns (start, end, holds_a_known_phrase) tuples."""
    real = sorted((w.start, w.end) for i, w in enumerate(words) if i not in hallucinated)
    found = []
    for a, b in speech:
        inside = [(s, e) for s, e in real if e > a and s < b]
        if min_gap is not None:
            t = a
            for start, end in inside + [(b, b)]:
                if start - t >= min_gap:
                    found.append((t, start, False))
                t = max(t, end)
        for i in sorted(hallucinated):
            w = words[i]
            if not (w.end > a and w.start < b):
                continue
            before = max([e for s, e in inside if e <= w.start] + [a])
            after = min([s for s, e in inside if s >= w.end] + [b])
            found.append((before, after, True))
    merged = []
    for a, b, phrase in sorted(found):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b), merged[-1][2] or phrase)
        elif b > a:
            merged.append((a, b, phrase))
    return merged


def _speech_duration(start, end, speech):
    """Seconds of speech (according to `speech`) between start and end."""
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in speech)


def _repetitive(words):
    """Mostly the same words repeated ("Non, non, non, non..."), a hallucination pattern."""
    tokens = [t for t in (re.sub(r"\W", "", w.word.lower()) for w in words) if t]
    return len(tokens) >= 6 and len(set(tokens)) / len(tokens) < 0.4


def _shift(segment, offset, keep):
    """The segment with its times shifted by offset, keeping only the words for which keep(word) is true."""
    words = [dataclasses.replace(w, start=w.start + offset, end=w.end + offset)
             for w in (segment.words or []) if keep(w.start + offset)]
    if not words:
        return None
    return dataclasses.replace(segment, start=words[0].start, end=words[-1].end, words=words,
                               text="".join(w.word for w in words))


def _attempts(start, end, has_hotwords):
    """(excerpt start, with hotwords) of each attempt, in order."""
    shifted = start - 1.0 if start >= 1.0 else start + 2.0
    for s, retry_without_hotwords in ((start, True), (start + 1.0, True), (shifted, False)):
        if s < end:
            yield s, True
            if has_hotwords and retry_without_hotwords:
                yield s, False


def redecode_suspicious(model, audio, segments, info, speech, transcribe_kwargs,
                        min_gap=HALLUCINATION_REDECODE_MIN_GAP):
    """Return segments where the suspicious time ranges were re-decoded (see the
    module docstring). `speech` is the list of (start, end) speech time ranges in
    seconds (the whole audio without VAD), `min_gap` the minimum duration of speech
    without any word that is suspicious (None: only the known phrases are, e.g.
    without VAD, where silences and music would count as speech), `transcribe_kwargs` the keyword
    arguments of the normal model.transcribe() call, except the audio, language,
    word_timestamps and clip_timestamps."""
    words = _words(segments)
    if not words:
        return segments
    time_ranges = _time_ranges(words, _hallucinated(segments), speech, min_gap)
    if not time_ranges:
        return segments
    duration = len(audio) / SAMPLE_RATE
    for a, b, phrase in time_ranges:
        before = " ".join(w.word.strip() for w in words if a <= w.start < b)
        # A plausible amount of words for the speech of the time range, whatever
        # made it suspicious.
        min_words = max(1, MIN_WORDS_PER_SECOND * _speech_duration(a, b, speech))
        accepted = None
        # For a known phrase, any clean decoding is better than the phrase: the
        # best one (most words; else no word while the speech after it is
        # transcribed) is kept in case no attempt reaches min_words.
        fallback, fallback_words = None, -1
        for start, with_hotwords in _attempts(a, b, bool(transcribe_kwargs.get("hotwords"))):
            # Context after the time range: on a short excerpt, Whisper tends to
            # write the same hallucination again.
            excerpt = audio[int(start * SAMPLE_RATE):int(min(b + CONTEXT_AFTER, duration) * SAMPLE_RATE)]
            attempt_kwargs = transcribe_kwargs if with_hotwords else {**transcribe_kwargs, "hotwords": None}
            new_segments, _ = model.transcribe(excerpt, language=info.language, word_timestamps=True,
                                               **attempt_kwargs)
            new_segments = list(new_segments)
            words_after = [w for w in _words(new_segments) if w.start + start >= b]
            new_segments = [s for s in (_shift(s, start, lambda t: a <= t < b) for s in new_segments) if s]
            new_words = _words(new_segments)
            if _hallucinated(new_segments) or _repetitive(new_words):
                continue
            if len(new_words) >= min_words:
                accepted = (new_segments, start, with_hotwords)
                break
            # No word in the time range, while the speech after it is transcribed:
            # no speech there (the known phrase is removed).
            if phrase and len(new_words) > fallback_words and (new_words or words_after):
                fallback, fallback_words = (new_segments, start, with_hotwords), len(new_words)
        accepted = accepted or fallback
        if accepted is None:
            logger.info(f"Suspicious time range {a:.2f}-{b:.2f} s ({before!r}): no better decoding, kept as is")
            continue
        new_segments, start, with_hotwords = accepted
        logger.info(f"Suspicious time range {a:.2f}-{b:.2f} s ({before!r}) re-decoded from {start:.2f} s"
                    f"{'' if with_hotwords else ' without hotwords'}: {len(_words(new_segments))} words")
        segments = [_shift(s, 0.0, lambda t: not a <= t < b) if any(a <= w.start < b for w in (s.words or [])) else s
                    for s in segments]
        segments = [s for s in segments if s] + new_segments
        segments.sort(key=lambda s: s.start)
        words = _words(segments)
    return segments
