import json
import sys
import string
import numpy as np
import logging
import asyncio
import re
import os
import time
import nemo.collections.asr as nemo_asr

from concurrent.futures import ThreadPoolExecutor
from .vad import remove_non_speech
from .utils import get_language
from .decoding import nemo_transcribe
from linto_stt.punctuation.recasepunc import apply_recasepunc
from linto_stt.engines.nemo.stt import (
    logger,
    VAD, VAD_DILATATION, VAD_MIN_SPEECH_DURATION, VAD_MIN_SILENCE_DURATION,
    STREAMING_BUFFER_TRIMMING_SEC, STREAMING_MIN_CHUNK_SIZE, STREAMING_TIMEOUT_FOR_SILENCE,
    STREAMING_FINAL_MIN_DURATION, STREAMING_FINAL_MAX_DURATION, STREAMING_PAUSE_FOR_FINAL,
    STREAMING_MAX_WORDS_IN_BUFFER, STREAMING_MAX_PARTIAL_ACTUALIZATION_PER_SECOND
)
from websockets.legacy.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger("__streaming__")
logger.setLevel(logging.INFO)
if os.environ.get("DEBUG", "0") in ("1", "true"):
    logger.setLevel(logging.DEBUG)

EOF_REGEX = re.compile(r' *\{.*"eof" *: *1.*\} *$')


def bytes_to_array(bytes):
    return np.frombuffer(bytes, dtype=np.int16).astype(np.float32) / 32768


def processor_output_to_text(o, punctuation_model):
    if o[0] is None:
        return ""
    res = o[2]
    if punctuation_model is not None:
        res = apply_recasepunc(punctuation_model, res)
    return res


def norm_str(text):
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def nemo_to_json(o, partial=False, punctuation_model=None):
    result = dict()
    key = "partial" if partial else "text"
    if isinstance(o, list):
        result[key] = ""
        for i in o:
            result[key] += processor_output_to_text(i, punctuation_model)
    else:
        result["partial" if partial else "text"] = processor_output_to_text(
            o, punctuation_model)
    json_res = json.dumps(result)
    return json_res


async def wssDecode(ws: WebSocketServerProtocol, model_and_alignementmodel):
    """Async Decode function endpoint"""
    try:
        res = await ws.recv()
        try:
            config = json.loads(res)["config"]
            sample_rate = config["sample_rate"]
            logger.info(f"Received config: {config}")
        except Exception as e:
            logger.error(f"Failed to read stream configuration {e}")
            await ws.close(reason="Failed to load configuration")

        model, punctuation_model = model_and_alignementmodel
        language = get_language(config.get("language"))
        streaming_processor = StreamingASRProcessor(model,
                                                    kwargs={"language": language},
                                                    buffer_trimming=STREAMING_BUFFER_TRIMMING_SEC, pause_for_final=STREAMING_PAUSE_FOR_FINAL, max_words_in_buffer=STREAMING_MAX_WORDS_IN_BUFFER,
                                                    vad=VAD, dilatation=VAD_DILATATION, min_silence_duration=VAD_MIN_SILENCE_DURATION, min_speech_duration=VAD_MIN_SPEECH_DURATION,
                                                    final_min_duration=STREAMING_FINAL_MIN_DURATION,
                                                    final_max_duration=STREAMING_FINAL_MAX_DURATION,
                                                    streaming_pause_for_final=STREAMING_PAUSE_FOR_FINAL
                                                    )
        logger.info("Starting transcription ...")
        executor = ThreadPoolExecutor()
        if STREAMING_MAX_PARTIAL_ACTUALIZATION_PER_SECOND > 0:
            partial_actualization = 1 / \
                (STREAMING_MAX_PARTIAL_ACTUALIZATION_PER_SECOND+1)
        else:
            partial_actualization = None
        current_task = None
        received_chunk_size = None
        last_responce_time = None
        pile = []
        timeout = None  # it will be computed after the first chunk is received, it is for finding silence in the input stream
        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                assert message is not None, "Received None message"
            except asyncio.TimeoutError:
                logger.debug(
                    f"Timeout after {timeout:.2f}s, checking for silence")
                message = None
            if (isinstance(message, str) and re.match(EOF_REGEX, message)):
                # End of file: send the last prediction
                final = []
                if current_task:    # wait for the last asynchronous prediction to finish
                    o, _ = await current_task
                    final.append(o)
                # make a last prediction in case chunk was too small
                o, _ = streaming_processor.process_iter()
                final.append(o)
                logger.debug(f"Last committed text: {o}")
                b = streaming_processor.finish()
                final.append(b)
                logger.debug(f"Last buffered text: {o}")
                await ws.send(nemo_to_json(final, punctuation_model=punctuation_model))
                await ws.close()
                logger.info("Closing connection")
                break

            if message is None:
                # No message received because of timeout: VAD is performed on the client side
                silence_chunk = np.zeros(
                    int(sample_rate * received_chunk_size), dtype=np.float32)
                off = streaming_processor.buffer_time_offset
                dur = len(streaming_processor.audio_buffer) / \
                    streaming_processor.sampling_rate
                streaming_processor.insert_audio_chunk(silence_chunk)
                logger.debug(
                    f"Silence chunk inserted ({(len(silence_chunk)/streaming_processor.sampling_rate):.2f}s) at {off:.2f} for {dur:.2f} (now {(len(streaming_processor.audio_buffer)/streaming_processor.sampling_rate):.2f})")
            else:
                if len(pile) < 2:
                    pile.append(message)
                audio_chunk = bytes_to_array(message)
                if received_chunk_size is None:
                    received_chunk_size = len(audio_chunk)/sample_rate
                    if STREAMING_TIMEOUT_FOR_SILENCE:
                        timeout = received_chunk_size * STREAMING_TIMEOUT_FOR_SILENCE
                    else:
                        timeout = None
                streaming_processor.insert_audio_chunk(audio_chunk)
                logger.debug(
                    f"Received chunk of {len(audio_chunk)/sample_rate:.2f}s")
            if streaming_processor.get_buffer_size() >= STREAMING_MIN_CHUNK_SIZE:
                if current_task and not current_task.done():
                    continue
                else:
                    if current_task:    # if the task is done, get the result
                        o, p = await current_task
                        if o[0] is not None:
                            logger.info(f"Sending final '{o}'")
                            await ws.send(nemo_to_json(o, punctuation_model=punctuation_model))
                            last_responce_time = None
                        if p[0] is not None:
                            t = time.time()
                            if last_responce_time is None or partial_actualization is None or t-last_responce_time > partial_actualization or len(pile) == 0:
                                logger.debug(f"Sending partial '{p}'")
                                await ws.send(nemo_to_json(p, partial=True))
                                last_responce_time = t
                        current_task = None
                    # if there are messages in the pile, launch a new transcription task
                    if len(pile) > 0:
                        logger.debug(
                            f"Launching new transcription on {(len(streaming_processor.audio_buffer)/streaming_processor.sampling_rate)+streaming_processor.buffer_time_offset:.2f}s")
                        current_task = asyncio.get_event_loop().run_in_executor(
                            executor, streaming_processor.process_iter)
                        pile.pop(0)
            else:
                logger.debug(
                    f"Chunk too small {streaming_processor.get_buffer_size()}<{STREAMING_MIN_CHUNK_SIZE} (added {len(audio_chunk)/sample_rate}), skipping")
    except ConnectionClosed as e:
        logger.info(f"Connection closed {e}")


class StreamingASRProcessor:

    def __init__(
        self,
        model: nemo_asr.models.EncDecCTCModel,
        buffer_trimming=15,
        pause_for_final=1.5,
        buffer_trimming_words=None,
        vad="auditok",
        logfile=sys.stderr,
        sample_rate=16000,
        min_speech_duration=0.1,
        min_silence_duration=0.1,
        dilatation=0.5,
        max_words_in_buffer=10,
        final_min_duration=2,
        final_max_duration=10,
        streaming_pause_for_final=1,
        kwargs={},
    ):
        self.model: nemo_asr.models.EncDecCTCModel = model
        self.logfile = logfile
        self.model.eval()
        self.init()

        self.buffer_trimming_sec = buffer_trimming
        self.buffer_trimming_words = buffer_trimming_words
        self.pause_for_final = pause_for_final
        self.vad = vad
        self.vad_dilatation = dilatation
        self.vad_min_speech_duration = min_speech_duration
        self.vad_min_silence_duration = min_silence_duration
        self.sampling_rate = sample_rate
        self.max_words_in_buffer = max_words_in_buffer
        self.final_min_duration = final_min_duration
        self.final_max_duration = final_max_duration
        self.streaming_pause_for_final = streaming_pause_for_final
        self.kwargs = kwargs

    def init(self):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0

        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.commited = []
        self.last_chunked_at = 0

        self.silence_iters = 0
        self.buffered_final = []

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def get_buffer_size(self):
        return len(self.audio_buffer) / self.sampling_rate

    def format_words(self, words, timestamps_convert_function=None):
        word_list = []
        for i in words:
            if timestamps_convert_function is not None:
                start, end = timestamps_convert_function(i['start'], i['end'])
                t = (start, end, i['word'])
            else:
                t = ((i['start'], i['end'], i['word']))
            word_list.append(t)
        return word_list

    def get_ends(self, words):
        word_list = []
        for i in words:
            word_list.append(i['end'])
        return word_list

    def transcribe(self, audio_speech, conversion_function):
        hypothesis = nemo_transcribe(self.model, [audio_speech], self.kwargs | {"verbose": False})[0]
        formatted_words = self.format_words(
            hypothesis.timestamp['word'], conversion_function if self.vad else None)
        return formatted_words, hypothesis

    def process_iter(self):
        if self.vad:
            np_buffer = np.array(self.audio_buffer)
            audio_speech, segments, conversion_function = remove_non_speech(
                np_buffer,
                method=self.vad,
                use_sample=True,
                sample_rate=self.sampling_rate,
                dilatation=self.vad_dilatation,
                min_speech_duration=self.vad_min_speech_duration,
                min_silence_duration=self.vad_min_silence_duration,
            )
        else:
            audio_speech = self.audio_buffer
            conversion_function = None
        try:
            formatted_words, raw_transcript = self.transcribe(
                audio_speech, conversion_function)
        except Exception as e:
            # Audio might be empty, or the model might not be able to process it
            logger.error(f"Encoutered an error while transcribing: {e}")
            return (None, None, ""), self.to_flush(self.buffered_final.copy())

        self.transcript_buffer.insert(formatted_words, self.buffer_time_offset)
        o, buffer = self.transcript_buffer.flush(self.max_words_in_buffer)
        self.commited.extend(o)         # contains all text that is commited
        self.buffered_final.extend(o)   # contains text for final
        if (buffer and (self.buffer_time_offset + len(self.audio_buffer) / self.sampling_rate) - buffer[-1][1] < 0.05):
            # remove the last word if it is too close to the end of the buffer
            buffer.pop(-1)
        if len(self.audio_buffer) / self.sampling_rate > self.buffer_trimming_sec:
            self.chunk_completed_segment(
                raw_transcript.timestamp['word'],
                chunk_silence=self.vad,
                speech_segments=segments if self.vad else False,
            )

        logger.debug(
            f"Len of buffer now: {len(self.audio_buffer)/self.sampling_rate:2.2f}s"
        )
        final = self.make_final()
        partial = self.buffered_final.copy()
        partial.extend(buffer)
        return final, self.to_flush(partial)

    def make_final(self):
        final = (None, None, "")
        end_word = None
        for item in reversed(self.buffered_final):
            if item[2] and item[2][-1] in [".", "!", "?"] and item[1]-self.buffered_final[0][0] > self.final_min_duration:
                end_word = item[1]
                logger.info(f"FINAL: punctuation detected: {item}")
                break
        if end_word is None:
            for prev, curr in zip(reversed(self.buffered_final[:-1]), reversed(self.buffered_final[1:])):
                if curr[0] - prev[1] >= self.streaming_pause_for_final and prev[1]-self.buffered_final[0][0] > self.final_min_duration:
                    end_word = prev[1]
                    logger.info(
                        f"FINAL: silence detected: {curr[0] - prev[1]}>{self.streaming_pause_for_final} (between {prev} and {curr})")
                    break
        if end_word is None and len(self.buffered_final) > 1:
            buffered_final_duration = self.buffered_final[-1][1] - \
                self.buffered_final[0][0]
            if buffered_final_duration > self.final_max_duration:
                end_word = self.buffered_final[-1][1]
                logger.info(
                    f"FINAL: max duration reached: {buffered_final_duration}>{self.final_max_duration} ({self.buffered_final})")
        if end_word:
            # assemble the final
            f = []
            for i in self.buffered_final:
                if i[1] > end_word:
                    break
                f.append(i)
            final = self.to_flush(f)
            self.buffered_final = self.buffered_final[len(f):]
        return final

    def to_flush(
        self,
        words,
        sep=" ",
        offset=0,
    ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        t = sep.join(s[2] for s in words)
        if len(words) == 0:
            b = None
            e = None
        else:
            b = offset + words[0][0]
            e = offset + words[-1][1]
        return (b, e, t)

    def chunk_completed_segment(self, res, chunk_silence=False, speech_segments=None):
        # if self.commited == [] and not chunk_silence:
        #     return
        ends = self.get_ends(res)
        if len(ends) > 1 and self.commited:
            t = self.commited[-1][1]
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= t:
                # : {ends[1]+ self.buffer_time_offset} ")
                logger.debug(f"Segment chunked at {e:2.2f}s")
                self.chunk_at(e)
                return
        elif chunk_silence:
            lenght = len(self.audio_buffer) / self.sampling_rate
            e = self.buffer_time_offset + lenght - 2
            if speech_segments:
                end_silence = lenght - speech_segments[-1][1]
                if end_silence > 2:
                    logger.debug(f"Silence segment chunked at {e:2.2f}")
                    self.chunk_at(e)
                    return
            elif speech_segments is not None:
                logger.debug(f"Silence segment chunked at {e:2.2f}")
                self.chunk_at(e)
                return
        logger.debug(f"Not enough segments to chunk")

    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time" """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(
            cut_seconds * self.sampling_rate):]
        self.buffer_time_offset = time
        self.last_chunked_at = time

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()

        self.buffered_final.extend(o)
        f = self.to_flush(self.buffered_final)

        logger.debug(f"last, noncommited:{f}")
        return f


class HypothesisBuffer:

    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_commited_time = 0
        self.last_commited_word = None
        self.last_buffered_time = -1

        self.logfile = logfile

    def insert(self, new, offset):
        # compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t)
                    for a, b, t in new if a > self.last_commited_time - 0.1]
        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):  # 5 is the maximum
                        c = " ".join([self.commited_in_buffer[-j][2]
                                     for j in range(1, i + 1)][::-1])
                        tail = " ".join(self.new[j - 1][2]
                                        for j in range(1, i + 1))
                        if c == tail:
                            logger.debug(f"removing last {i} words:")
                            for j in range(i):
                                logger.debug(f"\t{self.new.pop(0)}")
                            break

    def flush(self, max_partial_words=10):
        # returns commited chunk = the longest common prefix of 2 last inserts.
        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if norm_str(nt) == norm_str(self.buffer[0][2]) or (len(self.buffer) > max_partial_words and max_partial_words > -1):
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit, self.buffer

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer
