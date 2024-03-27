import json
import sys
import string
import numpy as np
from stt.processing.streaming_vad import remove_non_speech
from stt import logger, USE_CTRANSLATE2, USE_VAD
from websockets.legacy.server import WebSocketServerProtocol
from simple_websocket.ws import Server as WSServer

def bytes_to_array(bytes):
    return np.frombuffer(bytes, dtype=np.int16).astype(np.float32) / 32768

def processor_output_to_text(o):
    if o[0] is None:
        return ""
    return o[2]

def whisper_to_json(o):
    result = dict()
    result["text"] = processor_output_to_text(o)
    json_res = json.dumps(result)
    return json_res

async def wssDecode(ws: WebSocketServerProtocol, model_and_alignementmodel):
    """Async Decode function endpoint"""
    res = await ws.recv()
    try:
        config = json.loads(res)["config"]
        sample_rate = config["sample_rate"]
        logger.info(f"Received config: {config}")
    except Exception as e:
        logger.error("Failed to read stream configuration")
        await ws.close(reason="Failed to load configuration")
    model, _ = model_and_alignementmodel
    if USE_CTRANSLATE2:
        logger.info("Using ctranslate2 for decoding")
        asr = FasterWhisperASR(model=model, lan="fr")
    else:
        logger.info("Using whisper_timestamped for decoding")
        asr = WhisperTimestampedASR(model=model, lan="fr")
    online = OnlineASRProcessor(asr, logfile=sys.stderr, buffer_trimming=8, use_vad=USE_VAD)
    logger.info("Waiting for chunks")
    while True:
        try:
            message = await ws.recv()
            if message is None or message == "":  # Timeout
                logger.info("Connection closed by client")
                ws.close()
        except Exception as e:
            logger.info(f"Connection closed by client: {e}")
            break
        if "eof" in str(message):
            o = online.finish()
            await ws.send(whisper_to_json(o))
            logger.info(f"End of stream {message}")
            await ws.close(reason="End of stream")
            break
        online.insert_audio_chunk(bytes_to_array(message))
        o, _ = online.process_iter()
        logger.info(o)
        await ws.send(whisper_to_json(o))
        
        
def ws_streaming(websocket_server: WSServer, model_and_alignementmodel):
    """Sync Decode function endpoint"""
    res = websocket_server.receive(timeout=10)
    try:
        config = json.loads(res)["config"]
        sample_rate = config["sample_rate"]
        logger.info(f"Received config: {config}")
    except Exception as e:
        logger.error("Failed to read stream configuration")
        websocket_server.close()
    model, _ = model_and_alignementmodel
    if USE_CTRANSLATE2:
        logger.info("Using ctranslate2 for decoding")
        asr = FasterWhisperASR(model=model, lan="fr")
    else:
        logger.info("Using whisper_timestamped for decoding")
        asr = WhisperTimestampedASR(model=model, lan="fr")
    online = OnlineASRProcessor(asr, logfile=sys.stderr, buffer_trimming=8, use_vad=USE_VAD)
    logger.info("Waiting for chunks")
    while True:
        try:
            message = websocket_server.receive(timeout=10)
            if message is None or message == "":  # Timeout
                logger.info("Connection closed by client")
                websocket_server.close()
        except Exception as e:
            logger.info(f"Connection closed by client: {e}")
            break
        if "eof" in str(message):
            o = online.finish()
            websocket_server.send(whisper_to_json(o))
            logger.info(f"End of stream {message}")
            websocket_server.close()
            break
        online.insert_audio_chunk(bytes_to_array(message))
        o, _ = online.process_iter()
        logger.info(o)
        websocket_server.send(whisper_to_json(o))

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
        
        new = [(a+offset,b+offset,t) for a,b,t in new]
        self.new = [(a,b,t) for a,b,t in new if a > self.last_commited_time-0.1]

        if len(self.new) >= 1:
            a,b,t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1,min(min(cn,nn),5)+1):  # 5 is the maximum 
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1,i+1)][::-1])
                        tail = " ".join(self.new[j-1][2] for j in range(1,i+1))
                        if c == tail:
                            logger.debug(f"removing last {i} words:")
                            for j in range(i):
                                logger.debug(f"\t{self.new.pop(0)}")
                            break

    def flush(self):
        # returns commited chunk = the longest common prefix of 2 last inserts. 

        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt.lower().translate(str.maketrans('', '', string.punctuation)) == self.buffer[0][2].lower().translate(str.maketrans('', '', string.punctuation)):                
                commit.append((na,nb,nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                # print(f"SStop committing at '{nt}' and '{self.buffer[0][2]}'")
                break
        self.buffer = self.new
        new_non_commit = [i for i in self.buffer if i[1] > self.last_buffered_time-0.1]
        self.last_buffered_time = self.buffer[-1][1] if self.buffer else -1
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit, new_non_commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer

class OnlineASRProcessor:

    SAMPLING_RATE = 16000

    def __init__(self, asr, buffer_trimming=15, use_vad="auditok", logfile=sys.stderr):
        """asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language. Must have a method *split* that behaves like the one of MosesTokenizer. It can be None, if "segment" buffer trimming option is used, then tokenizer is not used at all.
        ("segment", 15)
        buffer_trimming: a pair of (option, seconds), where option is either "sentence" or "segment", and seconds is a number. Buffer is trimmed if it is longer than "seconds" threshold. Default is the most recommended option.
        logfile: where to store the log. 
        """
        self.asr = asr
        self.logfile = logfile

        self.init()

        self.buffer_trimming_sec = buffer_trimming
        self.use_vad = use_vad

    def init(self):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([],dtype=np.float32)
        self.buffer_time_offset = 0

        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.commited = []
        self.last_chunked_at = 0

        self.silence_iters = 0

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of commited text that is inside of the scrolled away part of audio buffer. 
        "context" is the commited text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.
        """
        k = max(0,len(self.commited)-1)
        while k > 0 and self.commited[k-1][1] > self.last_chunked_at:
            k -= 1

        p = self.commited[:k]
        p = [t for _,_,t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x)+1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(t for _,_,t in non_prompt)

    def process_iter(self):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, ""). 
        The non-emty text is confirmed (committed) partial transcript.
        """
        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT:{prompt}")
        logger.debug(f"CONTEXT:{non_prompt}")
        logger.debug(f"Transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds starting at {self.buffer_time_offset:2.2f}s")
        # print(f"Transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds starting at {self.buffer_time_offset:2.2f}s")
        # use VAD to filter out the silence        
        if self.use_vad:
            np_buffer = np.array(self.audio_buffer)
            audio_speech, segments, convertion_function = remove_non_speech(np_buffer, method=self.use_vad, sample_rate=self.SAMPLING_RATE, dilatation=0.5)
            res = self.asr.transcribe(audio_speech, init_prompt=prompt)
        else:
            res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)
        # transform to [(beg,end,"word1"), ...]
        tsw = self.asr.ts_words(res, convertion_function if self.use_vad else None)
        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o, buffer = self.transcript_buffer.flush()
        self.commited.extend(o)
        if buffer and (self.buffer_time_offset+len(self.audio_buffer)/self.SAMPLING_RATE)-buffer[-1][1]<0.05:
            # remove the last word if it is too close to the end of the buffer
            buffer.pop(-1)
        logger.debug(f">>>>COMPLETE NOW:{self.to_flush(o)}")
        logger.debug(f"INCOMPLETE:{self.to_flush(self.transcript_buffer.complete())}")
        
        if len(self.audio_buffer)/self.SAMPLING_RATE > self.buffer_trimming_sec:
            self.chunk_completed_segment(res, chunk_silence=self.use_vad, speech_segments=segments if self.use_vad else False)

        logger.debug(f"len of buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}")
        return self.to_flush(o), self.to_flush(buffer)

    def chunk_completed_sentence(self):
        if self.commited == []: return
        logger.info(self.commited)
        sents = self.words_to_sentences(self.commited)
        for s in sents:
            logger.debug("\t\tSENT:",s)
        if len(sents) < 2:
            return
        while len(sents) > 2:
            sents.pop(0)
        # we will continue with audio processing at this timestamp
        chunk_at = sents[-2][1]

        logger.debug(f"--- sentence chunked at {chunk_at:2.2f}")
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res, chunk_silence=False, speech_segments=None):
        if self.commited == [] and not chunk_silence: 
            return

        ends = self.asr.segments_end_ts(res)
        t = self.commited[-1][1]
        if len(ends) > 1:
            e = ends[-2]+self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2]+self.buffer_time_offset
            if e <= t:
                logger.debug(f"--- segment chunked at {e:2.2f}")
                # print(f"--- segment chunked at {e:2.2f}")
                self.chunk_at(e)
            else:
                logger.debug(f"--- last segment not within commited area")
        elif chunk_silence:
            lenght = len(self.audio_buffer)/self.SAMPLING_RATE
            e = self.buffer_time_offset + lenght - 2
            if speech_segments:
                end_silence = lenght - speech_segments[-1][1]
                if end_silence > 2:
                    logger.debug(f"--- Silence segment chunked at {e:2.2f}")
                    self.chunk_at(e)
            elif speech_segments is not None:
                logger.debug(f"--- Silence segment chunked at {e:2.2f}")
                self.chunk_at(e)  
        else:
            logger.debug(f"--- not enough segments to chunk")


    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time"
        """
        # print(f"chunking at {time:2.2f}")
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds*self.SAMPLING_RATE):]
        self.buffer_time_offset = time
        self.last_chunked_at = time

    def words_to_sentences(self, words):
        """Uses self.tokenizer for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """
        
        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b,e,w = cwords.pop(0)
                w = w.strip()
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg,end,fsent))
                    break
                sent = sent[len(w):].strip()
        return out

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logger.debug(f"last, noncommited:{f}")
        return f


    def to_flush(self, sents, sep=None, offset=0, ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b,e,t)
    
    
class ASRBase:

    sep = " "   # join transcribe words with this character (" " for whisper_timestamped,
                # "" for faster-whisper because it emits the spaces when needed)

    def __init__(self, lan, model=None, logfile=sys.stderr, condition_on_previous_text=None):
        self.logfile = logfile

        self.transcribe_kargs = {}
        self.original_language = lan 
        self.model = model

    def transcribe(self, audio, init_prompt=""):
        raise NotImplemented("must be implemented in the child class")

    def use_vad(self, vad_name=None):
        raise NotImplemented("must be implemented in the child class")
    
    
class FasterWhisperASR(ASRBase):
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version.
    """

    sep = ""

    def __init__(self, lan, model=None, logfile=sys.stderr, condition_on_previous_text=None):
        super().__init__(lan, model=model, logfile=logfile)
        self.transcribe_kargs['beam_size'] = 1
        self.transcribe_kargs['best_of'] = 1
        self.transcribe_kargs['temperature'] = 0
        self.transcribe_kargs['condition_on_previous_text'] = False if condition_on_previous_text is None else condition_on_previous_text

    def transcribe(self, audio, init_prompt=""):
        segments, info = self.model.transcribe(audio, language=self.original_language, initial_prompt=init_prompt, word_timestamps=True, **self.transcribe_kargs)
        return list(segments)

    def ts_words(self, segments, timestamps_convert_function=None):
        o = []
        for segment in segments:
            for word in segment.words:
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                if timestamps_convert_function is not None:
                    start, end = timestamps_convert_function(word.start, word.end)
                    t = (start, end, w)
                else:
                    t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]

class WhisperTimestampedASR(ASRBase):
    """Uses whisper_timestamped library as the backend. Initially, we tested the code on this backend. It worked, but slower than faster-whisper.
    On the other hand, the installation for GPU could be easier.
    """

    sep = " "

    def __init__(self, lan, model=None, logfile=sys.stderr, condition_on_previous_text=None):
        super().__init__(lan, model=model, logfile=logfile)
        self.transcribe_kargs["verbose"] = None
        self.transcribe_kargs["beam_size"] = None
        self.transcribe_kargs["best_of"] = None
        self.transcribe_kargs["temperature"] = 0
        self.transcribe_kargs['condition_on_previous_text'] = False if condition_on_previous_text is None else condition_on_previous_text
        from whisper_timestamped import transcribe_timestamped
        self.transcribe_timestamped = transcribe_timestamped


    def transcribe(self, audio, init_prompt=""):
        result = self.transcribe_timestamped(self.model,
                audio, language=self.original_language,
                initial_prompt=init_prompt, **self.transcribe_kargs)
        return result
 
    def ts_words(self,r, timestamps_convert_function=None):
        # return: transcribe result object to [(beg,end,"word1"), ...]
        o = []
        for s in r["segments"]:
            for w in s["words"]:
                if timestamps_convert_function is not None:
                    # print(f"start: {word.start}->{timestamps_convert_function(word.start)}, end: {word.end}->{timestamps_convert_function(word.end)}")
                    start, end = timestamps_convert_function(w["start"], w['end'])
                    t = (start, end, w["text"])
                else:
                    t = (w["start"],w["end"],w["text"])
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s["end"] for s in res["segments"]]

