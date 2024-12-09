import asyncio
import websockets
import json
import os
import shutil
import subprocess

  
def linstt_streaming(*kargs, **kwargs):
    text = asyncio.run(_linstt_streaming(*kargs, **kwargs))
    return text

async def _linstt_streaming(
    audio_file,
    ws_api = "ws://localhost:8080/streaming",
    verbose = False,
    language = None
):
    
    if audio_file is None:
        import pyaudio
        # Init pyaudio
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=2048)
        if verbose > 1:
            print("Start recording")
    else:
        subprocess.run(["ffmpeg", "-y", "-i", audio_file, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "tmp.wav"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stream = open("tmp.wav", "rb")
    text = ""
    partial = None
    duration = 0
    async with websockets.connect(ws_api) as websocket:
        if language is not None:
            config = {"config" : {"sample_rate": 16000, "language": language}}
        else: 
            config = {"config" : {"sample_rate": 16000}}
        await websocket.send(json.dumps(config))
        last_text_partial = None
        stream_duration = 1
        while True:
            data = stream.read(stream_duration*2*16000)
            duration += stream_duration
            if audio_file and not data:
                if verbose > 1:
                    print("\nAudio file finished")
                break
            await websocket.send(data)
            res = await websocket.recv()
            message = json.loads(res)
            if message is None:
                if verbose > 1:
                    print("\n Received None")
                continue
            if "partial" in message.keys():
                partial = message["partial"]
                if partial and verbose:
                    print_streaming(partial, partial=True, last_partial=last_text_partial)
                    last_text_partial = partial
            elif "text" in message.keys():
                line = message["text"]
                if line and verbose:
                    print_streaming(line, partial=False, last_partial=last_text_partial) 
                last_text_partial = None
                if line:
                    if text:
                        text += "\n"
                    text += line
            elif verbose:
                print(f"??? {message}")
        if verbose > 1:
            print("Sending EOF")
        await websocket.send('{"eof" : 1}')
        res = await websocket.recv()
        message = json.loads(res)
        if isinstance(message, str):
            message = json.loads(message)
        if verbose > 1:
            print("Received EOF", message)
        if text:
            text += " "
        text += message["text"]
    if verbose:
        terminal_size = shutil.get_terminal_size()
        width = terminal_size.columns
        print()
        print(" FULL TRANSCRIPTION ".center(width, "-"))
        print(text)
    if audio_file is not None:
        os.remove("tmp.wav")
    return text
    
def print_streaming(text, partial=True, last_partial=None):
    if partial:
        text = text + "…"
    terminal_size = shutil.get_terminal_size()
    width = terminal_size.columns
    if last_partial is not None:
        number_of_lines = ((len(last_partial)+1)//width)+1
        for i in range(number_of_lines):
            print("\033[F\033[K", end="")
    print(text)
    
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Transcribe input streaming (from mic or a file) with LinSTT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--server', help='Transcription server',
        default="ws://localhost:8080/streaming",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--audio_file", default=None, help="A path to an audio file to transcribe (if not provided, use mic)")
    parser.add_argument("--language", default=None, help="Language model to use")
    args = parser.parse_args()

    res = linstt_streaming(args.audio_file, args.server, verbose=2 if args.verbose else 1, language=args.language)