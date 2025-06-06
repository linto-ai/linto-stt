import asyncio
import websockets
import json
import os
import time
import shutil
import subprocess
import logging


logging.basicConfig(level=logging.INFO, filename="linstt_streaming.log", filemode="w")
logger = logging.getLogger(__name__)

async def send_data(websocket, stream, logger, stream_config):
    """Asynchronously load and send data to the WebSocket."""
    duration = 0
    try:
        while True:
            data = stream.read(int(stream_config['stream_duration'] * 2 * 16000))
            duration += stream_config['stream_duration']
            if stream_config['audio_file'] and not data:
                logger.debug("Audio file finished")
                break
            if stream_config['vad']:
                import auditok
                audio_events = auditok.split(
                    data,
                    energy_threshold=50,
                    sampling_rate=16000,
                    sample_width=2,
                    channels=1
                )
                audio_events = list(audio_events)
                if len(audio_events) == 0:
                    logger.debug(f"Full silence for chunk: {duration - stream_config['stream_duration']:.1f}s --> {duration:.1f}s")
                    if stream_config['stream_wait']>0:
                        await asyncio.sleep(stream_config['stream_wait'])
                    continue
                data = b''.join([event.data for event in audio_events])
            await websocket.send(data)
            logger.debug(f"Sent audio chunk: {duration - stream_config['stream_duration']:.1f}s --> {duration:.1f}s")
            if stream_config['stream_wait']>0:
                await asyncio.sleep(stream_config['stream_wait'])

    except asyncio.CancelledError:  # handle server errors and ctrl+c...
        logger.debug("Data sending task cancelled.")
    except Exception as e:          # handle data loading errors
        logger.error(f"Error in data sending: {e}")
    logger.debug(f"Waiting before sending EOF")
    await asyncio.sleep(5)
    logger.debug(f"Sending EOF")
    await websocket.send('{"eof" : 1}')
  
def linstt_streaming(*kargs, **kwargs):
    text = asyncio.run(_linstt_streaming(*kargs, **kwargs))
    return text

async def play_sound(sound):
    import winsound
    winsound.PlaySound(sound, winsound.SND_ALIAS|winsound.SND_ASYNC)

async def _linstt_streaming(
    audio_file,
    ws_api = "ws://localhost:8080/streaming",
    verbose = False,
    language = None,
    apply_vad = False,
    stream_duration = 0.5,
    stream_wait = 0.5,
    play_audio_file=False
):
    if verbose:
        logger.setLevel(logging.DEBUG)
    stream_config = {"language": language, "sample_rate": 16000, "vad": apply_vad, "stream_duration": stream_duration, "stream_wait": stream_wait}
    if audio_file is None:
        import pyaudio
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt8, channels=1, rate=stream_config['sample_rate'], input=True, frames_per_buffer=2048)
        logger.debug("Start recording")
        stream_config["audio_file"] = None
    else:
        subprocess.run(["ffmpeg", "-y", "-i", audio_file, "-acodec", "pcm_s16le", "-ar", str(stream_config['sample_rate']), "-ac", "1", "tmp.wav"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stream = open("tmp.wav", "rb")
        logger.debug(f"Start streaming file {audio_file}")
        stream_config["audio_file"] = audio_file
    play_audio_task = None
    text = ""
    partial = None
    duration = 0
    async with websockets.connect(ws_api, ping_interval=None, ping_timeout=None) as websocket:
        if language is not None:
            config = {"config" : {"sample_rate": stream_config['sample_rate'], "language": stream_config['language']}}
        else: 
            config = {"config" : {"sample_rate": stream_config['sample_rate']}}
        await websocket.send(json.dumps(config))
        send_task = asyncio.create_task(send_data(websocket, stream, logger, stream_config))
        if audio_file and play_audio_file:
            play_audio_task = asyncio.create_task(play_sound(audio_file))
        last_text_partial = None
        try:
            while True:
                res = await websocket.recv()
                message = json.loads(res)
                if message is None:
                    logger.debug("\n Received None")
                    continue
                if "text" in message.keys():
                    line = message["text"]
                    if line and verbose:
                        print_streaming(line, partial=False, last_partial=last_text_partial)
                    logger.debug(f'Final (after {duration:.1f}s): "{line}"')
                    last_text_partial = None
                    if line:
                        if text:
                            text += "\n"
                        text += line
                elif "partial" in message.keys():
                    partial = message["partial"]
                    if partial and verbose:
                        print_streaming(partial, partial=True, last_partial=last_text_partial)
                        last_text_partial = partial
                    logger.debug(f'Partial (after {duration:.1f}s): "{partial}"')
                elif verbose:
                    logger.debug(f"??? {message}")
        except asyncio.CancelledError:  # handle ctrl+c...
            logger.debug("Message processing thread stopped as websocket was closed.")
        except websockets.exceptions.ConnectionClosedOK:
            logger.debug("Websocket closed")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.error(f"Websocket closed with error: {e}")
        except Exception as e:
            raise Exception(f"Error in message processing {message} {type(message)}: {e}")
        finally:
            send_task.cancel()
            await send_task
            if play_audio_task:
                play_audio_task.cancel()
                await play_audio_task
            await websocket.close()
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
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose mode")
    parser.add_argument("--audio_file", default=None, help="A path to an audio file to transcribe (if not provided, use mic)")
    parser.add_argument("--language", default=None, help="Language model to use")
    parser.add_argument("--apply_vad", default=False, action="store_true", help="Apply VAD to the audio stream before sending it to the server")
    parser.add_argument("--stream_duration", default=0.5, type=float, help="Duration of the audio stream sent to the server")
    parser.add_argument("--stream_wait", default=0.5, type=float, help="Duration to wait between two audio stream chunks")
    parser.add_argument("--play_audio_file", default=False, action="store_true", help="")
    args = parser.parse_args()

    res = linstt_streaming(args.audio_file, args.server, verbose=args.verbose, language=args.language, apply_vad=args.apply_vad, stream_duration=args.stream_duration, stream_wait=args.stream_wait, play_audio_file=args.play_audio_file)