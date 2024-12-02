import sys
from celery import Celery

def transcribe_task(file_path, language=None):
    celery = Celery(broker='redis://localhost:6379/0', backend='redis://localhost:6379/1')
    r = celery.send_task(
        'transcribe_task', 
        (
            file_path,
            True,
            language
        ),
        queue='stt')
    return r.get()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Transcribe with LinSTT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--audio_file", default="bonjour.wav", help="A path to an audio file to transcribe (if not provided, use mic)")
    parser.add_argument("--language", default=None, help="Language model to use")
    args = parser.parse_args()
    print(transcribe_task(args.audio_file, args.language))