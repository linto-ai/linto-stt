import sys
from celery import Celery

def transcribe_task(file_path):
    celery = Celery(broker='redis://localhost:6379/0', backend='redis://localhost:6379/1')
    r = celery.send_task(
        'transcribe_task', 
        (
            file_path,
            True,
        ),
        queue='stt')
    return r.get()

if __name__ == '__main__':
    print(transcribe_task(sys.argv[1]))