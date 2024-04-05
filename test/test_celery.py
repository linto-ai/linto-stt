import sys
from celery import Celery
celery = Celery(broker='redis://localhost:6379/0', backend='redis://localhost:6379/1')
r = celery.send_task(
    'transcribe_task', 
    (
        sys.argv[1],
        True,
    ),
    queue='stt')
print(r.get())
