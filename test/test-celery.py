from celery import Celery

app = Celery('linto_stt', broker='redis://localhost:6379/0', backend='redis://localhost:6379/1')

result = app.send_task('transcribe_task', args=['baudelaire 2.wav', True, 'fr'])

# Attendre le résultat
print(result.get(timeout=120))
