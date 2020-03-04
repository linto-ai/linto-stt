curl -X POST "http://localhost:8888/transcribe" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@bonjour.wav;type=audio/wav"
