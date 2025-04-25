audio=${1:-bonjour.wav}
curl -X POST "http://localhost:8080/transcribe" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@$audio;type=audio/wav"
