audio=${1:-bonjour.wav}
if audio endswith .mp3; then
    ffmpeg -y -i $audio -acodec pcm_s16le -ar 16000 -ac 1 tmp.wav
    curl -X POST "http://localhost:8080/transcribe" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@tmp.wav;type=audio/wav"
else
    curl -X POST "http://localhost:8080/transcribe" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@$audio;type=audio/wav"
fi