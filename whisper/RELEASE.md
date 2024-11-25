# 1.0.5
- Upgrade faster-whisper and support (large v3) turbo models

# 1.0.4
- Add environment variables to control decoding strategy (USE_ACCURATE=0/1)
- Add environment variables to control streaming performance (STREAMING_MIN_CHUNK_SIZE, STREAMING_BUFFER_TRIMMING_SEC)
- Add environment variables to control VAD (VAD_DILATATION, VAD_MIN_SPEECH_DURATION, VAD_MIN_SILENCE_DURATION)
- Fix corner case where VAD detects no speech zone
- Fix corner case in streaming where "eof" was found in message
- Fix healthcheck on GPU

# 1.0.3
- Make Voice Activity Detection (VAD) configurable
- Change default VAD from silero (neural approach) to auditok (heuristical approach), because silero can have unpredictable behaviour on different corner cases
- Streaming support
- New NUM_THREADS env variable to control the number of threads
- Load the model when launching the service (not at the first request)

# 1.0.2
- ct2/faster_whisper: Upgrade faster_whisper and support recent distilled models
- ct2/faster_whisper: Fix possible gluing of different words together
- torch/whisper-timesptamped: Upgrade whisper-timestamped and delegate model loading

# 1.0.1
- ct2/faster_whisper: Information about used precision added in the logs
- torch/whisper-timesptamped: support of model.safetensors

# 1.0.0
- First build of linto-stt-whisper
- Based on 4.0.5 of linto-stt https://github.com/linto-ai/linto-stt/blob/a54b7b7ac2bc491a1795bb6dfb318a39c8b76d63/RELEASE.md
