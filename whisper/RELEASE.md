# 1.0.0
- Support of Whisper (including large-v3 model)
- Add integration of Whisper models from transformers
- Add support of prompt from Whisper models (env variable PROMPT)
- Fix possible failure when a Whisper segment starts with a punctuation
- Tune punctuation heuristics

# 0.0.0
- Added optional streaming route to the http serving mode
- Added serving mode: websocket
- Added Dynamic model conversion allowing to use either Vosk Models or Linagora AM/LM models
- Added celery connector for microservice integration.
- Added launch option to specify serving mode between task and http.
- Removed Async requests/Job management.
- New feature: Compute a confidence score per transcription
- put SWAGGER_PATH parameter as optional
