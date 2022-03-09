# 3.3.0
- Added optional streaming route to the http serving mode
- Added serving mode: websocket
- Added Dynamic model conversion allowing to use either Vosk Models or Linagora AM/LM models
- Changer Vosk dependency to alphacep/vosk
- Updated README.md

# 3.2.1
- Repository total rework. The goal being to have a simple transcription service embeddable within a micro-service infrastructure. 
- Changed repository name from linto-platform-stt-standalone-worker to linto-platform-stt.
- Added celery connector for microservice integration.
- Added launch option to specify serving mode between task and http.
- Removed diarization functionnality.
- Removed punctuation functionnality.
- Removed Async requests/Job management.
- Updated README to reflect those changes.

# 3.1.1
- Change Pykaldi with vosk-API (no python wrapper for decoding function, no extrat packages during installation, c++ implementation based on kaldi functions)
- New feature: Compute a confidence score per transcription
- Fix minor bugs

# 2.2.1
- Fix minor bugs
- put SWAGGER_PATH parameter as optional
- Generate the word_boundary file if it does not exist

# 2.2.0
- Speaker diarization feature: pyBK package
- Mulithreading feature: Speech decoding and Speaker diarization processes
- Optional parameter: real number of speaker in the audio

# 2.0.0
- Reimplement LinTO-Platform-stt-standalone-worker using Pykaldi package

# 1.1.2
- New features:
    - Word timestamp computing
    - Response type: plain/text: simple text output and application/json: the transcription and the words timestamp.
    - Swagger: integrate swagger in the service using a python package
    - Fix minor bugs

# 1.0.0
- First build of LinTO-Platform-stt-standalone-worker