# 2.2.1
- Fix minor bugs
- put SWAGGER_PATH parameter as optional
- Generate the word_boundary file if it not exists

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