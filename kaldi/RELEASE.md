#  1.1.0
- Fix EOF when streaming 
- Add possibility to add recase & punctuation in streaming
- Add an input option language to the request, that is just here for uniformity with other services (Whisper, that is multi-lingual and supports language detection).
  If passed, there is just a check that it is compatible with the LANGUAGE environment variable (if set)...

#  1.0.3
- Fix corner case in streaming where "eof" was found in message

#  1.0.2
- Fix task mode for kaldi by updating SERVICES_BROKER and BROKER_PASS in .envdefault

#  1.0.1
- Fix streaming mode (websocket) in linto-stt-kaldi

#  1.0.0
- First build of linto-stt-kaldi
- Based on 3.3.2 of linto-stt (https://github.com/linto-ai/linto-stt/blob/4361300a4463c90cec0bf3fa2975d7cc2ddf8d36/RELEASE.md)
