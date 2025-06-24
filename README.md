# LinTO-STT

LinTO-STT is an API for Automatic Speech Recognition (ASR).

LinTO-STT can either be used as a standalone transcription service or deployed within a micro-services infrastructure using a message broker connector.

It can be used to do offline or real-time transcriptions.

The following families of STT models are currently supported (please refer to respective documentation for more details):
* [Kaldi models](kaldi/README.md) 
* [Whisper models](whisper/README.md)
* [NeMo models](nemo/README.md)

Some functional tests can be found in [the `test/` subfolder](test/README.md).

LinTO-STT can either be used as a standalone transcription service or deployed within a micro-services infrastructure using a message broker connector.

## License
This project is developped under the AGPLv3 License (see LICENSE).
