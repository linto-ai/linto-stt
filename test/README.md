# LinTO-STT-Tests

## Use tests

### HTTP - transcribe

You can test your http server by using:

```bash
test_deployment.sh
```

> ⚠️ Be sure to check that you use the right port (default port for testing: 8080).         

### HTTP - streaming

You can test your http streaming route by using:
```bash
test_streaming.py
```
Be sure to have a working microphone.
> ⚠️ Be sure to check that you use the right port (default port for testing: 8080). 

If you want to test the streaming on a file:
```bash
test_streaming.py --audio_file bonjour.wav
```

### Task

You can test your deployment of the task service mode by using:

```bash
test_celery.py AUDIO.wav
```

with AUDIO.wav the file you want to test on, for example, you can use bonjour.wav. 

> ⚠️ Be sure to check that you use the same port in your .env and in test_celery.py (default port for testing: 6379)


## Unit tests

You will need to install:
```bash
pip3 install ddt
```

To test the Kaldi models, you will need to download the models (see [Kaldi models](../kaldi/README.md)) and then fill the test_config.ini AM_PATH and LM_PATH fields. 
> ⚠️ If you don't specify the models, the tests about Kaldi will fail.

To launch the test you can do :
```bash
python test/test.py
```

> ⚠️ Be sure to launch it from the root folder of the repository.

If you want the test to stop at the first fail use the -f flag:
```bash
python test/test.py -f
```
If you want to run a subset of test you can use -k with a part of a test name. for example only kaldi tests:
```bash
python test/test.py -k kaldi
```
or test with VAD=auditok, DEVICE=cuda:
```bash
python test/test.py -k VAD_auditok_DEVICE_cuda
```