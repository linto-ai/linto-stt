# Linto-Platform-Stt-Standalone-Worker

This service is mandatory in a LinTO platform stack as the main worker for speech to text toolkit.

Generally, Automatic Speech Recognition (ASR) is the task of recognition and translation of spoken language into text. Our ASR system takes advantages from the recent advances in machine learning technologies and in particular deep learning ones (TDNN, LSTM, attentation-based architecture). The core of our system consists of two main components: an acoustic model and a decoding graph. A high-performance ASR system relies on an accurate acoustic model as well as a perfect decoding graph.

**NB**: The service works as follows: 
* If the audio's duration is less that 30 minutes, the service will return the transcription after decoding.
* Otherwise, the server will return a **jobid** that could be used to get the transcription after decoding using the API **`/transcription/{jobid}`**.

## Usage
See documentation : [doc.linto.ai](https://doc.linto.ai)

# Deploy

With our proposed stack [linto-platform-stack](https://github.com/linto-ai/linto-platform-stack)

# Hardware requirements
In order to install and run this service, you need to have at least:

* 5Go available on your hard drive for the installation, and

* 500Mo/3Go/7Go of RAM memory available for models loading and decoding. The size depends mainly on the choosed decoding model (small, medium or big).

While there is no specific minimal requirement on the CPU, speech recognition is a computationally task.

**`—The better your hardware performance, the lower your decoding time—`**

# Develop

## Installation

### Packaged in Docker
To start the STT worker on your local machine or your cloud, you need first to download the source code and set the environment file, as follows:

```bash
git clone https://github.com/linto-ai/linto-platform-stt-standalone-worker
cd linto-platform-stt-standalone-worker
git submodule update --init
mv .envdefault .env
```

Then, to build the docker image, execute:

```bash
docker build -t lintoai/linto-platform-stt-standalone-worker:latest .
```

Or by docker-compose, by using:
```bash
docker-compose build
```


Or, download the pre-built image from docker-hub:

```bash
docker pull lintoai/linto-platform-stt-standalone-worker:latest
```

NB: You must install docker and docker-compose on your machine.

## Configuration
The STT worker that will be set-up here require KALDI models, the acoustic model and the decoding graph. Indeed, these models are not included in the repository; you must download them in order to run LinSTT. You can use our pre-trained models from here: [Downloads](https://doc.linto.ai/#/services/linstt_download).

1- Download the French acoustic model and the small decoding graph (linstt.v1). You can download the latest version for optimal performance and you should make sure that you have the hardware requirement in terms of RAM.

```bash
wget https://dl.linto.ai/downloads/model-distribution/acoustic-models/fr-FR/linSTT_AM_fr-FR_v1.0.0.zip
wget https://dl.linto.ai/downloads/model-distribution/decoding-graphs/LVCSR/fr-FR/decoding_graph_fr-FR_Small_v1.1.0.zip
```

2- Uncompress both files

```bash
unzip linSTT_AM_fr-FR_v1.0.0.zip -d AM_fr-FR
unzip decoding_graph_fr-FR_Small_v1.1.0.zip -d DG_fr-FR_Small
```

3- Move the uncompressed files into the shared storage directory

```bash
mkdir ~/linstt_model_storage
mv AM_fr-FR ~/linstt_model_storage
mv DG_fr-FR ~/linstt_model_storage
```

4- Configure the environment file `.env` included in this repository

```bash
    AM_PATH=~/linstt_model_storage/AM_fr-FR
    LM_PATH=~/linstt_model_storage/DG_fr-FR
```

NB: if you want to use the visual user interface of the service, you need also to configure the swagger file `document/swagger.yml` included in this repository. Specifically, in the section `host`, specify the adress of the machine in which the service is deployed.

## Execute
In order to run the service, you have only to execute:

```bash
cd linto-platform-stt-standalone-worker
docker run -p 8888:80 -v /full/path/to/linstt_model_storage/AM_fr-FR:/opt/models/AM -v /full/path/to/linstt_model_storage/DG_fr-FR:/opt/models/LM -v /full/path/to/linto-platform-stt-standalone-worker/document/swagger.yml:/opt/swagger.yml -e SWAGGER_PATH="/opt/swagger.yml" lintoai/linto-platform-stt-standalone-worker:latest
```

or simply by executing:
```bash
cd linto-platform-stt-standalone-worker
docker-compose up
```

Our service requires an audio file in `Waveform format`. It should has the following parameters:

    - sample rate: 16000 Hz
    - number of bits per sample: 16 bits
    - number of channels: 1 channel
    - microphone: any type
    - duration: <30 minutes

### API
<!-- tabs:start -->

#### /transcribe

Convert a speech to text

### Functionality
>  `post`  <br>
> Make a POST request
>>  <b  style="color:green;">Arguments</b> :
>>  -  **{File} file** Audio File - Waveform Audio File Format is required

>
>>  <b  style="color:green;">Header</b> :
>>  -  **{String} Accept**: response content type (text/plain|application/json)
>
>  **{text|Json}** : Return the full transcription or a json object with metadata


#### /transcription/{jobid}

Get the transcription using the jobid

### Functionality
>  `get`  <br>
> Make a GET request
>>  <b  style="color:green;">Arguments</b> :
>>  -  **{String} jobid** jobid - An identifier used to find the corresponding transcription
>
>  **{text|Json}** : Return the transcription


#### /jobids

List of the transcription jobids

### Functionality
>  `get`  <br>
> Make a GET request
>>  <b  style="color:green;">Arguments</b> :
>>  - no arguments
>
>  **{Json}** : Return a json object with jobids

<!-- tabs:end -->


### Run Example Applications
To run an automated test, go to the test folder:

```bash
cd linto-platform-stt-standalone-worker/test
```

And run the test script:

```bash
./test_deployment.sh
```

To run personal test, you can use swagger interface: `localhost:8888/api-doc/`


### Extrat metadata
If you would like to have a transcription with speaker information and punctuation marks, it's possible thanks to our open-source services:

* Speaker diarization worker: https://github.com/linto-ai/linto-platform-speaker-diarization-worker
* Text punctuation worker: https://github.com/linto-ai/linto-platform-text-punctuation-worker

To do that, you need first to start either the speaker or punctuation service or you can start both if it's necessary. **Please read the documentation to know how to install, configure, and start these services.**

Once the services are on, you need to configure the STT worker as follows:

1- Edit the environment file `.env` as follows:

* if you started the punctuation worker, the following variables should be used

```bash
    PUCTUATION_HOST=text-punctuation-worker-host-name
    PUCTUATION_PORT=worker-port-example-80
    PUCTUATION_ROUTE=/api/route/path/
```
* if you started the speaker diarization worker, the following variables should be used

```bash
    SPEAKER_DIARIZATION_HOST=speaker-diarization-worker-host-name
    SPEAKER_DIARIZATION_PORT=worker-port-example-80
```

2- Start the service using the same command described in section **Execute**