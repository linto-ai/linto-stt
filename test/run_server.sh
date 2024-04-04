#!/bin/bash

docker build . -f $1 -t linto-stt-whisper:latest
cp $2 whisper/.env
touch build_finished
docker run --rm -p 8080:80 --name test_container --env-file whisper/.env --gpus all -v /home/abert/.cache:/root/.cache linto-stt-whisper:latest 
