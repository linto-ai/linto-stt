#!/bin/bash

dockerfile=$1
shift
env_file=$1
shift

tag=test_`basename $dockerfile`

docker build . -f $dockerfile -t linto-stt-whisper:$tag
touch build_finished

CMD="docker run --rm -p 8080:80 --name test_container --env-file $env_file --gpus all $* linto-stt-whisper:$tag"
echo $CMD
eval $CMD
