#!/bin/bash

dockerfile=$1
shift
env_file=$1
shift

tag=test_`basename $dockerfile`

CMD="docker build . -f $dockerfile -t linto-stt-test:$tag"
echo "$ "$CMD
eval $CMD > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi
touch build_finished

CMD="docker run --rm -p 8080:80 --name test_container --env-file $env_file --gpus all $* linto-stt-test:$tag"
# grep -v "^#" $env_file | grep "=" | grep -v SERVICE_NAME | grep -v BROKER | grep -v PORT
echo "$ "$CMD
eval $CMD > /dev/null 2>&1
