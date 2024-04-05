#!/bin/bash

tests_run=0
passed=0
failed=0
global_start=$(date +%s)
test_log=test/test.log

function echo_success() {
    # Print a green tick (with colour only on the terminal, not the log file)
    printf '\033[0;32m'
    printf '\xE2\x9C\x94 ' | tee -a $test_log
    printf '\033[0m' # No Color
    echo $* | tee -a $test_log
}

function echo_failure() {
    # Print a red cross (with colour only on the terminal, not the log file)
    printf '\033[0;31m'
    printf '\xE2\x9C\x96 ' | tee -a $test_log
    printf '\033[0m' # No Color
    echo $* | tee -a $test_log
}

function echo_note() {
    printf '🕓 ' | tee -a $test_log
    echo $* | tee -a $test_log
}

function test_failed() {
    local end=$(date +%s)
    failed=$((failed + 1))
    echo "-----------------------" | tee -a $test_log
    echo_failure "Test failed after "$((end-start))" seconds ($passed/$tests_run tests succeeded in "$((end-global_start))" seconds)"
    test_teardown
    echo 'See $test_log for more details.'
    # exit 1
}

function test_succeeded(){
    local end=$(date +%s)
    passed=$((passed + 1))
    echo "-----------------------" | tee -a $test_log
    echo_success "Test passed in "$((end-start))" seconds ($passed/$tests_run tests succeeded in "$((end-global_start))" seconds)"
    test_teardown
}

function test_teardown(){
    rm -f build_finished
    local end=$(date +%s)
    docker stop test_redis > /dev/null 2> /dev/null
    docker stop test_container > /dev/null 2> /dev/null
    pkill -P $pids
    echo | tee -a $test_log
}

function ending() {
    local end=$(date +%s)
    echo_note 'Time to run tests: '$((end-global_start))' seconds.'
    if [ $passed -gt 0 ];then
        echo_success $passed/$tests_run tests passed.
    fi
    if [ $failed -gt 0 ];then
        echo_failure $failed/$tests_run tests failed.
    fi
    if [ $passed -eq $tests_run ]; then
        echo_success 'TEST PASSED.'
        exit 0
    else
        echo_failure 'TEST FAILED.'
        exit 1
    fi
}

function ctrl_c() {
    echo ''
    echo_failure "Interruption signal received, stopping the server... (do not press Ctrl + C again)"
    test_teardown
    ending
}


# Attend la création du fichier avec un timeout de 600 secondes
wait_for_file_creation_with_timeout() {
    local file="$1"
    local pid="$2"
    local timeout=600  # 10 minutes en secondes
    local start_time=$(date +%s)
    
    while [ ! -f "$file" ]; do
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        if [ $elapsed_time -ge $timeout ]; then
            echo "Fatal Error: Timeout. The docker image took too long to be built." | tee -a $test_log
            exit 1
        fi
        # Vérifie si le processus est toujours en cours d'exécution
        if ! ps -p $pid > /dev/null; then
            echo "Fatal Error: Docker build failed." | tee -a $test_log
            exit 1
        fi
        sleep 1
    done
    end_time=$(date +%s)
    echo_note "Docker image has been successfully built in "$((end_time - start_time))" sec."
    rm $file
    if [[ "$(ps -p $pid > /dev/null)" ]]; then
        echo_failure "Fatal Error: Docker container start failed immediately."
        exit 1
    fi
    return 0
}

check_http_server_availability() {
    local server="$1"
    local total_wait_time=600  # 10 minutes en secondes
    local retry_interval=1    # Interval entre les tentatives (en secondes)
    local elapsed_time=0

    while [ $elapsed_time -lt $total_wait_time ]; do
        # Test de la disponibilité du serveur HTTP
        curl -s --head --request GET "$server" | grep "200 OK"
        if [ $? -eq 0 ]; then
            echo_note "$server is available after $elapsed_time sec."
            return 0
        fi

        if [[ `docker ps -a -q -f name=test_container | wc -l` -eq 0 ]];then
            echo_failure "Fatal error: the server container has stopped for unexpected reason."
            exit 1
        fi

        # Attendre avant la prochaine tentative
        sleep $retry_interval
        elapsed_time=$((elapsed_time + retry_interval))
    done

    echo_failure "$server is not available after $total_wait_time seconds, server launching must have failed."
    exit 1
}

build_and_run_container()
{
    # Input parameters
    local serving="$1"
    local docker_image="$2"
    local use_local_cache="$3"
    env_variables=$(echo $@ | cut -d' ' -f4-)

    tests_run=$((tests_run + 1))
    echo "=== Starting test $tests_run ===" | tee -a $test_log
    echo "* Docker image: $docker_image" | tee -a $test_log
    echo "* Audio file..: $test_file" | tee -a $test_log
    build_args=""
    for env in $env_variables; do
        build_args="$build_args --env $env"
    done
    build_args="$build_args --env SERVICE_MODE=$serving"
    if [ $use_local_cache -gt 0 ];then
        build_args="$build_args -v $HOME/.cache:/root/.cache"
    fi
    echo "* Options.....:$build_args" | tee -a $test_log
    echo "-----------------------" | tee -a $test_log

    pids=""
    if [ "$serving" == "task" ]; then
        build_args="$build_args -v `pwd`:/opt/audio"
        # Launch Redis server
        test/launch_redis.sh &
        if [ $? -ne 0 ]; then
            echo_failure "Redis server failed to start."
            test_failed
            exit 1
        fi
        pids=$!
    fi

    start=$(date +%s)
    # Exécute la fonction de construction dans un sous-processus
    rm -f build_finished
    test/build_and_run_container.sh $docker_image test/.env $build_args &
    local pid=$!
    pids="$pids $pid"
    
    # Attend la création du fichier avec un timeout de 600 secondes
    wait_for_file_creation_with_timeout build_finished $pid
    if [ $? -ne 0 ]; then
        test_failed
        exit 1
    fi
}

run_test()
{
    local serving="$1"
    shift
    if [ "$serving" == "http" ]; then
        run_test_http $*
    elif [ "$serving" == "task" ]; then
        run_test_task $*
    else
        echo_failure "Error: Unknown serving mode '$serving'."
        exit 1
    fi
}

run_test()
{
    # Input parameters
    local serving="$1"
    shift
    local test_file="$1"
    shift
    if [ "$test_file" == "test/GOLE7.wav" ] ; then
        regex=".*Je crois que j'avais des profs.*"
    elif [ "$test_file" == "test/bonjour.wav" ]; then
        regex=".*Bonjour.*"
    fi

    build_and_run_container $serving $*
    
    if [ "$serving" == "http" ]; then
        check_http_server_availability "http://localhost:8080/healthcheck"
        if [ $? -ne 0 ]; then
            test_failed
            return 1
        fi
        
        # Test HTTP
        CMD='curl -X POST "http://localhost:8080/transcribe" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@$test_file;type=audio/wav"'
        echo "$ "$CMD
        local start_time=$(date +%s)
        local res=$(eval $CMD 2>/dev/null)
        local end_time=$(date +%s)
        if [ -z "$res" ]; then
            echo_failure "The server didn't transcribed, retrying in 2 sec..."
            sleep 2
            res=$(eval $CMD 2>/dev/null)
            end_time=$(date +%s)
        fi
        echo_note "HTTP route 'transcribe' has transcribed $test_file in $((end_time - start_time)) sec."
        if [[ ! $res =~ $regex ]]; then
            echo_note "Error: The string '$res' is not matching the regex ($regex), the server didn't transcribed correctly. Output text : $res"
            test_failed
            return 1
        fi

        # Test streaming
        CMD="python3 test/test_streaming.py --audio_file $test_file"
        echo "$ "$CMD
        start_time=$(date +%s)
        res=$(eval $CMD 2> >(tee -a $test_log >&2))
        end_time=$(date +%s)
        echo_note "HTTP websocket has transcribed $test_file in $((end_time - start_time)) sec."
        if [[ ! $res =~ $regex ]]; then
            echo_failure "The string '$res' is not matching the regex ($regex), the server didn't transcribed correctly. Output text : $res"
            test_failed
            return 1
        fi

    elif [ "$serving" == "task" ]; then

        CMD="python3 test/test_celery.py $test_file"
        echo "$ "$CMD
        local start_time=$(date +%s)
        local res=$(eval $CMD 2> >(tee -a $test_log >&2))
        local end_time=$(date +%s)
        if [ $? -ne 0 ]; then
            test_failed
            return 1
        fi
        echo_note "Celery task has transcribed $test_file in $((end_time - start_time)) sec."
        if [[ ! $res =~ $regex ]]; then
            echo_failure "The string '$res' is not matching the regex ($regex), the server didn't transcribed correctly. Output text : $res"
            test_failed
            return 1
        fi

    fi

    test_succeeded
    return 0
}

trap ctrl_c INT
echo Starting tests at $(date '+%d/%m/%Y %H:%M:%S') | tee $test_log
echo '' | tee -a $test_log

# Prepare env file for tests
cat whisper/.envdefault | grep -v "DEVICE=" | grep -v "VAD=" | grep -v "MODEL=" | grep -v "SERVICE_MODE=" > test/.env

#######################
# List of what to test

dockerfiles+=" whisper/Dockerfile.ctranslate2"
dockerfiles+=" whisper/Dockerfile.ctranslate2.cpu"
dockerfiles+=" whisper/Dockerfile.torch"
dockerfiles+=" whisper/Dockerfile.torch.cpu"

use_local_caches+=" 1"
# use_local_caches+=" 0"

servings+=" task"
servings+=" http"

vads+=" NONE"
vads+=" false"
vads+=" auditok"
vads+=" silero"

devices+=" NONE"
devices+=" cpu"
devices+=" cuda"

models+=" tiny"

#######################
# Run tests

for use_local_cache in $use_local_caches;do
for dockerfile in $dockerfiles; do
for device in $devices; do
for vad in $vads; do
for model in $models; do
for serving in $servings; do

    # Tests to skip
    if [[ "$device" != "cpu" ]] && [[ `echo $dockerfile | grep cpu | wc -l` -gt 0 ]]; then continue; fi

    # Set env variables
    envs=""
    if [ "$vad" != "NONE" ]; then envs="$envs VAD=$vad"; fi
    if [ "$device" != "NONE" ]; then envs="$envs DEVICE=$device"; fi
    envs="$envs MODEL=$model"
    
    # Run test
    run_test $serving test/bonjour.wav $dockerfile $use_local_cache $envs

done
done
done
done
done
done

ending