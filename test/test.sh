#!/bin/bash

tests_run=0
passed=0
failed=0

function test_failed() {
    mkdir -p test/tests_failed
    mv $2 .envtmp test/tests_failed/$local_test_id.env
    echo 'Test failed.'
    echo 'See test/test.log for more details.'
    echo '.Env file has been moved to tests_failed directory.' >> test/test.log
    echo 'Test failed.' >> test/test.log
    failed=$((failed + 1))
    docker stop test_container
    pkill -P $pid
    echo '' >> test/test.log
    # exit 1
}

function test_finished(){
    echo 'Test passed.'
    echo 'Test passed.' >> test/test.log
    passed=$((passed + 1))
    docker stop test_container
    pkill -P $pid
    echo '' >> test/test.log
}

function ending() {
    echo ''
    echo 'Ending the tests...'
    echo $passed/$tests_run tests passed.
    echo $failed/$tests_run tests failed.
    if [ $failed -eq 0 ]; then
        echo 'TEST PASSED.'
    else
        echo 'TEST FAILED.'
    fi
    docker stop test_container
    pkill -P $pid
    exit 1
}

# Fonction pour construire l'image Docker
build_docker_image() {
    local docker_image="$1"
    local config_file="$2"
    test/run_server.sh $docker_image $2 # > /dev/null 2>&1
}

function ctrl_c() {
    echo ''
    echo "Ctrl + C happened, attempting to stop the server..."
    rm build_finished
    rm .envtmp
    ending
}


# Attend la création du fichier avec un timeout de 600 secondes
wait_for_file_creation_with_timeout() {
    local file="$1"
    local timeout=600  # 10 minutes en secondes
    local start_time=$(date +%s)
    
    while [ ! -f "$file" ]; do
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        if [ $elapsed_time -ge $timeout ]; then
            echo "Timeout. The docker image took too long to be built." >> test/test.log
            return 1
        fi
        sleep 1
    done
    sleep 1
    if ps -p $pid > /dev/null; then
        process_running=true
    else
        echo "Docker building process failed." | tee -a test/test.log
        rm $file
        return 1
    fi
    echo "File $file has been created. Docker image has been successfully built in $elapsed_time sec." | tee -a test/test.log
    rm $file
    return 0
}



check_http_server_availability() {
    local server="$1"
    local total_wait_time=600  # 10 minutes en secondes
    local retry_interval=5    # Interval entre les tentatives (en secondes)
    local elapsed_time=0

    while [ $elapsed_time -lt $total_wait_time ]; do
        # Test de la disponibilité du serveur HTTP
        curl -s --head --request GET "$server" | grep "200 OK"
        if [ $? -eq 0 ]; then
            echo "The server $server is available after $elapsed_time sec." | tee -a test/test.log
            sleep 2
            return 0
        fi

        # Attendre avant la prochaine tentative
        sleep $retry_interval
        elapsed_time=$((elapsed_time + retry_interval))
    done

    echo "The server $server is not available after $total_wait_time seconds, server launching must have failed." | tee -a test/test.log
    return 1
}

make_env()
{
    local env_file="$1"
    cp $env_file .envtmp
    if [ -z "$2" ]; then
        return 0
    else
        echo $2 >> test/test.log
        echo $2 >> .envtmp
    fi
    if [ -z "$3" ]; then
        return 0
    else
        echo $3 >> test/test.log
        echo $3 >> .envtmp
    fi
    if [ -z "$4" ]; then
        return 0
    else
        echo $4 >> test/test.log
        echo $4 >> .envtmp
    fi

}

process_test()
{
    echo '' >> test/test.log
    
    echo Test $test_id >> test/test.log
    echo Docker image: $1 >> test/test.log
    echo Audio file: $3 >> test/test.log
    echo Test type: $4 >> test/test.log
    echo ''
    echo Starting test $test_id
    local config_file="$2"
    make_env $config_file $5 $6 $7
    local local_test_id=$test_id
    test_id=$((test_id + 1))
    tests_run=$((tests_run + 1))
    local docker_image="$1"
    local test_file="$3"
    local test_type="$4"
    # Exécute la fonction de construction dans un sous-processus
    build_docker_image $docker_image .envtmp &
    pid=$!
    echo "The server is creating and will be running with the PID $pid." | tee -a test/test.log
    
    # Attend la création du fichier avec un timeout de 600 secondes
    wait_for_file_creation_with_timeout build_finished 
    local r=$?
    if [ "$r" -ne 0 ]; then
        mv $2 tests_failed/$local_test_id.env
        test_failed $2
        return 1
    fi
    check_http_server_availability "http://localhost:8080/healthcheck"
    local r=$?
    if [ "$r" -ne 0 ]; then
        mv $2 tests_failed/$local_test_id.env
        test_failed $2
        return 1
    fi
    if [ "$test_file" == "test/GOLE7.wav" ] ; then
        regex=".*Je crois que j'avais des profs.*"
    elif [ "$test_file" == "test/bonjour.wav" ]; then
        regex=".*Bonjour.*"
    fi
    if [ "$test_type" == "decoding" ]; then
        local start_time=$(date +%s)
        local res=$(curl -X POST "http://localhost:8080/transcribe" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@$test_file;type=audio/wav" 2>/dev/null)
        local end_time=$(date +%s)
        if [ -z "$res" ]; then
            echo "The server didn't transcribed, retrying in 10sec">> test/test.log
            sleep 10
            start_time=$(date +%s)
            res=$(curl -X POST "http://localhost:8080/transcribe" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@$test_file;type=audio/wav" 2>/dev/null)
            end_time=$(date +%s)
        fi
        echo "The server has transcribed $3 in $((end_time - start_time)) sec." >> test/test.log
       
        if [[ $res =~ $regex ]]; then
            echo "The string is matching the regex ($regex), the server must has successfully transcribed." >> test/test.log
            test_finished $2
            return 0
        else
            echo "The string is not matching the regex ($regex), the server didn't transcribed correctly. Output text : $res" >> test/test.log
            test_failed $2
        fi
    elif [ "$test_type" == "streaming" ]; then
        echo "Starting streaming test" >> test/test.log
        res=$(python3 test/test_streaming.py --audio_file $test_file)
        if [[ $res =~ $regex ]]; then
            echo "The string is matching the regex ($regex), the server must has successfully transcribed." >> test/test.log
            test_finished $2
            return 0
        else
            echo "The string is not matching the regex ($regex), the server didn't transcribed correctly. Output text : $res" >> test/test.log
            test_failed $2
        fi
    else
        echo "Test type $test_type not supported." >> test/test.log
        test_failed $2
    fi
    return 1
}

test_id=0
trap ctrl_c INT
echo Starting tests at $(date '+%d/%m/%Y %H:%M:%S') > test/test.log
echo '' >> test/test.log

for serving in decoding streaming;do
    for vad in False auditok silero; do
        for device in cpu cuda; do
            process_test whisper/Dockerfile.ctranslate2 test/.envtest test/bonjour.wav $serving DEVICE=$device VAD=$vad
        done
    done
done

ending