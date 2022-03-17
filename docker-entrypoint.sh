#!/bin/bash
<<<<<<< HEAD
set -e

max_attempts=3
delay=5

punctuation_state=0
if [[ ! -z $PUCTUATION_HOST && ! -z $PUCTUATION_PORT ]]; then
for retry in $(seq 1 $max_attempts); do
    echo "Waiting punctuation service... [attempt=$retry]"
    punctuation_state=1
    ./wait-for-it.sh $PUCTUATION_HOST:$PUCTUATION_PORT --timeout=$delay || punctuation_state=0
    if [ $punctuation_state == 1 ]; then break; fi
done
fi

spkdiarization_state=0
if [[ ! -z $SPEAKER_DIARIZATION_HOST && ! -z $SPEAKER_DIARIZATION_PORT ]]; then
for retry in $(seq 1 $max_attempts); do
    echo "Waiting speaker diarization service... [attempt=$retry]"
    spkdiarization_state=1
    ./wait-for-it.sh $SPEAKER_DIARIZATION_HOST:$SPEAKER_DIARIZATION_PORT --timeout=$delay || spkdiarization_state=0
    if [ $spkdiarization_state == 1 ]; then break; fi
done
fi

echo "Start service"

python3 ./run.py --puctuation $punctuation_state --speaker_diarization $spkdiarization_state
=======
set -ea

echo "RUNNING STT"

# Check model
echo "Checking model format ..."
if [ -z "$MODEL_TYPE" ]
then
    echo "Model type not specified, expecting Vosk Model"
    export MODEL_TYPE=vosk
fi

if  [ "$MODEL_TYPE" = "vosk" ]
then
    echo "Using Vosk format's model"

elif [ "$MODEL_TYPE" = "lin" ]
then
    echo "Processing model ... "
    ./lin_to_vosk.py
else
    echo "Unknown model type $MODEL_TYPE. Assuming vosk model"
fi
# Launch parameters, environement variables and dependencies check
if [ -z "$SERVICE_MODE" ]
then
    echo "ERROR: Must specify a serving mode: [ http | task | websocket ]"
    exit -1
else
    if [ "$SERVICE_MODE" = "http" ] 
    then
        echo "RUNNING STT HTTP SERVER"
        python http_server/ingress.py --debug
    elif [ "$SERVICE_MODE" == "task" ]
    then
        if [[ -z "$SERVICES_BROKER" ]]
        then 
            echo "ERROR: SERVICES_BROKER variable not specified, cannot start celery worker."
            exit -1
        fi
        /usr/src/app/wait-for-it.sh $(echo $SERVICES_BROKER | cut -d'/' -f 3) --timeout=20 --strict -- echo " $SERVICES_BROKER (Service Broker) is up"
        echo "RUNNING STT CELERY WORKER"
        celery --app=celery_app.celeryapp worker -Ofair --queues=${SERVICE_NAME} -c ${CONCURRENCY} -n ${SERVICE_NAME}_worker@%h

    elif [ "$SERVICE_MODE" == "websocket" ]
    then
        echo "Running Websocket server on port ${STREAMING_PORT:=80}"
        python websocket/websocketserver.py
    else
        echo "ERROR: Wrong serving command: $1"
        exit -1
    fi
fi

echo "Service stopped"
>>>>>>> next
