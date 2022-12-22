#!/bin/bash
set -ea

echo "RUNNING STT"

# Check model
echo "Checking model format ..."
if [ -z "$MODEL_TYPE" ]
then
    echo "Model type not specified, choosing Whisper medium model"
    export MODEL_TYPE=medium
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

    else
        echo "ERROR: Wrong serving command: $1"
        exit -1
    fi
fi

echo "Service stopped"