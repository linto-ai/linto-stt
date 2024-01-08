#!/bin/bash
set -a

echo "RUNNING STT"

# Check model
echo "Checking model format ..."
if [ -z "$MODEL" ]
then
    echo "Model type not specified, choosing Whisper medium model"
    export MODEL=medium
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
        python3 http_server/ingress.py --debug
    elif [ "$SERVICE_MODE" == "task" ]
    then
        if [[ -z "$SERVICES_BROKER" ]]
        then 
            echo "ERROR: SERVICES_BROKER variable not specified, cannot start celery worker."
            exit -1
        fi
        nvidia-smi 2> /dev/null > /dev/null
        if [ $? -eq 0 ];then
            echo "GPU detected"
            GPU=1
            OPT="--pool=solo"
        else
            echo "No GPU detected"
            GPU=0
            OPT=""
        fi
        /usr/src/app/wait-for-it.sh $(echo $SERVICES_BROKER | cut -d'/' -f 3) --timeout=20 --strict -- echo " $SERVICES_BROKER (Service Broker) is up" || exit 1
        echo "RUNNING STT CELERY WORKER"
        celery --app=celery_app.celeryapp worker $OPT -Ofair --queues=${SERVICE_NAME} -c ${CONCURRENCY} -n ${SERVICE_NAME}_worker@%h

    else
        echo "ERROR: Wrong serving command: $SERVICE_MODE"
        exit -1
    fi
fi

echo "Service stopped"