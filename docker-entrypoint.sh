#!/bin/bash
set -e

max_attempts=3
delay=5

for retry in $(seq 1 $max_attempts); do
    echo "Waiting punctuation service... [attempt=$retry]"
    punctuation_state=1
    ./wait-for-it.sh $PUCTUATION_HOST:$PUCTUATION_PORT --timeout=$delay || punctuation_state=0
    if [ $punctuation_state == 1 ]; then break; fi
done

for retry in $(seq 1 $max_attempts); do
    echo "Waiting speaker diarization service... [attempt=$retry]"
    spkdiarization_state=1
    ./wait-for-it.sh $SPEAKER_DIARIZATION_HOST:$SPEAKER_DIARIZATION_PORT --timeout=$delay || spkdiarization_state=0
    if [ $spkdiarization_state == 1 ]; then break; fi
done

echo "RUNNING service"

python3 ./run.py --puctuation $punctuation_state --speaker_diarization $spkdiarization_state