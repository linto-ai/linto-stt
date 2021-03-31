#!/bin/bash
set -e

max_attempts=3
delay=5

for retry in $(seq 1 $max_attempts); do
    echo "Waiting punctuation service... [attempt=$retry]"
    punctuation_state=1
    ./wait-for-it.sh $PUCTUATION_HOST:$PUCTUATION_PORT --timeout=$delay || punctuation_state=0
done

if [ $punctuation_state == 1 ]; then
    echo "$PUCTUATION_HOST:$PUCTUATION_PORT is up"
else
    echo "punctuation service is not runninig"
fi

for retry in $(seq 1 $max_attempts); do
    echo "Waiting speaker diarization service... [attempt=$retry]"
    spkdiarization_state=1
    ./wait-for-it.sh $SPEAKER_DIARIZATION_HOST:$SPEAKER_DIARIZATION_PORT --timeout=$delay || spkdiarization_state=0
done

if [ $spkdiarization_state == 1 ]; then
    echo "$SPEAKER_DIARIZATION_HOST:$SPEAKER_DIARIZATION_PORT is up"
else
    echo "speaker diarization service is not runninig"
fi

echo "RUNNING service"

python3 ./run.py --puctuation $punctuation_state --speaker_diarization $spkdiarization_state