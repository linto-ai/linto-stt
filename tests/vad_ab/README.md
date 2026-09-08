# VAD A/B: hallucinations vs speech loss (manual procedure)

Whisper hallucinates subtitle credits ("Sous-titrage ST' 501", "Société
Radio-Canada", "www...") on the silence, noise or music left inside the chunks it
receives. The per-chunk VAD (`VAD=auditok|silero`) is what keeps that audio away
from the model, so it must (a) remove non-speech and (b) not drop quiet or
noisy speech. This folder gives a repeatable way to measure both on any audio.
Nothing here is shipped in the Docker image (the Dockerfile only copies
`linto_stt/`, `tests/bonjour.wav` and the entrypoint).

## 1. Build a test set

    python tests/vad_ab/make_testset.py my_speech.wav /tmp/vadset --music my_music.mp3

`my_speech.wav`: at least ~6 min of speech (a meeting recording is the most
relevant: several speakers, far microphones). Output: `conditions/` (one file per
condition) and `composite.wav` + `composite.json` (regions with known
positions, long non-speech blocks glued to speech).

## 2. Start one server per configuration to compare

    docker run -d --name vad_false   --gpus all -p 127.0.0.1:8097:80 -v stt-models:/var/www/.cache \
      -e SERVICE_MODE=http -e MODEL=large-v3-turbo -e LANGUAGE=fr -e DEVICE=cuda -e VAD=false \
      lintoai/linto-stt-whisper:latest
    docker run -d --name vad_auditok ... -p 127.0.0.1:8098:80 ... -e VAD=auditok ...
    docker run -d --name vad_silero  ... -p 127.0.0.1:8095:80 ... -e VAD=silero  ...

Set the same `MODEL`, `PROMPT`, `LANGUAGE` as the deployment under test.

## 3. Per-condition files (what each VAD does on pure content)

    python tests/vad_ab/run_ab.py false=8097 auditok=8098 silero=8095 -- /tmp/vadset/conditions/*.wav

Expected: 0 words on `silence`, `pink_noise`, `white_noise`, `room_noise`,
`music`; the same word count on `speech_quiet_-45dB` as on `speech_ref_-45dB`.

## 4. Production-like fragments

In production the file is split upstream by linto-transcription-service
(`transcriptionservice/transcription/utils/audio.py:splitFile`, WebRTC mode 1,
cuts in the middle of silences >= 0.6 s, `minDuration` sent by the client, 30 s
for LinTO Studio). Fragments therefore always contain speech plus halves of the
surrounding silence/noise/music. Reproduce that split with the real code:

    docker run --rm -v /tmp/vadset:/work --entrypoint python lintoai/linto-transcription-service:latest -c "
    from transcriptionservice.transcription.utils.audio import splitFile
    subfiles, stats = splitFile('/work/composite.wav', method='WebRTC', min_segment_duration=30,
                                max_segment_duration=1200.0, min_length=30)
    import os
    for path, offset, duration in subfiles:
        os.rename(path, path.rsplit('_', 1)[0] + f'_{offset:.2f}.wav')
        print(path, offset, duration)"

then

    python tests/vad_ab/run_ab.py --regions /tmp/vadset/composite.json \
      false=8097 auditok=8098 silero=8095 -- /tmp/vadset/composite_*.wav

The line per server gives the number of words attributed to each region kind:
words under `silence`/`room`/`music`/`white` are hallucinations, words under
`speech*` must match between configurations.

## Reference measurements (2026-09-08, large-v3-turbo, RTX 4090, French conference audio)

Composite with 90-120 s blocks, split by the transcription-service splitter
(5 fragments), hallucinated words per non-speech region / words kept in the
five 60 s speech regions (655 in the reference):

| config | silence | room noise | music | white noise | speech |
|---|---|---|---|---|---|
| 2.1.1 VAD=false | 7 | 10 | 8 | 4 | 654 |
| 2.1.1 VAD=auditok | 0 | 0 | 8 | 4 | 653 |
| 2.1.1 VAD=silero | 0 | 0 | 0 | 0 | 655 |
| 2.1.0 VAD=auditok | 0 | 0 | 0 | 0 | 601 |

Quiet speech at -45 dB (30 s, 70 words): auditok 55, silero 70, no VAD 70.
Pure 30 s files with no speech at all: before this fix every VAD setting
decoded the whole file (2-4 hallucinated words each); with the fix `silero`
gives 0 words on all of them, `auditok` still decodes music and white noise
(its per-file volume normalisation turns them into "speech").

`VAD=silero` here is the Silero v6 ONNX model bundled in faster-whisper
(`faster_whisper/assets/silero_vad_v6.onnx`, threshold 0.5), with
`VAD_MIN_SPEECH_DURATION`/`VAD_MIN_SILENCE_DURATION` (0.1 s) and
`VAD_DILATATION` (0.5 s) applied on top.
