"""Build the VAD/hallucination test set from a speech WAV you provide.

usage: make_testset.py SPEECH.wav OUT_DIR [--music MUSIC_FILE]

SPEECH.wav: any recording with continuous speech (>= 5 min recommended),
converted to 16 kHz mono PCM16 here. MUSIC_FILE (optional): any audio file with
instrumental music. Nothing is downloaded and no audio is shipped with the repo.

Produces, in OUT_DIR:
  conditions/  one 30 s (or 60 s) file per condition: speech, quiet speech
               (-25/-35/-45 dB), speech + pink noise (SNR ~10 dB), digital
               silence, pink noise, white noise, low room noise, music
  composite.wav  the conditions concatenated at KNOWN positions (see layout
               printed at the end and written to composite.json) so that words
               can be attributed to a speech or a non-speech region
Requires ffmpeg.
"""
import json
import os
import subprocess
import sys


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    speech, out = sys.argv[1], sys.argv[2]
    music = sys.argv[sys.argv.index("--music") + 1] if "--music" in sys.argv else None
    cond = os.path.join(out, "conditions")
    os.makedirs(cond, exist_ok=True)
    base = os.path.join(out, "speech_16k.wav")
    ff("-i", speech, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", base)

    def seg(name, start, dur, filt=None):
        args = ["-ss", str(start), "-t", str(dur), "-i", base]
        if filt:
            args += ["-af", filt]
        ff(*args, "-c:a", "pcm_s16le", os.path.join(cond, name))

    seg("speech_60s.wav", 0, 60)
    seg("speech_b_60s.wav", 60, 60)
    seg("speech_c_60s.wav", 120, 60)
    seg("speech_quiet_-25dB_60s.wav", 180, 60, "volume=-25dB")
    seg("speech_ref_quiet_60s.wav", 180, 60)
    seg("speech_quiet_-35dB_30s.wav", 240, 30, "volume=-35dB")
    seg("speech_quiet_-45dB_30s.wav", 270, 30, "volume=-45dB")
    seg("speech_ref_-35dB_30s.wav", 240, 30)
    seg("speech_ref_-45dB_30s.wav", 270, 30)
    ff("-i", base, "-filter_complex",
       "[0:a]atrim=300:330,asetpts=PTS-STARTPTS,volume=-10dB[s];"
       "anoisesrc=r=16000:color=pink:amplitude=0.03:seed=3,atrim=0:30,asetpts=PTS-STARTPTS,"
       "aformat=channel_layouts=mono[n];[s][n]amix=inputs=2:duration=first:normalize=0[out]",
       "-map", "[out]", "-c:a", "pcm_s16le", os.path.join(cond, "speech_noisy_snr10dB_30s.wav"))
    ff("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "30", "-c:a", "pcm_s16le",
       os.path.join(cond, "silence_30s.wav"))
    ff("-f", "lavfi", "-i", "anoisesrc=r=16000:color=pink:amplitude=0.01:seed=1", "-t", "30",
       "-ac", "1", "-c:a", "pcm_s16le", os.path.join(cond, "pink_noise_30s.wav"))
    ff("-f", "lavfi", "-i", "anoisesrc=r=16000:color=white:amplitude=0.02:seed=2", "-t", "30",
       "-ac", "1", "-c:a", "pcm_s16le", os.path.join(cond, "white_noise_30s.wav"))
    ff("-f", "lavfi", "-i", "anoisesrc=r=16000:color=pink:amplitude=0.003:seed=1", "-t", "120",
       "-ac", "1", "-c:a", "pcm_s16le", os.path.join(cond, "room_noise_120s.wav"))
    if music:
        ff("-ss", "20", "-t", "60", "-i", music, "-ar", "16000", "-ac", "1", "-af", "volume=-12dB",
           "-c:a", "pcm_s16le", os.path.join(cond, "music_60s.wav"))

    # Composite: long non-speech blocks glued to speech, like the fragments the
    # transcription-service WebRTC splitter produces in production.
    layout = [("speech_60s.wav", "speech"), ("silence_30s.wav", "silence"), ("silence_30s.wav", "silence"),
              ("silence_30s.wav", "silence"), ("speech_b_60s.wav", "speech"), ("room_noise_120s.wav", "room"),
              ("speech_c_60s.wav", "speech")]
    if music:
        layout += [("music_60s.wav", "music"), ("music_60s.wav", "music")]
    layout += [("speech_quiet_-25dB_60s.wav", "speech-quiet"), ("white_noise_30s.wav", "white"),
               ("white_noise_30s.wav", "white"), ("white_noise_30s.wav", "white"), ("speech_ref_quiet_60s.wav", "speech")]
    lst = os.path.join(out, "concat.txt")
    with open(lst, "w") as f:
        for name, _ in layout:
            f.write(f"file '{os.path.join(cond, name)}'\n")
    ff("-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", os.path.join(out, "composite.wav"))
    regions, t = [], 0.0
    for name, kind in layout:
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                             "-of", "csv=p=0", os.path.join(cond, name)]))
        regions.append({"start": round(t, 2), "end": round(t + dur, 2), "kind": kind})
        t += dur
    json.dump(regions, open(os.path.join(out, "composite.json"), "w"), indent=1)
    for r in regions:
        print(f"{r['start']:7.1f}-{r['end']:7.1f}  {r['kind']}")
    print("composite:", os.path.join(out, "composite.wav"))


if __name__ == "__main__":
    main()
