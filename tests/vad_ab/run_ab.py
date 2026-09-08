"""Send audio files to one or more running linto-stt-whisper HTTP servers and
compare the outputs: words per file, words falling in non-speech regions
(hallucinations) and words in speech regions (loss).

usage: run_ab.py [--regions composite.json] label=port [label=port ...] -- FILE.wav [FILE.wav ...]

Each FILE is sent as-is to every server (one request per file). With --regions
(the composite.json written by make_testset.py) the word timestamps are
attributed to the regions of that composite, so FILEs must then be the
composite itself or fragments of it named <anything>_<offset_seconds>.wav
(offset = where the fragment starts in the composite, e.g. the fragments
produced by the transcription-service splitter can be renamed that way).
"""
import json
import re
import sys

import requests

HALU = re.compile(r"sous[- ]?titr|radio[- ]?canada|amara|merci d'avoir regardé|abonnez|www\.|©", re.I)


def transcribe(port, path):
    with open(path, "rb") as f:
        r = requests.post(f"http://127.0.0.1:{port}/transcribe", headers={"accept": "application/json"},
                          files={"file": (path.split("/")[-1], f, "audio/wav")}, timeout=3600)
    r.raise_for_status()
    d = r.json()
    return json.loads(d) if isinstance(d, str) else d


def main():
    args = sys.argv[1:]
    regions = None
    if "--regions" in args:
        i = args.index("--regions")
        regions = json.load(open(args[i + 1]))
        del args[i:i + 2]
    sep = args.index("--")
    servers = dict(a.split("=") for a in args[:sep])
    files = args[sep + 1:]
    for label, port in servers.items():
        total, halu, per_kind = 0, 0, {}
        for path in files:
            d = transcribe(port, path)
            words = d.get("words", [])
            total += len(words)
            halu += len(HALU.findall(d.get("text", "")))
            if regions:
                m = re.search(r"_(\d+(?:\.\d+)?)\.wav$", path)
                off = float(m.group(1)) if m else 0.0
                for w in words:
                    t = off + w["start"]
                    kind = next((r["kind"] for r in regions if r["start"] <= t < r["end"]), "?")
                    per_kind[kind] = per_kind.get(kind, 0) + 1
            else:
                print(f"  {label:14s} {path.split('/')[-1]:36s} {len(words):5d} words  {d.get('text', '')[:80]!r}")
        line = f"{label:14s} total={total:5d} halu-phrases={halu}"
        if regions:
            line += "  words per region kind: " + ", ".join(f"{k}={v}" for k, v in sorted(per_kind.items()))
        print(line)


if __name__ == "__main__":
    main()
