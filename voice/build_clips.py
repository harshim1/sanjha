"""
Build-time voice library (ElevenLabs). Generates a fixed set of 35 short
Swahili clips ONCE; they ship inside the offline web app and are stitched
together on the phone to read any card aloud — no cloud call at run time,
so the "core feature works offline" rule holds.

    export ELEVENLABS_API_KEY=...            (never commit this)
    python -m voice.build_clips              generate missing clips
    python -m voice.build_clips --list       show the clip list, no API calls
    python -m voice.build_clips --force      regenerate everything

Optional: ELEVENLABS_VOICE_ID (default below), ELEVENLABS_MODEL (default
eleven_v3 — per the ElevenLabs models page, eleven_multilingual_v2 and the
flash models do NOT list Swahili).

Any price 1-999 is spoken as hundreds + "na" + tens + "na" + units
(380 = "mia tatu" "na" "themanini"), so 27 number clips cover every card.

Before shipping: have a Swahili speaker LISTEN to the stitched card (single
words generated in isolation can come out with odd intonation), and confirm
the licence terms of the ElevenLabs plan used cover this use.
"""
from __future__ import annotations

import json
import os
import sys
import time

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "web", "voice")
API = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"  # an ElevenLabs premade voice; pick one that sounds right in Swahili
DEFAULT_MODEL = "eleven_v3"
OUTPUT_FORMAT = "mp3_22050_32"  # small files: the whole library should stay well under 1 MB

UNITS = ["moja", "mbili", "tatu", "nne", "tano", "sita", "saba", "nane", "tisa"]
TENS = ["kumi", "ishirini", "thelathini", "arobaini", "hamsini", "sitini", "sabini", "themanini", "tisini"]

CLIPS = {
    "p_fair": "Bei sawa wiki hii ni kati ya",  # the fair price this week is between
    "p_and": "na",  # and
    "p_perkg": "shilingi kwa kilo.",  # shillings per kilo
    "p_below": "Ukipewa chini ya",  # if you are offered below
    "p_ask": "uliza ushirika kwanza.",  # ask the cooperative first
    "p_right": "Tumekuwa sahihi wiki",  # we have been right in weeks
    "p_outof": "kati ya",  # out of
    "p_abstain": "Bei inayumba sana wiki hii. Sina uhakika. Uliza ushirika.",
    **{f"u{i + 1}": w for i, w in enumerate(UNITS)},
    **{f"t{i + 1}": w for i, w in enumerate(TENS)},
    **{f"h{i + 1}": f"mia {w}" for i, w in enumerate(UNITS)},
}


def number_clips(n: int) -> list[str]:
    """Clip ids that speak n (1-999). Mirrors numberClips() in app/web/app.js."""
    if not 1 <= n <= 999:
        raise ValueError(n)
    parts = [f"{p}{d}" for p, d in (("h", n // 100), ("t", n % 100 // 10), ("u", n % 10)) if d]
    out = []
    for part in parts:
        out += (["p_and"] if out else []) + [part]
    return out


def synthesize(text: str, key: str, voice: str, model: str) -> bytes:
    import httpx

    body = {"text": text, "model_id": model, "language_code": "sw"}
    for attempt in range(2):
        r = httpx.post(API.format(voice_id=voice), params={"output_format": OUTPUT_FORMAT}, json=body,
                       headers={"xi-api-key": key}, timeout=60)
        if r.status_code == 200:
            return r.content
        if attempt == 0 and r.status_code in (400, 422) and "language_code" in body:
            body.pop("language_code")  # some models reject an explicit language code
            continue
        raise RuntimeError(f"ElevenLabs {r.status_code}: {r.text[:300]}")
    raise RuntimeError("unreachable")


def main(argv: list[str]) -> int:
    if "--list" in argv:
        for cid, text in CLIPS.items():
            print(f"{cid:10s} {text}")
        print(f"{len(CLIPS)} clips; e.g. 380 -> {number_clips(380)}, 12 -> {number_clips(12)}")
        return 0
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        print("ELEVENLABS_API_KEY is not set — nothing generated. See this file's docstring.", file=sys.stderr)
        return 1
    voice = os.environ.get("ELEVENLABS_VOICE_ID", DEFAULT_VOICE)
    model = os.environ.get("ELEVENLABS_MODEL", DEFAULT_MODEL)
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for cid, text in CLIPS.items():
        path = os.path.join(OUT_DIR, f"{cid}.mp3")
        if os.path.exists(path) and "--force" not in argv:
            total += os.path.getsize(path)
            continue
        audio = synthesize(text, key, voice, model)
        with open(path, "wb") as f:
            f.write(audio)
        total += len(audio)
        print(f"wrote {cid}.mp3 ({len(audio) / 1024:.1f} KB)  {text}")
        time.sleep(0.3)
    manifest = {"clips": sorted(CLIPS), "texts": CLIPS, "model": model, "voice_id": voice, "format": OUTPUT_FORMAT,
                "generated": time.strftime("%Y-%m-%d"), "source": "ElevenLabs text-to-speech (synthetic voice)"}
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(f"{len(CLIPS)} clips, {total / 1024:.0f} KB total -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
