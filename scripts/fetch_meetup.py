#!/usr/bin/env python3
"""Refresh _data/meetup_events.json from the AI Office Hours Meetup calendar.

Run it locally after adding or changing an event on Meetup, then commit
_data/meetup_events.json and push:

    python3 scripts/fetch_meetup.py            # fetch from Meetup
    python3 scripts/fetch_meetup.py feed.ics   # parse a saved .ics file

Standard library only. If Meetup can't be reached, the existing JSON file is
left untouched so the site still builds with the last known events.
"""
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

FEED_URL = "https://www.meetup.com/machine-learning-workshop/events/ical/"
OUT = Path(__file__).resolve().parent.parent / "_data" / "meetup_events.json"
GROUP_NAME = "AI Office Hours"
MAX_EVENTS = 4
DESC_CHARS = 220
LOCAL_TZ = ZoneInfo("America/New_York")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "jue-guo.com site builder"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def unfold(text):
    # RFC 5545: a line starting with a space/tab continues the previous one.
    return re.sub(r"\r?\n[ \t]", "", text)


def unescape(value):
    # \n or \N -> newline; any other backslash escape (\, \; \\ ...) -> the character
    return re.sub(r"\\(.)", lambda m: "\n" if m.group(1) in "nN" else m.group(1), value)


def parse_dt(prop, value):
    """prop is e.g. 'DTSTART;TZID=America/New_York'."""
    m = re.search(r"TZID=([^;:]+)", prop)
    if value.endswith("Z"):
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    fmt = "%Y%m%dT%H%M%S" if "T" in value else "%Y%m%d"
    dt = datetime.strptime(value, fmt)
    return dt.replace(tzinfo=ZoneInfo(m.group(1)) if m else LOCAL_TZ)


def clean_description(raw, title):
    text = unescape(raw)
    text = re.sub(r"https?://\S+", "", text)
    lines = [l.strip() for l in text.split("\n")]
    keep = []
    for l in lines:
        if not l or l == GROUP_NAME:
            continue
        bare = l.strip("*_ ").strip()
        if not bare:
            continue
        # skip short heading-like lines (fully bold, no sentence end) and a repeat of the title
        is_heading = re.fullmatch(r"\*\*.+\*\*", l) and len(bare) < 70 and not bare.endswith((".", "!", "?"))
        if is_heading or bare.lower() == title.lower():
            continue
        keep.append(l)
    text = " ".join(keep)
    text = re.sub(r"[*_`#>]+", "", text)          # drop markdown emphasis
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = ""
    for s in sentences:
        if not out:
            out = s
        elif len(out) + 1 + len(s) <= DESC_CHARS:
            out += " " + s
        else:
            break
    if len(out) > DESC_CHARS + 40:
        out = out[:DESC_CHARS].rsplit(" ", 1)[0].rstrip(",;:—-") + "…"
    return out


def fmt_when(start, end):
    s = start.astimezone(LOCAL_TZ)
    day = s.strftime("%a, %b %-d")
    if s.year != datetime.now(LOCAL_TZ).year:
        day += s.strftime(", %Y")
    t = s.strftime("%-I:%M %p")
    return f"{day} · {t} {s.strftime('%Z')}"


def parse(ics):
    ics = unfold(ics)
    events = []
    for block in ics.split("BEGIN:VEVENT")[1:]:
        block = block.split("END:VEVENT")[0]
        props = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            props.setdefault(key.split(";")[0], (key, value))
        if "DTSTART" not in props or "SUMMARY" not in props:
            continue
        if props.get("STATUS", ("", ""))[1].upper() == "CANCELLED":
            continue
        start = parse_dt(*props["DTSTART"])
        end = parse_dt(*props["DTEND"]) if "DTEND" in props else start
        title = unescape(props["SUMMARY"][1]).strip()
        url = props.get("URL", ("", ""))[1].split("?")[0].strip()
        desc = clean_description(props.get("DESCRIPTION", ("", ""))[1], title)
        events.append({
            "title": title,
            "url": url,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "when": fmt_when(start, end),
            "month": start.astimezone(LOCAL_TZ).strftime("%b"),
            "day": start.astimezone(LOCAL_TZ).strftime("%-d"),
            "desc": desc,
        })
    now = datetime.now(timezone.utc)
    upcoming = [e for e in events if datetime.fromisoformat(e["end"]) > now]
    upcoming.sort(key=lambda e: e["start"])
    return upcoming[:MAX_EVENTS]


def main():
    try:
        ics = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else fetch(FEED_URL)
        events = parse(ics)
    except Exception as exc:  # network hiccup, Meetup change, etc.
        print(f"Meetup feed unavailable ({exc}); keeping {OUT.name} as is.")
        return 0
    OUT.write_text(json.dumps(events, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(events)} upcoming event(s) to {OUT}")
    for e in events:
        print(f"  {e['when']}  {e['title']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
