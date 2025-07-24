import sys
import os
import json
import datetime
import spacy

output_dir = sys.argv[1]
aligned_json = os.path.join(output_dir, "aligned_segments.json")
diar_json = os.path.join(output_dir, "nemo_diarization.json")
scene_file = os.path.join(output_dir, "scene_boundaries.json")

def format_timestamp(seconds):
    td = datetime.timedelta(seconds=seconds)
    ms = int(td.total_seconds() * 1000)
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms = ms % 1000
    return f"{h:02}:{m:02}:{s:02}:{ms:03}"

with open(aligned_json, "r") as f:
    segments = json.load(f)
with open(diar_json, "r") as f:
    diar = json.load(f)

for seg in segments:
    mid = (seg["start"] + seg["end"]) / 2
    speaker = "UNKNOWN"
    for d in diar:
        if d["start"] <= mid <= d["end"]:
            speaker = d["speaker"]
            break
    seg["speaker"] = speaker

merged = []
max_gap = 1.5
current = None
for seg in segments:
    if current is None:
        current = dict(seg)
    elif seg["speaker"] == current["speaker"] and seg["start"] - current["end"] <= max_gap:
        current["end"] = seg["end"]
        current["text"] += " " + seg["text"]
        current["words"].extend(seg.get("words", []))
    else:
        merged.append(current)
        current = dict(seg)
if current:
    merged.append(current)

nlp = spacy.load("en_core_web_sm")
sesame_characters = {"elmo", "big bird", "cookie monster", "bert", "ernie", "abby", "grover", "oscar", "count", "zoe", "rosita", "snuffy", "telly", "baby bear", "prairie dawn", "herry"}
sesame_places = {"123 sesame street", "hooper's store", "oscar's can", "elmo's world", "big bird's nest"}

def is_letter_of_day(text): return "letter of the day" in text.lower()
def is_number_of_day(text): return "number of the day" in text.lower()

scenes = []
if os.path.exists(scene_file):
    with open(scene_file, "r") as f:
        scenes = json.load(f)

def get_scene_markers(seg_start, seg_end):
    markers = []
    for i, (start, end) in enumerate(scenes):
        if seg_start <= start <= seg_end:
            markers.append(f"[Scene {i+1} Starts {format_timestamp(start)}]")
        if seg_start <= end <= seg_end:
            markers.append(f"[Scene {i+1} Ends {format_timestamp(end)}]")
    return " ".join(markers)

md_path = os.path.join(output_dir, "transcription.md")
with open(md_path, "w") as f:
    for seg in merged:
        seg_start = seg["start"]
        seg_end = seg["end"]
        speaker = seg["speaker"]
        text = seg["text"].strip()
        scene_marker = get_scene_markers(seg_start, seg_end)
        f.write(f"[{format_timestamp(seg_start)} - {format_timestamp(seg_end)}] {speaker} {scene_marker}\n")
        if speaker == "SILENCE":
            f.write("[silence]\n")
        else:
            f.write(f"{text}\n")
            doc = nlp(text)
            ner_labels = {ent.text.lower() for ent in doc.ents}
            mentions = set()
            character_hits = sesame_characters.intersection(ner_labels)
            if character_hits:
                mentions.update(s.title() for s in character_hits)
            place_hits = sesame_places.intersection(text.lower())
            if place_hits:
                mentions.update(s.title() for s in place_hits)
            if is_letter_of_day(text):
                mentions.add("Letter Of The Day")
            if is_number_of_day(text):
                mentions.add("Number Of The Day")
            if mentions:
                f.write(f"(NER Labels: {', '.join(sorted(mentions))})\n")
        f.write("\n\n")

json_path = os.path.join(output_dir, "transcription.json")
for seg in merged:
    duration = seg["end"] - seg["start"]
    seg["wpm"] = round(len(seg["text"].split()) / (duration / 60), 2) if duration > 0 else 0
with open(json_path, "w") as f:
    json.dump(merged, f, indent=2)