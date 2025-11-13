import os
import subprocess
import datetime
import whisperx
import json
from pyannote.audio import Pipeline
from pyannote.core import Segment
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import spacy

###USE THIS VERSION FOR TRANSCRIPTION WITH FEATURES###
# === Set Paths ===
base_dir = "/Volumes/LANDLAB/projects/Project_Sesame/ssa_sesame-street-archive/scripts/ssa_scaling/3_audio-transcriber/versions"
output_dir = os.path.join(base_dir, "versions", "25-07-09_12.14.00")
os.makedirs(output_dir, exist_ok=True)
audio_path = os.path.join(output_dir, "audio.wav")
video_path = os.path.join(output_dir, "scenechange.mp4")

# === Extract audio if not present ===
if not os.path.exists(audio_path):
    if os.path.exists(video_path):
        print(f"Extracting audio from {video_path} to {audio_path}...")
        ffmpeg_cmd = [
            "ffmpeg", "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)
    else:
        raise FileNotFoundError(f"Audio file not found at {audio_path} and video file not found at {video_path}")
else:
    print(f"Audio already exists at {audio_path}")

# === Load ASR Model ===
device = "cpu"
compute_type = "float32"
print("Loading WhisperX model...")
model = whisperx.load_model("large-v3", device, compute_type=compute_type)

# === Transcribe ===
print("Starting transcription...")
transcription = model.transcribe(audio_path)
segments = transcription["segments"]

# === Align words ===
print("Aligning segments...")
model_a, metadata = whisperx.load_align_model(language_code=transcription["language"], device=device)
result_aligned = whisperx.align(segments, model_a, metadata, audio_path, device)

# === Run Diarization ===
print("Loading diarization model...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token="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"
)
diarization = diarization_pipeline(audio_path)

# === Assign Speakers ===
print("Assigning speaker labels...")
speaker_mapping = {}
next_speaker_id = 1

for seg in result_aligned["segments"]:
    seg_start, seg_end = seg["start"], seg["end"]
    seg_interval = Segment(seg_start, seg_end)
    dia_crop = diarization.crop(seg_interval, mode="loose")
    speakers = dia_crop.labels()

    if not speakers:
        seg["speaker"] = "UNKNOWN"
    elif len(speakers) == 1:
        speaker = speakers[0]
    else:
        speaker = max(speakers, key=lambda s: dia_crop.label_duration(s))

    if speaker not in speaker_mapping:
        speaker_mapping[speaker] = f"SPEAKER {next_speaker_id}"
        next_speaker_id += 1
    seg["speaker"] = speaker_mapping[speaker]

# === Merge segments by speaker (0.8s max gap) & insert silences ===
merged_segments = []
current = None
max_gap = 0.8
for seg in result_aligned["segments"]:
    if current is None:
        current = seg
    elif seg["speaker"] == current["speaker"] and seg["start"] - current["end"] <= max_gap:
        current["end"] = seg["end"]
        current["text"] += " " + seg["text"]
        current["words"].extend(seg["words"])
    else:
        merged_segments.append(current)
        if seg["start"] - current["end"] > max_gap:
            merged_segments.append({
                "start": current["end"],
                "end": seg["start"],
                "speaker": "SILENCE",
                "text": "[silence]",
                "words": []
            })
        current = seg
if current:
    merged_segments.append(current)

# === Scene Detection ===
scene_boundaries = []
if os.path.exists(video_path):
    print("Running scene detection...")
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    scene_boundaries = [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

# === Load spaCy model ===
nlp = spacy.load("en_core_web_sm")
sesame_characters = {
    "elmo", "big bird", "cookie monster", "bert", "ernie",
    "abby", "grover", "oscar", "count", "zoe", "rosita",
    "snuffy", "telly", "baby bear", "prairie dawn", "herry"
}
sesame_places = {"123 sesame street", "hooper's store", "oscar's can", "elmo's world", "big bird's nest"}

def is_letter_of_day(text):
    return "letter of the day" in text.lower()

def is_number_of_day(text):
    return "number of the day" in text.lower()

# === Format timestamp ===
def format_timestamp(seconds):
    td = datetime.timedelta(seconds=seconds)
    total_ms = int(td.total_seconds() * 1000)
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    seconds = (total_ms % 60000) // 1000
    milliseconds = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}"

# === Build scene events for side display ===
scene_markers = []
for i, (start, end) in enumerate(scene_boundaries):
    scene_markers.append((start, f"⟵ **Scene {i+1} Starts [{format_timestamp(start)}]**"))
    scene_markers.append((end, f"⟶ **Scene {i+1} Ends [{format_timestamp(end)}]**"))
scene_markers.sort()

# === Write to Markdown file ===
md_path = os.path.join(output_dir, "transcription.md")
with open(md_path, "w") as f:
    marker_idx = 0
    for seg in merged_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        speaker = seg["speaker"]
        text = seg["text"].strip()

        markers_inline = []
        while marker_idx < len(scene_markers) and scene_markers[marker_idx][0] <= seg_end:
            markers_inline.append(scene_markers[marker_idx][1])
            marker_idx += 1

        marker_str = "    " + "    ".join(markers_inline) if markers_inline else ""

        f.write(f"[{format_timestamp(seg_start)} - {format_timestamp(seg_end)}] {speaker}:{marker_str}\n")

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

# === Write to JSON ===
json_path = os.path.join(output_dir, "transcription.json")
for seg in merged_segments:
    duration = seg["end"] - seg["start"]
    seg["wpm"] = round(len(seg["text"].split()) / (duration / 60), 2) if duration > 0 else 0
json_data = merged_segments
with open(json_path, "w") as f:
    json.dump(json_data, f, indent=2)

print(f"\n✅ Transcription complete.")
print(f"Markdown saved to: {md_path}")
print(f"JSON saved to: {json_path}")