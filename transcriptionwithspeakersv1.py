import os
import json
import datetime
import subprocess
from pyannote.audio import Pipeline
from pyannote.core import Segment
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import whisperx

# === Set Paths ===
base_dir = "/Users/landlab/Desktop/ssaTranscription"
output_dir = os.path.join(base_dir, "25-07-09_10.16.00")
os.makedirs(output_dir, exist_ok=True)
audio_path = os.path.join(output_dir, "audio.wav")
video_path = os.path.join(output_dir, "EP70-cpb-aacip-c0e028b4a94-New.mp4")  # Optional for scene detection

# === Extract audio if not present ===
if not os.path.exists(audio_path):
    if os.path.exists(video_path):
        print(f"Extracting audio from {video_path} to {audio_path}...")
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)
    else:
        raise FileNotFoundError(f"Audio file not found at {audio_path} and video file not found at {video_path}")
else:
    print(f"Audio already exists at {audio_path}")

# === Load ASR Model ===
device = "cpu"  # or "cuda" if available
compute_type = "float32"  # ensure compatibility
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
next_speaker_id = 1  # start from Speaker 1

for seg in result_aligned["segments"]:
    seg_start, seg_end = seg["start"], seg["end"]
    seg_interval = Segment(seg_start, seg_end)
    dia_crop = diarization.crop(seg_interval, mode="loose")  # more inclusive
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

# === Merge segments by speaker (0.8s max gap) ===
merged_segments = []
current = None
max_gap = 0.8

for seg in result_aligned["segments"]:
    if current is None:
        current = seg
    elif (
        seg["speaker"] == current["speaker"]
        and seg["start"] - current["end"] <= max_gap
    ):
        current["end"] = seg["end"]
        current["text"] += " " + seg["text"]
    else:
        merged_segments.append(current)
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

# === Format timestamp ===
def format_timestamp(seconds):
    td = datetime.timedelta(seconds=seconds)
    return str(td)[:-3]

# === Write to TXT file ===
txt_path = os.path.join(output_dir, "transcription.txt")
with open(txt_path, "w") as f:
    current_scene = None
    for seg in merged_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        speaker = seg["speaker"]
        text = seg["text"].strip()

        for i, (scene_start, scene_end) in enumerate(scene_boundaries):
            if scene_start <= seg_start < scene_end:
                if current_scene != i:
                    f.write(f"\n--- Scene {i+1}: [{format_timestamp(scene_start)} - {format_timestamp(scene_end)}] ---\n\n")
                    current_scene = i
                break

        f.write(f"[{format_timestamp(seg_start)} - {format_timestamp(seg_end)}] {speaker}:\n{text}\n\n")

# === Save JSON output ===
json_path = os.path.join(output_dir, "transcription.json")
with open(json_path, "w") as f:
    json.dump(merged_segments, f, indent=2)

print(f"\n✅ Transcription complete.")
print(f"TXT saved to: {txt_path}")
print(f"JSON saved to: {json_path}")
