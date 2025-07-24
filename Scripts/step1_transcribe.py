import sys
import os
import whisperx
import json

audio_path = sys.argv[1]      # audio.wav
output_dir = sys.argv[2]      # output folder
os.makedirs(output_dir, exist_ok=True)

device = "cuda"
model = whisperx.load_model("large-v3", device, compute_type="float16")
transcription = model.transcribe(audio_path)
segments = transcription["segments"]
model_a, metadata = whisperx.load_align_model(language_code=transcription["language"], device=device)
result_aligned = whisperx.align(segments, model_a, metadata, audio_path, device)
with open(os.path.join(output_dir, "aligned_segments.json"), "w") as f:
    json.dump(result_aligned["segments"], f, indent=2)

print("✅ Transcription and alignment complete.")