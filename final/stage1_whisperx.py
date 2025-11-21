#!/usr/bin/env python3
"""
Stage 1: WhisperX Transcription + PyAnnote Diarization
TorchCodec tensor input (no disk audio) + full original functionality
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def decode_audio_tensor(input_path: Path, device: str = "cpu"):
    """
    Decode audio with TorchCodec and return tensor [T] in float32
    and the sample rate. Mono channels only.
    """
    try:
        from torchcodec.decoders import AudioDecoder
        decoder = AudioDecoder(str(input_path), device=device)
        waveform = decoder.to_tensor()  # [C, T]
        if waveform.shape[0] > 1:  # convert to mono if needed
            import torch
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0), decoder.sample_rate
    except ImportError:
        logger.error("TorchCodec not installed. Install with: pip install torchcodec")
        return None, None
    except Exception as e:
        logger.error(f"Audio decoding failed: {e}")
        return None, None


def detect_scenes(video_path: Path):
    """Detect scene changes using scenedetect"""
    if not video_path.exists():
        logger.warning(f"Video file not found for scene detection: {video_path}")
        return []
    try:
        from scenedetect import VideoManager, SceneManager
        from scenedetect.detectors import ContentDetector
        logger.info("Running scene detection...")
        video_manager = VideoManager([str(video_path)])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]
    except ImportError:
        logger.warning("scenedetect not installed, skipping scene detection")
        return []
    except Exception as e:
        logger.warning(f"Scene detection failed: {e}")
        return []


def run_whisperx_transcription(input_file: str, output_dir: str, hf_token: Optional[str] = None) -> bool:
    """Run WhisperX transcription + PyAnnote diarization using TorchCodec tensors directly"""
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_output = output_path / "transcription.json"
    md_output = output_path / "transcription.md"

    if json_output.exists():
        logger.info(f"✓ Already processed: {json_output}")
        return True

    try:
        import whisperx
        from pyannote.audio import Pipeline
        from pyannote.core import Segment
        import datetime
        import torch

        # Determine file type
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        is_video = input_path.suffix.lower() in video_extensions
        is_audio = input_path.suffix.lower() in audio_extensions

        if not (is_video or is_audio):
            logger.error(f"Unsupported file format: {input_path.suffix}")
            return False

        device = "cpu"
        compute_type = "float32"

        # Decode audio directly as tensor
        waveform, sample_rate = decode_audio_tensor(input_path, device=device)
        if waveform is None:
            return False

        # Scene detection for video
        scene_boundaries = detect_scenes(input_path) if is_video else []

        # WhisperX transcription
        logger.info(f"Processing: {input_path.name}")
        logger.info("Loading WhisperX model (large-v3)...")
        model = whisperx.load_model("large-v3", device=device, compute_type=compute_type)

        logger.info("Starting transcription from tensor...")
        # Directly feed waveform tensor to WhisperX
        transcription = model.transcribe_tensor(waveform, sample_rate)
        segments = transcription["segments"]
        logger.info(f"Initial transcription: {len(segments)} segments")

        # Align segments
        logger.info("Aligning segments...")
        model_a, metadata = whisperx.load_align_model(language_code=transcription["language"], device=device)
        result_aligned = whisperx.align(segments, model_a, metadata, waveform=waveform, sample_rate=sample_rate, device=device)
        del model, model_a

        # PyAnnote diarization
        logger.info("Loading PyAnnote diarization model...")
        auth_token = hf_token or os.environ.get('HF_TOKEN')
        if not auth_token:
            logger.error("HuggingFace token required!")
            return False
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=auth_token)
        diarization = diarization_pipeline(str(input_path))

        # Assign speakers
        speaker_mapping = {}
        next_speaker_id = 1
        for seg in result_aligned["segments"]:
            seg_interval = Segment(seg["start"], seg["end"])
            dia_crop = diarization.crop(seg_interval, mode="loose")
            speakers = dia_crop.labels()
            if not speakers:
                seg["speaker"] = "UNKNOWN"
            elif len(speakers) == 1:
                speaker = list(speakers)[0]
            else:
                speaker = max(speakers, key=lambda s: dia_crop.label_duration(s))
            if speaker != "UNKNOWN" and speaker not in speaker_mapping:
                speaker_mapping[speaker] = f"SPEAKER {next_speaker_id}"
                next_speaker_id += 1
            seg["speaker"] = speaker_mapping.get(speaker, "UNKNOWN")
        del diarization_pipeline, diarization

        # Merge segments & insert silences
        merged_segments = []
        current = None
        max_gap = 0.8
        for seg in result_aligned["segments"]:
            if current is None:
                current = seg.copy()
            elif seg["speaker"] == current["speaker"] and seg["start"] - current["end"] <= max_gap:
                current["end"] = seg["end"]
                current["text"] += " " + seg["text"]
                if "words" in current and "words" in seg:
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
                current = seg.copy()
        if current:
            merged_segments.append(current)

        # NER processing
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("Spacy model not found, skipping NER")
            nlp = None

        sesame_characters = {"elmo", "big bird", "cookie monster", "bert", "ernie",
                             "abby", "grover", "oscar", "count", "zoe", "rosita",
                             "snuffy", "telly", "baby bear", "prairie dawn", "herry"}
        sesame_places = {"123 sesame street", "sesame street", "hooper's store",
                         "oscar's can", "elmo's world", "big bird's nest"}

        def is_letter_of_day(text): return "letter of the day" in text.lower()
        def is_number_of_day(text): return "number of the day" in text.lower()

        for seg in merged_segments:
            duration = seg["end"] - seg["start"]
            seg["wpm"] = round(len(seg["text"].split()) / (duration / 60), 2) if duration > 0 else 0
            if nlp and seg["speaker"] not in ["SILENCE", "UNKNOWN"]:
                doc = nlp(seg["text"])
                ner_labels = {ent.text.lower() for ent in doc.ents}
                mentions = set()
                mentions.update(s.title() for s in sesame_characters.intersection(ner_labels))
                text_lower = seg["text"].lower()
                for place in sesame_places:
                    if place in text_lower:
                        mentions.add(place.title())
                if is_letter_of_day(seg["text"]):
                    mentions.add("Letter Of The Day")
                if is_number_of_day(seg["text"]):
                    mentions.add("Number Of The Day")
                if mentions:
                    seg["ner_labels"] = list(mentions)

        # Save JSON
        with open(json_output, 'w') as f:
            json.dump(merged_segments, f, indent=2)

        # Save Markdown
        def format_timestamp(seconds):
            td = datetime.timedelta(seconds=seconds)
            total_ms = int(td.total_seconds() * 1000)
            hours = total_ms // 3600000
            minutes = (total_ms % 3600000) // 60000
            secs = (total_ms % 60000) // 1000
            milliseconds = total_ms % 1000
            return f"{hours:02}:{minutes:02}:{secs:02}:{milliseconds:03}"

        scene_markers = []
        for i, (start, end) in enumerate(scene_boundaries):
            scene_markers.append((start, f"⟵ **Scene {i+1} Starts [{format_timestamp(start)}]**"))
            scene_markers.append((end, f"⟶ **Scene {i+1} Ends [{format_timestamp(end)}]**"))
        scene_markers.sort()

        with open(md_output, 'w') as f:
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
                    if seg.get("ner_labels"):
                        f.write(f"(NER Labels: {', '.join(sorted(seg['ner_labels']))})\n")
                f.write("\n\n")

        unique_speakers = set(s["speaker"] for s in merged_segments if s["speaker"] not in ["SILENCE", "UNKNOWN"])
        logger.info(f"✓ Complete: {len(unique_speakers)} speakers, {len(merged_segments)} segments")
        logger.info(f"  JSON: {json_output}")
        logger.info(f"  Markdown: {md_output}")

        return True

    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: WhisperX Transcription + PyAnnote Diarization with TorchCodec tensor input",
        epilog="""
This script handles both video and audio files:
- Video files (.mp4, .mkv, etc.): Audio is decoded automatically via TorchCodec
- Audio files (.wav, .mp3, etc.): Decoded via TorchCodec

Outputs:
- transcription.json: Full transcription with speakers, timestamps, NER
- transcription.md: Human-readable markdown format
"""
    )
    parser.add_argument("input_file", help="Path to video or audio file")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    parser.add_argument("--hf-token", help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    success = run_whisperx_transcription(args.input_file, args.output_dir, args.hf_token)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
