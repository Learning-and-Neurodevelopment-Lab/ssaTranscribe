import os
import json
import logging
from nemo.collections.asr.models.clustering_diarizer import ClusteringDiarizer
from omegaconf import OmegaConf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
BASE_DIR = "/Volumes/LANDLAB/projects/Project_Sesame/ssa_sesame-street-archive/scripts/ssa_scaling/3_audio-transcriber"
FOLDER = os.path.join(BASE_DIR, "versions", "25-07-14_12.11.00")
AUDIO_FILE = "audio.wav"
MANIFEST_FILE = "input_manifest.json"

AUDIO_PATH = os.path.join(FOLDER, AUDIO_FILE)
OUTPUT_DIR = FOLDER
MANIFEST_PATH = os.path.join(FOLDER, MANIFEST_FILE)

# === VALIDATE PATHS ===
if not os.path.exists(AUDIO_PATH):
    raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === GET AUDIO DURATION (EFFICIENT) ===
import librosa
try:
    duration = librosa.get_duration(path=AUDIO_PATH)
    logger.info(f"Audio duration: {duration:.2f}s")
except Exception as e:
    logger.warning(f"Could not determine duration: {e}")
    duration = None

# === CREATE MANIFEST ===
manifest_entry = {
    "audio_filepath": AUDIO_PATH,
    "offset": 0,
    "duration": duration,
    "label": "infer",
    "text": "-",
    "rttm_filepath": None,
    "uem_filepath": None,
}

with open(MANIFEST_PATH, 'w') as f:
    json.dump(manifest_entry, f)
    f.write('\n')

# === OPTIMIZED CONFIGURATION ===
CONFIG = {
    "sample_rate": 16000,
    "batch_size": 1,
    "num_workers": 0,
    "device": "cpu",
    "verbose": False,  # Reduce console spam
    "diarizer": {
        "manifest_filepath": MANIFEST_PATH,
        "out_dir": OUTPUT_DIR,
        "oracle_vad": False,
        "collar": 0.25,
        "ignore_overlap": True,
        
        # Voice Activity Detection - optimized for Sesame Street
        "vad": {
            "model_path": "vad_multilingual_marblenet",
            "parameters": {
                "threshold": 0.55,  # Slightly more sensitive
                "window_length_in_sec": 0.15,
                "shift_length_in_sec": 0.01,
                "smooth_duration": 0.15,
                "smoothing": "median",
                "overlap": 0.5,
                "onset": 0.8,
                "offset": 0.6,
                "min_duration_on": 0.1,  # Detect shorter utterances
                "min_duration_off": 0.1,  # Detect shorter pauses
                "filter_speech_first": True
            },
        },
        
        # Speaker Embeddings - shorter windows for quick speaker changes
        "speaker_embeddings": {
            "model_path": "titanet_large",
            "parameters": {
                "window_length_in_sec": [1.5, 1.0, 0.5],
                "shift_length_in_sec": [0.75, 0.5, 0.25],
                "multiscale_weights": [1, 1, 1],
                "save_embeddings": False  # Don't save unnecessary files
            },
        },
        
        # Clustering - prevent under-segmentation
        "clustering": {
            "parameters": {
                "oracle_num_speakers": None,  # Auto-detect
                "min_num_speakers": 1,  # Allow single-speaker scenes
                "max_num_speakers": 8,  # Reasonable upper bound
                "threshold": 0.27,  # Lower = more aggressive splitting
                "enhanced_count_thresholding": True,  # Better auto-detection
                "sparse_search_volume": 50,  # Explore more clustering options
                "max_rp_threshold": 0.1,
            },
        },
    }
}

def parse_rttm_file(output_dir):
    """Parse RTTM file and return segments"""
    # Find RTTM file efficiently
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.rttm'):
                rttm_path = os.path.join(root, file)
                
                segments = []
                speakers = set()
                
                with open(rttm_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 8 and parts[0] == 'SPEAKER':
                            start_time = float(parts[3])
                            duration = float(parts[4])
                            speaker = parts[7]
                            
                            segments.append({
                                'start': start_time,
                                'end': start_time + duration,
                                'duration': duration,
                                'speaker': speaker
                            })
                            speakers.add(speaker)
                
                return rttm_path, segments, speakers
    
    return None, [], set()

def save_readable_output(segments, speakers, output_dir):
    """Save human-readable diarization results"""
    output_path = os.path.join(output_dir, "nemo_diarization.txt")
    
    with open(output_path, 'w') as f:
        f.write("NEMO SPEAKER DIARIZATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Detected Speakers: {len(speakers)}\n")
        f.write(f"Total Segments: {len(segments)}\n")
        
        if segments:
            total_speech = sum(seg['duration'] for seg in segments)
            f.write(f"Total Speech Time: {total_speech:.2f}s\n\n")
            
            # Speaker breakdown
            f.write("Speaker Breakdown:\n")
            speaker_times = {}
            for seg in segments:
                speaker_times[seg['speaker']] = speaker_times.get(seg['speaker'], 0) + seg['duration']
            
            for speaker, time in sorted(speaker_times.items(), key=lambda x: x[1], reverse=True):
                percentage = (time / total_speech) * 100
                f.write(f"  {speaker}: {time:.2f}s ({percentage:.1f}%)\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
            
            # Segment timeline
            f.write("Timeline:\n")
            for i, seg in enumerate(segments, 1):
                start_min = int(seg['start'] // 60)
                start_sec = seg['start'] % 60
                end_min = int(seg['end'] // 60)
                end_sec = seg['end'] % 60
                
                f.write(f"{i:3d}. {seg['speaker']:12s} "
                       f"{start_min:02d}:{start_sec:05.2f} -> "
                       f"{end_min:02d}:{end_sec:05.2f} "
                       f"({seg['duration']:5.2f}s)\n")
    
    return output_path

def save_json_output(segments, speakers, output_dir):
    """Save JSON output for programmatic use"""
    output_path = os.path.join(output_dir, "nemo_diarization.json")
    
    data = {
        'metadata': {
            'num_speakers': len(speakers),
            'total_segments': len(segments),
            'speakers': sorted(list(speakers))
        },
        'segments': segments
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_path

def main():
    try:
        logger.info("Starting NeMo speaker diarization...")
        
        # Convert config to OmegaConf
        cfg = OmegaConf.create(CONFIG)
        
        # Initialize and run diarizer
        model = ClusteringDiarizer(cfg=cfg)
        model.diarize()
        
        # Parse results
        rttm_path, segments, speakers = parse_rttm_file(OUTPUT_DIR)
        
        if not segments:
            logger.error("No diarization results found!")
            return
        
        logger.info(f"Diarization complete: {len(speakers)} speakers, {len(segments)} segments")
        
        # Check for potential under-segmentation
        if len(speakers) == 1 and duration and duration > 60:
            logger.warning("Only 1 speaker detected in long audio - possible under-segmentation")
            logger.warning("Consider lowering clustering threshold if this seems incorrect")
        
        # Save readable outputs
        txt_path = save_readable_output(segments, speakers, OUTPUT_DIR)
        json_path = save_json_output(segments, speakers, OUTPUT_DIR)
        
        logger.info(f"Results saved:")
        logger.info(f"  RTTM: {rttm_path}")
        logger.info(f"  Readable: {txt_path}")
        logger.info(f"  JSON: {json_path}")
        
    except Exception as e:
        logger.error(f"Error during diarization: {e}")
        raise

if __name__ == "__main__":
    main()
