import os
import json
import torch
from nemo.collections.asr.models.clustering_diarizer import ClusteringDiarizer
from omegaconf import OmegaConf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up paths
BASE_DIR = "/Volumes/LANDLAB/projects/Project_Sesame/ssa_sesame-street-archive/scripts/ssa_scaling/3_audio-transcriber"
FOLDER = os.path.join(base_dir, "versions", "25-07-14_12.11.00")
AUDIO_FILE = "audio.wav"
MANIFEST_FILE = "input_manifest.json"

AUDIO_PATH = os.path.join(BASE_DIR, FOLDER, AUDIO_FILE)
OUTPUT_DIR = os.path.join(BASE_DIR, FOLDER)
MANIFEST_PATH = os.path.join(BASE_DIR, FOLDER, MANIFEST_FILE)

# Verify paths exist
if not os.path.exists(AUDIO_PATH):
    raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info(f"Audio file: {AUDIO_PATH}")
logger.info(f"Output directory: {OUTPUT_DIR}")

# Get audio duration
import librosa
try:
    audio_data, sr = librosa.load(AUDIO_PATH, sr=16000)
    duration = len(audio_data) / sr
    logger.info(f"Audio duration: {duration:.2f} seconds")
    logger.info(f"Audio samples: {len(audio_data)}, Sample rate: {sr}")
except Exception as e:
    logger.warning(f"Could not determine audio duration: {e}")
    duration = None

# Create manifest file - WITH num_speakers specified
manifest_entry = {
    "audio_filepath": AUDIO_PATH,
    "offset": 0,
    "duration": duration,
    "label": "infer",
    "text": "-",
    "num_speakers": 3,  # SPECIFY the number of speakers here!
    "rttm_filepath": None,
    "uem_filepath": None,
}

with open(MANIFEST_PATH, 'w') as f:
    f.write(json.dumps(manifest_entry) + '\n')

logger.info(f"Manifest file created: {MANIFEST_PATH}")

# AGGRESSIVE Configuration - Force multiple speakers for Sesame Street
CONFIG = {
    "sample_rate": 16000,
    "batch_size": 1,
    "num_workers": 0,
    "device": "cpu",
    "verbose": True,
    "diarizer": {
        "manifest_filepath": MANIFEST_PATH,
        "out_dir": OUTPUT_DIR,
        "oracle_vad": False,
        "collar": 0.25,
        "ignore_overlap": True,
        "vad": {
            "model_path": "vad_multilingual_marblenet",
            "parameters": {
                "threshold": 0.6,
                "window_length_in_sec": 0.15,
                "shift_length_in_sec": 0.01,
                "smooth_duration": 0.15,
                "smoothing": "median",
                "overlap": 0.5,
                "onset": 0.8,
                "offset": 0.6,
                "min_duration_on": 0.08,
                "min_duration_off": 0.08,
                "filter_speech_first": True
            },
        },
        "speaker_embeddings": {
            "model_path": "titanet_large",
            "parameters": {
                "window_length_in_sec": [1.5, 1.0, 0.5],
                "shift_length_in_sec": [0.75, 0.5, 0.25],
                "multiscale_weights": [1, 1, 1],
                "save_embeddings": False
            },
        },
        "clustering": {
            "parameters": {
                "oracle_num_speakers": 3,  # Force exactly 3 speakers
                "threshold": 0.3,
                "min_num_speakers": 3,
                "max_num_speakers": 8,
                "max_rp_threshold": 0.1,
                "sparse_search_volume": 50,
            },
        },
    }
}

def find_rttm_files(output_dir):
    """Find all RTTM files in output directory and subdirectories"""
    rttm_files = []
    
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.rttm'):
                rttm_files.append(os.path.join(root, file))
    
    return rttm_files

def debug_clustering_results(output_dir):
    """Debug what the clustering algorithm is seeing"""
    
    # Look for clustering output files
    cluster_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if 'cluster' in file.lower() or 'label' in file.lower():
                full_path = os.path.join(root, file)
                cluster_files.append(full_path)
    
    print("\n🔍 CLUSTERING DEBUG INFO:")
    print("-" * 50)
    
    for file_path in cluster_files:
        print(f"File: {os.path.relpath(file_path, output_dir)}")
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                print(f"Content preview: {content[:200]}...")
                
                # Try to count unique labels
                lines = content.strip().split('\n')
                if lines:
                    labels = [line.split()[-1] if line.split() else '' for line in lines]
                    unique_labels = set(label for label in labels if label)
                    print(f"Unique cluster labels found: {sorted(unique_labels)}")
        except Exception as e:
            print(f"Could not read file: {e}")
        print()

def parse_rttm_to_readable(rttm_path):
    """Convert RTTM to human-readable format - optimized for Sesame Street!"""
    
    if not os.path.exists(rttm_path):
        logger.error(f"RTTM file not found: {rttm_path}")
        return
    
    segments = []
    
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                start_time = float(parts[3])
                duration = float(parts[4])
                end_time = start_time + duration
                speaker = parts[7]
                
                segments.append({
                    'speaker': speaker,
                    'start': start_time,
                    'end': end_time,
                    'duration': duration
                })
    
    if not segments:
        logger.warning("No speaker segments found in RTTM file")
        return
    
    # Sort by start time
    segments.sort(key=lambda x: x['start'])
    
    print("\n" + "="*70)
    print("🎭 SESAME STREET CHARACTER DIARIZATION RESULTS (FORCED 3 SPEAKERS)")
    print("="*70)
    
    # Map speaker IDs to more friendly names
    speaker_names = {}
    unique_speakers = sorted(set(seg['speaker'] for seg in segments))
    
    # Generic character names (not actual identification!)
    character_suggestions = ["Speaker_A", "Speaker_B", "Speaker_C", "Speaker_D", "Speaker_E"]
    
    for i, speaker_id in enumerate(unique_speakers):
        if i < len(character_suggestions):
            speaker_names[speaker_id] = character_suggestions[i]
        else:
            speaker_names[speaker_id] = f"Speaker_{i+1}"
    
    for i, seg in enumerate(segments):
        start_min = int(seg['start'] // 60)
        start_sec = seg['start'] % 60
        end_min = int(seg['end'] // 60)
        end_sec = seg['end'] % 60
        
        friendly_name = speaker_names[seg['speaker']]
        print(f"{i+1:2d}. {friendly_name:12s}: {start_min:02d}:{start_sec:05.2f} → {end_min:02d}:{end_sec:05.2f} ({seg['duration']:5.2f}s)")
    
    # Summary
    speakers = set(seg['speaker'] for seg in segments)
    total_speech_time = sum(seg['duration'] for seg in segments)
    
    print("\n" + "-"*70)
    print("📊 SESAME STREET SUMMARY:")
    print(f"   Total segments: {len(segments)}")
    print(f"   Detected speakers: {len(speakers)} (forced to find 3)")
    print(f"   Total speech time: {total_speech_time:.2f}s out of {duration:.2f}s")
    
    # Show speaker speaking time breakdown
    print(f"\n👥 SPEAKER SPEAKING TIME:")
    speaker_times = {}
    for seg in segments:
        if seg['speaker'] not in speaker_times:
            speaker_times[seg['speaker']] = 0
        speaker_times[seg['speaker']] += seg['duration']
    
    for speaker_id, total_time in sorted(speaker_times.items(), key=lambda x: x[1], reverse=True):
        friendly_name = speaker_names[speaker_id]
        percentage = (total_time / total_speech_time) * 100
        print(f"   {friendly_name}: {total_time:.2f}s ({percentage:.1f}%)")
    
    print("\n💡 NOTE: Speaker names are generic labels, not actual character identification!")
    print("="*70)

def show_all_output_files(output_dir):
    """Show all files created in the output directory"""
    print("\n" + "="*60)
    print("📁 ALL OUTPUT FILES:")
    print("="*60)
    
    all_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, output_dir)
            file_size = os.path.getsize(full_path)
            all_files.append((rel_path, file_size))
    
    # Sort by path
    all_files.sort()
    
    for rel_path, file_size in all_files:
        print(f"   {rel_path} ({file_size} bytes)")
    
    print("="*60)

def main():
    try:
        # Convert config dict to OmegaConf
        cfg = OmegaConf.create(CONFIG)
        
        logger.info("🎭 Initializing FORCED 3-Speaker Diarizer...")
        logger.info("🎯 Forcing detection of exactly 3 speakers!")
        # Use ClusteringDiarizer instead of NeuralDiarizer
        model = ClusteringDiarizer(cfg=cfg)
        
        logger.info("Starting speaker diarization...")
        # Run diarization pipeline
        model.diarize()
        
        logger.info("✅ Speaker diarization complete!")
        logger.info(f"Output written to: {OUTPUT_DIR}")
        
        # Show all output files
        show_all_output_files(OUTPUT_DIR)
        
        # Debug clustering results
        debug_clustering_results(OUTPUT_DIR)
        
        # Find and parse RTTM files
        rttm_files = find_rttm_files(OUTPUT_DIR)
        
        if rttm_files:
            logger.info(f"Found {len(rttm_files)} RTTM file(s)")
            
            for rttm_file in rttm_files:
                logger.info(f"Parsing: {rttm_file}")
                parse_rttm_to_readable(rttm_file)
        else:
            logger.warning("❌ No RTTM files found!")
            print("\nTip: Check the 'pred_rttms' or 'speaker_outputs' subdirectories")
            
    except Exception as e:
        logger.error(f"Error during diarization: {str(e)}")
        raise

if __name__ == "__main__":
    main()
