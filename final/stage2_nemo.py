#!/usr/bin/env python3
"""
Stage 2: NeMo Speaker Diarization
Standalone script - can be run independently or via orchestrator
"""

import os
import json
import logging
import argparse
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_nemo_diarization(audio_file: str, output_dir: str) -> bool:
    """Run NeMo speaker diarization"""
    
    audio_path = Path(audio_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_output = output_path / "nemo_diarization.json"
    txt_output = output_path / "nemo_diarization.txt"
    
    # Check if already processed
    if json_output.exists():
        logger.info(f"✓ Already processed: {json_output}")
        return True
    
    try:
        from nemo.collections.asr.models.clustering_diarizer import ClusteringDiarizer
        from omegaconf import OmegaConf
        import librosa
        
        logger.info(f"Processing: {audio_path.name}")
        
        # Get audio duration
        try:
            duration = librosa.get_duration(path=str(audio_path))
            logger.info(f"Audio duration: {duration:.2f}s")
        except Exception as e:
            logger.warning(f"Could not determine duration: {e}")
            duration = None
        
        # Create manifest
        manifest_path = output_path / "input_manifest.json"
        manifest_entry = {
            "audio_filepath": str(audio_path.absolute()),
            "offset": 0,
            "duration": duration,
            "label": "infer",
            "text": "-",
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_entry, f)
            f.write('\n')
        
        # NeMo configuration
        config = {
            "sample_rate": 16000,
            "batch_size": 1,
            "num_workers": 0,
            "device": "cpu",
            "verbose": False,
            "diarizer": {
                "manifest_filepath": str(manifest_path),
                "out_dir": str(output_path),
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": True,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "parameters": {
                        "threshold": 0.55,
                        "window_length_in_sec": 0.15,
                        "shift_length_in_sec": 0.01,
                        "smooth_duration": 0.15,
                        "smoothing": "median",
                        "overlap": 0.5,
                        "onset": 0.8,
                        "offset": 0.6,
                        "min_duration_on": 0.1,
                        "min_duration_off": 0.1,
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
                        "oracle_num_speakers": None,
                        "min_num_speakers": 1,
                        "max_num_speakers": 8,
                        "threshold": 0.27,
                        "enhanced_count_thresholding": True,
                        "sparse_search_volume": 50,
                        "max_rp_threshold": 0.1,
                    },
                },
            }
        }
        
        logger.info("Running NeMo diarization...")
        cfg = OmegaConf.create(config)
        model = ClusteringDiarizer(cfg=cfg)
        model.diarize()
        
        # Clean up model
        del model
        
        # Parse RTTM results
        rttm_path = None
        segments = []
        speakers = set()
        
        for root, dirs, files in os.walk(output_path):
            for file in files:
                if file.endswith('.rttm'):
                    rttm_path = Path(root) / file
                    
                    with open(rttm_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                                start = float(parts[3])
                                dur = float(parts[4])
                                speaker = parts[7]
                                
                                segments.append({
                                    'start': start,
                                    'end': start + dur,
                                    'duration': dur,
                                    'speaker': speaker
                                })
                                speakers.add(speaker)
                    break
            if rttm_path:
                break
        
        if not segments:
            logger.error("No diarization results found!")
            return False
        
        logger.info(f"Found {len(speakers)} speakers, {len(segments)} segments")
        
        # Warning for potential issues
        if len(speakers) == 1 and duration and duration > 60:
            logger.warning("⚠ Only 1 speaker detected in long audio - possible under-segmentation")
        
        # Save JSON
        data = {
            'metadata': {
                'num_speakers': len(speakers),
                'total_segments': len(segments),
                'speakers': sorted(list(speakers))
            },
            'segments': segments
        }
        with open(json_output, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save TXT
        with open(txt_output, 'w') as f:
            f.write("NEMO SPEAKER DIARIZATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Detected Speakers: {len(speakers)}\n")
            f.write(f"Total Segments: {len(segments)}\n")
            
            if segments:
                total_speech = sum(seg['duration'] for seg in segments)
                f.write(f"Total Speech Time: {total_speech:.2f}s\n\n")
                
                speaker_times = {}
                for seg in segments:
                    speaker_times[seg['speaker']] = speaker_times.get(seg['speaker'], 0) + seg['duration']
                
                f.write("Speaker Breakdown:\n")
                for speaker, time in sorted(speaker_times.items(), key=lambda x: x[1], reverse=True):
                    percentage = (time / total_speech) * 100
                    f.write(f"  {speaker}: {time:.2f}s ({percentage:.1f}%)\n")
        
        logger.info(f"✓ Complete")
        logger.info(f"  RTTM: {rttm_path}")
        logger.info(f"  JSON: {json_output}")
        logger.info(f"  TXT: {txt_output}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Stage 2: NeMo Diarization")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    success = run_nemo_diarization(args.audio_file, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
