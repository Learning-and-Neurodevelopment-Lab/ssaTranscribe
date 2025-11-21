#!/usr/bin/env python3
"""
Stage 3: Fact Checking and Confidence Analysis
Standalone script - can be run independently or via orchestrator
"""

import json
import logging
import argparse
import sys
import csv
from pathlib import Path
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_timestamp(seconds):
    """Format seconds to HH:MM:SS.mmm"""
    td = timedelta(seconds=seconds)
    total_ms = int(td.total_seconds() * 1000)
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    milliseconds = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"


def load_pyannote_results(base_dir):
    """Load PyAnnote results from JSON"""
    json_path = base_dir / "transcription.json"
    
    if not json_path.exists():
        return None, 0, []
    
    with open(json_path, 'r') as f:
        segments = json.load(f)
    
    speakers = set()
    speech_segments = []
    
    for seg in segments:
        if seg.get("speaker") and seg["speaker"] not in ["SILENCE", "UNKNOWN"]:
            speakers.add(seg["speaker"])
            speech_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'speaker': seg['speaker'],
                'text': seg.get('text', ''),
                'wpm': seg.get('wpm', 0)
            })
    
    return segments, len(speakers), speech_segments


def load_nemo_results(base_dir):
    """Load NeMo results from JSON or RTTM files"""
    # First try JSON (preferred)
    json_path = base_dir / "nemo_diarization.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['segments'], data['metadata']['num_speakers'], data['segments']
    
    # Fallback to RTTM
    rttm_files = []
    for rttm_file in base_dir.rglob('*.rttm'):
        rttm_files.append(rttm_file)
    
    if not rttm_files:
        return None, 0, []
    
    segments = []
    speakers = set()
    
    with open(rttm_files[0], 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                
                segments.append({
                    'start': start,
                    'end': start + duration,
                    'duration': duration,
                    'speaker': speaker
                })
                speakers.add(speaker)
    
    return segments, len(speakers), segments


def calculate_overlap(seg1, seg2):
    """Calculate overlap between two segments in seconds"""
    overlap_start = max(seg1['start'], seg2['start'])
    overlap_end = min(seg1['end'], seg2['end'])
    return max(0, overlap_end - overlap_start)


def run_fact_check(output_dir: str, output_prefix: str = "fact_check") -> bool:
    """Run fact checking and confidence analysis"""
    
    output_path = Path(output_dir)
    
    csv_output = output_path / f"{output_prefix}_segments.csv"
    summary_output = output_path / f"{output_prefix}_summary.txt"
    
    # Check if already processed
    if csv_output.exists() and summary_output.exists():
        logger.info(f"✓ Already processed: {csv_output}")
        return True
    
    try:
        logger.info(f"Processing: {output_path.name}")
        
        # Load PyAnnote results
        pyannote_all, pyannote_count, pyannote_speech = load_pyannote_results(output_path)
        
        if not pyannote_all:
            logger.error(f"PyAnnote results not found in {output_path}")
            return False
        
        # Load NeMo results
        nemo_all, nemo_count, nemo_speech = load_nemo_results(output_path)
        
        if not nemo_all:
            logger.error(f"NeMo results not found in {output_path}")
            return False
        
        logger.info(f"PyAnnote: {len(pyannote_speech)} segments, {pyannote_count} speakers")
        logger.info(f"NeMo: {len(nemo_speech)} segments, {nemo_count} speakers")
        
        # Single-pass analysis
        results = []
        speaker_changes_pyannote = 0
        speaker_changes_nemo = 0
        boundary_matches = 0
        total_speech_pyannote = 0
        total_speech_nemo = 0
        
        for i, p_seg in enumerate(pyannote_speech):
            # Count speaker changes
            if i > 0 and p_seg['speaker'] != pyannote_speech[i-1]['speaker']:
                speaker_changes_pyannote += 1
            
            duration = p_seg['end'] - p_seg['start']
            total_speech_pyannote += duration
            
            # Find best matching NeMo segment
            best_overlap = 0
            best_nemo_speaker = "NO_MATCH"
            
            for n_seg in nemo_speech:
                overlap = calculate_overlap(p_seg, n_seg)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_nemo_speaker = n_seg['speaker']
            
            # Calculate confidence
            overlap_ratio = best_overlap / duration if duration > 0 else 0
            
            # Check boundary match
            boundary_match = False
            for n_seg in nemo_speech:
                if abs(p_seg['start'] - n_seg['start']) < 0.5:
                    boundary_match = True
                    boundary_matches += 1
                    break
            
            # Confidence scoring
            confidence_score = 0
            if overlap_ratio > 0.8:
                confidence_score += 0.5
            if 0.5 <= duration <= 10.0:
                confidence_score += 0.3
            if boundary_match:
                confidence_score += 0.2
            
            confidence_level = "HIGH" if confidence_score >= 0.7 else "MEDIUM" if confidence_score >= 0.4 else "LOW"
            
            results.append({
                'segment_id': i + 1,
                'start': round(p_seg['start'], 2),
                'end': round(p_seg['end'], 2),
                'duration': round(duration, 2),
                'pyannote_speaker': p_seg['speaker'],
                'nemo_speaker': best_nemo_speaker,
                'overlap_ratio': round(overlap_ratio, 3),
                'boundary_match': boundary_match,
                'confidence_score': round(confidence_score, 3),
                'confidence_level': confidence_level,
                'text': p_seg['text'][:100],  # Truncate for CSV
                'wpm': p_seg.get('wpm', 0)
            })
        
        # Count NeMo speaker changes
        for i in range(1, len(nemo_speech)):
            if nemo_speech[i]['speaker'] != nemo_speech[i-1]['speaker']:
                speaker_changes_nemo += 1
            total_speech_nemo += (nemo_speech[i]['end'] - nemo_speech[i]['start'])
        
        # Calculate metrics
        boundary_agreement = boundary_matches / max(speaker_changes_pyannote, 1)
        speaker_agreement = 1.0 if pyannote_count == nemo_count else \
                           1.0 - abs(pyannote_count - nemo_count) / max(pyannote_count, nemo_count)
        
        high_conf = sum(1 for r in results if r['confidence_level'] == 'HIGH')
        med_conf = sum(1 for r in results if r['confidence_level'] == 'MEDIUM')
        low_conf = sum(1 for r in results if r['confidence_level'] == 'LOW')
        
        # Detect issues
        issues = []
        if pyannote_count == 1 or nemo_count == 1:
            issues.append("Under-segmentation detected (only 1 speaker)")
        if abs(pyannote_count - nemo_count) > 2:
            issues.append(f"Large speaker count mismatch ({pyannote_count} vs {nemo_count})")
        if boundary_agreement < 0.4:
            issues.append(f"Low boundary agreement ({boundary_agreement:.1%})")
        if abs(total_speech_pyannote - total_speech_nemo) > 10:
            issues.append(f"Large timing discrepancy ({abs(total_speech_pyannote - total_speech_nemo):.1f}s)")
        
        # Overall confidence
        overall_confidence = "HIGH" if (speaker_agreement > 0.8 and boundary_agreement > 0.7 and not issues) else \
                            "MEDIUM" if (speaker_agreement > 0.6 and boundary_agreement > 0.4) else "LOW"
        
        # Save CSV
        with open(csv_output, 'w', newline='', encoding='utf-8') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        # Save summary report
        with open(summary_output, 'w') as f:
            f.write("DIARIZATION FACT-CHECK SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("SPEAKER COUNT\n")
            f.write(f"  PyAnnote: {pyannote_count} speakers\n")
            f.write(f"  NeMo: {nemo_count} speakers\n")
            f.write(f"  Agreement: {speaker_agreement:.1%}\n\n")
            
            f.write("BOUNDARY ANALYSIS\n")
            f.write(f"  PyAnnote changes: {speaker_changes_pyannote}\n")
            f.write(f"  NeMo changes: {speaker_changes_nemo}\n")
            f.write(f"  Matched boundaries: {boundary_matches}\n")
            f.write(f"  Agreement: {boundary_agreement:.1%}\n\n")
            
            f.write("CONFIDENCE DISTRIBUTION\n")
            f.write(f"  High: {high_conf} segments ({high_conf/len(results)*100:.1f}%)\n")
            f.write(f"  Medium: {med_conf} segments ({med_conf/len(results)*100:.1f}%)\n")
            f.write(f"  Low: {low_conf} segments ({low_conf/len(results)*100:.1f}%)\n\n")
            
            f.write("TOTAL SPEECH TIME\n")
            f.write(f"  PyAnnote: {total_speech_pyannote:.1f}s\n")
            f.write(f"  NeMo: {total_speech_nemo:.1f}s\n")
            f.write(f"  Difference: {abs(total_speech_pyannote - total_speech_nemo):.1f}s\n\n")
            
            if issues:
                f.write("DETECTED ISSUES\n")
                for issue in issues:
                    f.write(f"  - {issue}\n")
                f.write("\n")
            
            f.write(f"OVERALL CONFIDENCE: {overall_confidence}\n\n")
            
            if overall_confidence == "HIGH":
                f.write("RECOMMENDATION: Results are reliable. Use PyAnnote output.\n")
            elif overall_confidence == "MEDIUM":
                f.write("RECOMMENDATION: Moderate confidence. Review low-confidence segments.\n")
            else:
                f.write("RECOMMENDATION: Low confidence. Manual review needed.\n")
        
        logger.info(f"✓ Complete - Overall Confidence: {overall_confidence}")
        logger.info(f"  Segments CSV: {csv_output}")
        logger.info(f"  Summary: {summary_output}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Stage 3: Fact Checking and Confidence Analysis",
        epilog="""
This script compares PyAnnote and NeMo diarization results to assess confidence.

Input files required in output directory:
  - transcription.json (from Stage 1: WhisperX)
  - nemo_diarization.json or *.rttm (from Stage 2: NeMo)

Output files created:
  - fact_check_segments.csv: Per-segment confidence scores
  - fact_check_summary.txt: Overall analysis and recommendations
        """
    )
    
    parser.add_argument(
        "--output-dir", "-o", 
        required=True, 
        help="Output directory containing transcription.json and nemo_diarization.json"
    )
    parser.add_argument(
        "--prefix",
        default="fact_check",
        help="Output file prefix (default: fact_check)"
    )
    
    args = parser.parse_args()
    
    # Validate directory exists
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        sys.exit(1)
    
    # Check for required input files
    if not (output_dir / "transcription.json").exists():
        logger.error(f"Missing transcription.json in {output_dir}")
        logger.error("Run Stage 1 (WhisperX) first!")
        sys.exit(1)
    
    nemo_json = output_dir / "nemo_diarization.json"
    nemo_rttm = list(output_dir.rglob("*.rttm"))
    
    if not nemo_json.exists() and not nemo_rttm:
        logger.error(f"Missing NeMo results in {output_dir}")
        logger.error("Run Stage 2 (NeMo) first!")
        sys.exit(1)
    
    # Run fact check
    success = run_fact_check(args.output_dir, args.prefix)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
