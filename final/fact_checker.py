import json
import os
import csv
from datetime import timedelta
from collections import defaultdict

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
    json_path = os.path.join(base_dir, "transcription.json")
    
    if not os.path.exists(json_path):
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
    """Load NeMo results from RTTM files"""
    rttm_files = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.rttm'):
                rttm_files.append(os.path.join(root, file))
    
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
                    'speaker': speaker
                })
                speakers.add(speaker)
    
    return segments, len(speakers), segments

def calculate_overlap(seg1, seg2):
    """Calculate overlap between two segments in seconds"""
    overlap_start = max(seg1['start'], seg2['start'])
    overlap_end = min(seg1['end'], seg2['end'])
    return max(0, overlap_end - overlap_start)

def analyze_diarization(base_dir, output_prefix="fact_check"):
    """Main analysis function - efficient single pass"""
    
    print(f"Loading results from: {base_dir}")
    
    # Load both results
    pyannote_all, pyannote_count, pyannote_speech = load_pyannote_results(base_dir)
    nemo_all, nemo_count, nemo_speech = load_nemo_results(base_dir)
    
    if not pyannote_all or not nemo_all:
        print("ERROR: Missing PyAnnote or NeMo results")
        return None
    
    print(f"PyAnnote: {len(pyannote_speech)} segments, {pyannote_count} speakers")
    print(f"NeMo: {len(nemo_speech)} segments, {nemo_count} speakers")
    
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
        
        total_speech_pyannote += (p_seg['end'] - p_seg['start'])
        
        # Find best matching NeMo segment
        best_overlap = 0
        best_nemo_speaker = "NO_MATCH"
        
        for n_seg in nemo_speech:
            overlap = calculate_overlap(p_seg, n_seg)
            if overlap > best_overlap:
                best_overlap = overlap
                best_nemo_speaker = n_seg['speaker']
        
        # Calculate confidence
        duration = p_seg['end'] - p_seg['start']
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
            'start': p_seg['start'],
            'end': p_seg['end'],
            'duration': duration,
            'pyannote_speaker': p_seg['speaker'],
            'nemo_speaker': best_nemo_speaker,
            'overlap_ratio': overlap_ratio,
            'boundary_match': boundary_match,
            'confidence_score': confidence_score,
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
    speaker_agreement = 1.0 if pyannote_count == nemo_count else 1.0 - abs(pyannote_count - nemo_count) / max(pyannote_count, nemo_count)
    
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
    csv_path = os.path.join(base_dir, f"{output_prefix}_segments.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    # Save summary report
    report_path = os.path.join(base_dir, f"{output_prefix}_summary.txt")
    with open(report_path, 'w') as f:
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
    
    print(f"\nResults saved:")
    print(f"  Segments: {csv_path}")
    print(f"  Summary: {report_path}")
    print(f"  Overall confidence: {overall_confidence}")
    
    return {
        'overall_confidence': overall_confidence,
        'speaker_agreement': speaker_agreement,
        'boundary_agreement': boundary_agreement,
        'issues': issues
    }

def main():
    """Main entry point"""
    # Change this to your episode directory
    base_dir = "/Users/landlab/Desktop/ssaTranscription/25-07-14_12.11.00"
    
    if not os.path.exists(base_dir):
        print(f"ERROR: Directory not found: {base_dir}")
        return
    
    analyze_diarization(base_dir)

if __name__ == "__main__":
    main()
