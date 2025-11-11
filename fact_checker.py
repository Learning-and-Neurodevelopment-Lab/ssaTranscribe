import json
import os
from datetime import timedelta
import numpy as np

def format_timestamp(seconds):
    """Format seconds to HH:MM:SS:mmm"""
    td = timedelta(seconds=seconds)
    total_ms = int(td.total_seconds() * 1000)
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    seconds = (total_ms % 60000) // 1000
    milliseconds = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}"

def analyze_pyannote_results(base_dir):
    """Load and analyze PyAnnote results from JSON"""
    json_path = os.path.join(base_dir, "transcription.json")
    
    if not os.path.exists(json_path):
        print(f"❌ PyAnnote results not found: {json_path}")
        return None, 0, []
    
    with open(json_path, 'r') as f:
        segments = json.load(f)
    
    # Extract speaker info
    speakers = set()
    speech_segments = []
    
    for seg in segments:
        if seg.get("speaker") and seg["speaker"] not in ["SILENCE", "UNKNOWN"]:
            speakers.add(seg["speaker"])
            speech_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'duration': seg['end'] - seg['start'],
                'speaker': seg['speaker'],
                'text': seg.get('text', ''),
                'method': 'pyannote'
            })
    
    return segments, len(speakers), speech_segments

def analyze_nemo_results(base_dir):
    """Load and analyze NeMo results from RTTM files"""
    rttm_files = []
    
    # Look for RTTM files in all subdirectories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.rttm'):
                rttm_files.append(os.path.join(root, file))
    
    if not rttm_files:
        print(f"❌ NeMo results not found in: {base_dir}")
        return None, 0, []
    
    # Parse RTTM file
    nemo_segments = []
    speakers = set()
    
    print(f"📂 Found NeMo RTTM file: {rttm_files[0]}")
    
    with open(rttm_files[0], 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                start_time = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                
                nemo_segments.append({
                    'start': start_time,
                    'end': start_time + duration,
                    'duration': duration,
                    'speaker': speaker,
                    'method': 'nemo'
                })
                speakers.add(speaker)
    
    return nemo_segments, len(speakers), nemo_segments

def validate_speaker_count(pyannote_count, nemo_count):
    """Cross-validate speaker counts between methods"""
    print(f"\n🔢 SPEAKER COUNT VALIDATION:")
    print(f"   PyAnnote detected: {pyannote_count} speakers")
    print(f"   NeMo detected: {nemo_count} speakers")
    
    if pyannote_count == nemo_count:
        print("   ✅ Perfect agreement on speaker count!")
        return "high", "consensus"
    
    elif pyannote_count == 1 and nemo_count > 1:
        print("   ⚠️  PyAnnote under-segmenting (only 1 speaker)")
        print("   💡 Recommendation: Trust NeMo's detection")
        return "medium", "trust_nemo"
    
    elif nemo_count == 1 and pyannote_count > 1:
        print("   ⚠️  NeMo under-segmenting (only 1 speaker)")
        print("   💡 Recommendation: Trust PyAnnote's detection")
        return "medium", "trust_pyannote"
    
    elif abs(pyannote_count - nemo_count) == 1:
        print("   ⚠️  Methods differ by 1 speaker (acceptable)")
        print("   💡 Recommendation: Use the higher count")
        return "medium", "use_higher"
    
    else:
        print(f"   🚨 Large disagreement! Difference: {abs(pyannote_count - nemo_count)}")
        print("   💡 Recommendation: Manual review needed")
        return "low", "manual_review"

def cross_check_boundaries(pyannote_segments, nemo_segments, tolerance=0.5):
    """Find where both methods agree on speaker changes"""
    print(f"\n🎯 BOUNDARY CROSS-CHECK (±{tolerance}s tolerance):")
    
    # Extract speaker change points
    pyannote_changes = []
    for i in range(1, len(pyannote_segments)):
        if (pyannote_segments[i]['speaker'] != pyannote_segments[i-1]['speaker'] and 
            pyannote_segments[i]['speaker'] != 'SILENCE'):
            pyannote_changes.append(pyannote_segments[i]['start'])
    
    nemo_changes = []
    for i in range(1, len(nemo_segments)):
        if nemo_segments[i]['speaker'] != nemo_segments[i-1]['speaker']:
            nemo_changes.append(nemo_segments[i]['start'])
    
    # Find agreements
    agreements = []
    for p_change in pyannote_changes:
        for n_change in nemo_changes:
            if abs(p_change - n_change) <= tolerance:
                agreements.append({
                    'pyannote_time': p_change,
                    'nemo_time': n_change,
                    'difference': abs(p_change - n_change),
                    'timestamp': format_timestamp(p_change)
                })
                break
    
    # Calculate confidence
    total_changes = max(len(pyannote_changes), len(nemo_changes), 1)
    confidence = len(agreements) / total_changes
    
    print(f"   PyAnnote speaker changes: {len(pyannote_changes)}")
    print(f"   NeMo speaker changes: {len(nemo_changes)}")
    print(f"   Agreed boundaries: {len(agreements)}")
    print(f"   Agreement confidence: {confidence:.1%}")
    
    if confidence >= 0.7:
        print("   ✅ High boundary agreement!")
    elif confidence >= 0.4:
        print("   ⚠️  Moderate boundary agreement")
    else:
        print("   🚨 Low boundary agreement!")
    
    # Show top agreements
    if agreements:
        print(f"   📍 Top boundary agreements:")
        for i, agreement in enumerate(sorted(agreements, key=lambda x: x['difference'])[:5]):
            print(f"      {i+1}. {agreement['timestamp']} (±{agreement['difference']:.2f}s)")
    
    return agreements, confidence

def detect_red_flags(pyannote_segments, nemo_segments, pyannote_count, nemo_count):
    """Detect potential issues in diarization"""
    print(f"\n🚩 RED FLAGS DETECTION:")
    
    red_flags = []
    warnings = []
    
    # Flag 1: Only 1 speaker (unlikely for Sesame Street)
    if pyannote_count == 1:
        red_flags.append("PyAnnote detected only 1 speaker (under-segmentation)")
    if nemo_count == 1:
        red_flags.append("NeMo detected only 1 speaker (under-segmentation)")
    
    # Flag 2: Too many speakers
    if pyannote_count > 8:
        red_flags.append(f"PyAnnote detected {pyannote_count} speakers (possible over-segmentation)")
    if nemo_count > 8:
        red_flags.append(f"NeMo detected {nemo_count} speakers (possible over-segmentation)")
    
    # Flag 3: Very short segments (< 0.3s)
    short_pyannote = [seg for seg in pyannote_segments 
                     if seg.get('speaker') != 'SILENCE' and 
                        (seg.get('end', 0) - seg.get('start', 0)) < 0.3]
    
    if len(short_pyannote) > len(pyannote_segments) * 0.3:
        warnings.append(f"PyAnnote: {len(short_pyannote)} very short segments (< 0.3s)")
    
    short_nemo = [seg for seg in nemo_segments if seg['duration'] < 0.3]
    if len(short_nemo) > len(nemo_segments) * 0.3:
        warnings.append(f"NeMo: {len(short_nemo)} very short segments (< 0.3s)")
    
    # Flag 4: Huge timing discrepancies
    pyannote_total = sum(seg.get('end', 0) - seg.get('start', 0) 
                        for seg in pyannote_segments 
                        if seg.get('speaker') != 'SILENCE')
    nemo_total = sum(seg['duration'] for seg in nemo_segments)
    
    if abs(pyannote_total - nemo_total) > 10:
        red_flags.append(f"Large timing difference: PyAnnote={pyannote_total:.1f}s, NeMo={nemo_total:.1f}s")
    
    # Report findings
    if red_flags:
        print("   🚨 CRITICAL ISSUES:")
        for flag in red_flags:
            print(f"      • {flag}")
    
    if warnings:
        print("   ⚠️  WARNINGS:")
        for warning in warnings:
            print(f"      • {warning}")
    
    if not red_flags and not warnings:
        print("   ✅ No major issues detected!")
    
    return red_flags, warnings

def create_confidence_scores(pyannote_segments, nemo_segments, agreements):
    """Score each segment based on cross-method agreement"""
    print(f"\n📊 CONFIDENCE SCORING:")
    
    scored_segments = []
    
    for seg in pyannote_segments:
        if seg.get('speaker') == 'SILENCE':
            continue
            
        start, end = seg.get('start', 0), seg.get('end', 0)
        confidence_factors = []
        
        # Factor 1: Boundary agreement
        boundary_agreement = False
        for agreement in agreements:
            if abs(start - agreement['pyannote_time']) < 1.0:
                boundary_agreement = True
                break
        
        if boundary_agreement:
            confidence_factors.append("boundary_match")
        
        # Factor 2: Segment duration (reasonable length)
        duration = end - start
        if 0.5 <= duration <= 10.0:
            confidence_factors.append("good_duration")
        
        # Factor 3: Find overlapping NeMo segments
        overlapping_nemo = []
        for n_seg in nemo_segments:
            overlap = min(end, n_seg['end']) - max(start, n_seg['start'])
            if overlap > 0.1:  # At least 0.1s overlap
                overlapping_nemo.append((n_seg, overlap))
        
        if len(overlapping_nemo) == 1:
            confidence_factors.append("single_nemo_match")
        elif len(overlapping_nemo) > 1:
            confidence_factors.append("multiple_nemo_overlap")
        
        # Calculate confidence score
        confidence_score = len(confidence_factors) / 3.0  # Max 3 factors
        
        if confidence_score >= 0.8:
            confidence_level = "HIGH"
        elif confidence_score >= 0.5:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        scored_segments.append({
            'start': start,
            'end': end,
            'speaker': seg.get('speaker'),
            'text': seg.get('text', ''),
            'confidence_score': confidence_score,
            'confidence_level': confidence_level,
            'factors': confidence_factors,
            'overlapping_nemo': len(overlapping_nemo)
        })
    
    # Show confidence summary
    high_conf = len([s for s in scored_segments if s['confidence_level'] == 'HIGH'])
    med_conf = len([s for s in scored_segments if s['confidence_level'] == 'MEDIUM'])
    low_conf = len([s for s in scored_segments if s['confidence_level'] == 'LOW'])
    
    print(f"   High confidence segments: {high_conf}")
    print(f"   Medium confidence segments: {med_conf}")
    print(f"   Low confidence segments: {low_conf}")
    
    return scored_segments

def generate_recommendations(speaker_confidence, boundary_confidence, red_flags, pyannote_count, nemo_count):
    """Generate specific recommendations based on analysis"""
    print(f"\n💡 SMART RECOMMENDATIONS:")
    
    recommendations = []
    
    # Speaker count recommendations
    if pyannote_count == 1 and nemo_count > 1:
        recommendations.append({
            'priority': 'HIGH',
            'issue': 'PyAnnote under-segmentation',
            'action': 'Re-run PyAnnote with more aggressive settings',
            'settings': 'min_speakers=2, max_speakers=6'
        })
    
    elif nemo_count == 1 and pyannote_count > 1:
        recommendations.append({
            'priority': 'HIGH', 
            'issue': 'NeMo under-segmentation',
            'action': 'Re-run NeMo with lower clustering threshold',
            'settings': 'threshold=0.2, min_num_speakers=2'
        })
    
    # Boundary recommendations
    if boundary_confidence < 0.4:
        recommendations.append({
            'priority': 'MEDIUM',
            'issue': 'Poor boundary agreement',
            'action': 'Try different VAD settings or audio preprocessing',
            'settings': 'Consider noise reduction or different VAD thresholds'
        })
    
    # Red flag recommendations
    if any('over-segmentation' in flag for flag in red_flags):
        recommendations.append({
            'priority': 'MEDIUM',
            'issue': 'Possible over-segmentation',
            'action': 'Increase clustering thresholds',
            'settings': 'Higher thresholds, longer minimum segment duration'
        })
    
    # General recommendations
    if speaker_confidence == "high" and boundary_confidence > 0.7:
        recommendations.append({
            'priority': 'INFO',
            'issue': 'High confidence results',
            'action': 'Results are reliable - proceed with confidence',
            'settings': 'Consider using hybrid approach for best segments'
        })
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        priority_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢', 'INFO': '🔵'}
        print(f"   {priority_emoji.get(rec['priority'], '📌')} {rec['priority']} PRIORITY:")
        print(f"      Issue: {rec['issue']}")
        print(f"      Action: {rec['action']}")
        print(f"      Settings: {rec['settings']}")
        print()
    
    return recommendations

def create_hybrid_output(scored_segments, base_dir):
    """Create a hybrid output file with confidence scores"""
    hybrid_path = os.path.join(base_dir, "hybrid_analysis.json")
    
    hybrid_data = {
        'metadata': {
            'created_by': 'Diarization Fact-Checker',
            'analysis_type': 'PyAnnote + NeMo Cross-Validation',
            'confidence_scoring': True,
            'base_directory': base_dir
        },
        'segments': scored_segments,
        'summary': {
            'total_segments': len(scored_segments),
            'high_confidence': len([s for s in scored_segments if s['confidence_level'] == 'HIGH']),
            'medium_confidence': len([s for s in scored_segments if s['confidence_level'] == 'MEDIUM']),
            'low_confidence': len([s for s in scored_segments if s['confidence_level'] == 'LOW'])
        }
    }
    
    with open(hybrid_path, 'w') as f:
        json.dump(hybrid_data, f, indent=2)
    
    print(f"\n💾 HYBRID ANALYSIS SAVED:")
    print(f"   File: {hybrid_path}")
    return hybrid_path

def main():
    """Main fact-checking function"""
    # YOUR SPECIFIC FOLDER PATH
    base_dir = "/Users/landlab/Desktop/ssaTranscription/25-07-14_12.11.00"
    
    print("🕵️‍♂️ ULTIMATE DIARIZATION FACT-CHECKER")
    print("=" * 60)
    print(f"📁 Analyzing: {base_dir}")
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        return
    
    # Check for audio file
    audio_file = os.path.join(base_dir, "audio.wav")
    if os.path.exists(audio_file):
        print(f"🎵 Audio file found: audio.wav")
    else:
        print(f"⚠️  Audio file not found: audio.wav")
    
    print()
    
    # Load both results
    pyannote_all, pyannote_count, pyannote_speech = analyze_pyannote_results(base_dir)
    nemo_all, nemo_count, nemo_speech = analyze_nemo_results(base_dir)
    
    if not pyannote_all and not nemo_all:
        print("❌ No diarization results found!")
        print("\n📋 TO DO:")
        print("   1. Run your PyAnnote script first")
        print("   2. Run your NeMo script")
        print("   3. Then run this fact-checker")
        return
    elif not pyannote_all:
        print("❌ PyAnnote results not found!")
        print("💡 Run your PyAnnote script first")
        return
    elif not nemo_all:
        print("❌ NeMo results not found!")
        print("💡 Run your NeMo script first")  
        return
    
    print(f"✅ Loaded PyAnnote: {len(pyannote_speech)} speech segments")
    print(f"✅ Loaded NeMo: {len(nemo_speech)} speech segments")
    
    # 1. Validate speaker counts
    speaker_confidence, speaker_recommendation = validate_speaker_count(pyannote_count, nemo_count)
    
    # 2. Cross-check boundaries
    agreements, boundary_confidence = cross_check_boundaries(pyannote_speech, nemo_speech)
    
    # 3. Detect red flags
    red_flags, warnings = detect_red_flags(pyannote_all, nemo_speech, pyannote_count, nemo_count)
    
    # 4. Create confidence scores
    scored_segments = create_confidence_scores(pyannote_all, nemo_speech, agreements)
    
    # 5. Generate recommendations
    recommendations = generate_recommendations(
        speaker_confidence, boundary_confidence, red_flags, pyannote_count, nemo_count
    )
    
    # 6. Create hybrid output
    hybrid_file = create_hybrid_output(scored_segments, base_dir)
    
    # 7. Final summary
    print("\n🏆 FINAL VERDICT:")
    print("=" * 40)
    
    if speaker_confidence == "high" and boundary_confidence > 0.7 and not red_flags:
        print("✅ HIGH CONFIDENCE: Both methods agree well!")
        print("💡 Recommendation: Use PyAnnote results with confidence")
        winner = "pyannote_validated"
        
    elif speaker_confidence in ["medium"] and boundary_confidence > 0.5:
        print("⚠️  MEDIUM CONFIDENCE: Some disagreements but manageable")
        print("💡 Recommendation: Use hybrid approach with caution")
        winner = "hybrid_recommended"
        
    else:
        print("🚨 LOW CONFIDENCE: Significant issues detected!")
        print("💡 Recommendation: Manual review and parameter tuning needed")
        winner = "manual_review_needed"
    
    print(f"\n📊 CONFIDENCE BREAKDOWN:")
    print(f"   Speaker Count: {speaker_confidence}")
    print(f"   Boundary Agreement: {boundary_confidence:.1%}")
    print(f"   Red Flags: {len(red_flags)} critical, {len(warnings)} warnings")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   Original PyAnnote: transcription.json")
    print(f"   Original NeMo: (search for .rttm files)")
    print(f"   Fact-Check Analysis: hybrid_analysis.json")
    
    return winner, recommendations

if __name__ == "__main__":
    main()
