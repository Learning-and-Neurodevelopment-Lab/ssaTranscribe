import torch
import torchaudio
from nemo.collections.asr.models.msdd_models import EncDecDiarLabelModel
from nemo.collections.asr.parts.features import FilterbankFeatures
from pathlib import Path
import numpy as np
import json
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dynamic_multiscale_args(audio_duration_sec):
    """Get multiscale arguments based on audio duration for memory efficiency"""
    
    if audio_duration_sec < 60:  # < 1 minute - ultra-conservative for memory
        logger.info(f"Using ultra-conservative settings for {audio_duration_sec:.1f}s audio")
        return {
            "scale_dict": {
                0: [6000, 3000],  # 6.0s segments with 3.0s hop (very coarse)
                1: [4000, 2000],  # 4.0s segments with 2.0s hop  
                2: [2000, 1000],  # 2.0s segments with 1.0s hop
                3: [1000, 500],   # 1.0s segments with 0.5s hop
                4: [500, 250]     # 0.5s segments with 0.25s hop (removed finest scale)
            }
        }
    elif audio_duration_sec < 600:  # 1-10 minutes - conservative
        logger.info(f"Using conservative settings for {audio_duration_sec:.1f}s audio")
        return {
            "scale_dict": {
                0: [10000, 5000], # 10.0s segments with 5.0s hop
                1: [6000, 3000],  # 6.0s segments with 3.0s hop  
                2: [3000, 1500],  # 3.0s segments with 1.5s hop
                3: [1500, 750],   # 1.5s segments with 0.75s hop
                4: [750, 375]     # 0.75s segments with 0.375s hop
            }
        }
    elif audio_duration_sec < 1800:  # 10-30 minutes - coarse granularity
        logger.info(f"Using coarse-scale settings for {audio_duration_sec:.1f}s audio")
        return {
            "scale_dict": {
                0: [15000, 7500], # 15.0s segments with 7.5s hop
                1: [10000, 5000], # 10.0s segments with 5.0s hop  
                2: [5000, 2500],  # 5.0s segments with 2.5s hop
                3: [2500, 1250],  # 2.5s segments with 1.25s hop
                4: [1250, 625]    # 1.25s segments with 0.625s hop
            }
        }
    else:  # > 30 minutes - very coarse for long episodes
        logger.info(f"Using extra-coarse settings for {audio_duration_sec:.1f}s audio")
        return {
            "scale_dict": {
                0: [20000, 10000], # 20.0s segments with 10.0s hop
                1: [15000, 7500],  # 15.0s segments with 7.5s hop  
                2: [10000, 5000],  # 10.0s segments with 5.0s hop
                3: [5000, 2500],   # 5.0s segments with 2.5s hop
                4: [2500, 1250]    # 2.5s segments with 1.25s hop
            }
        }

def load_multiscale_args(model=None, audio_duration_sec=None):
    """Load multiscale arguments from model config or use dynamic defaults"""
    
    # Try to get from model config first
    if model is not None and hasattr(model, 'cfg') and hasattr(model.cfg, 'multiscale_args_dict'):
        logger.info("Found multiscale_args_dict in model config")
        return model.cfg.multiscale_args_dict
    
    # Also check for multiscale_args.yaml files
    cache_dir = Path.home() / ".cache" / "torch" / "NeMo"
    if cache_dir.exists():
        yaml_files = list(cache_dir.rglob("multiscale_args.yaml"))
        if yaml_files:
            logger.info(f"Found multiscale_args.yaml at {yaml_files[0]}")
            try:
                import yaml
                with open(yaml_files[0], 'r') as f:
                    yaml_data = yaml.safe_load(f)
                if 'scale_dict' in yaml_data:
                    return yaml_data
            except Exception as e:
                logger.warning(f"Failed to load YAML file: {e}")
    
    # Check for JSON config files
    json_files = list(cache_dir.rglob("model_config.json")) if cache_dir.exists() else []
    for config_json_path in json_files:
        try:
            with open(config_json_path, "r") as f:
                config_data = json.load(f)
            multiscale_args_dict = config_data.get("multiscale_args_dict", None)
            if multiscale_args_dict is not None:
                logger.info(f"Loaded multiscale args from {config_json_path}")
                return multiscale_args_dict
        except Exception as e:
            logger.warning(f"Failed to load {config_json_path}: {e}")
    
    # Use dynamic multiscale args based on audio duration
    if audio_duration_sec is not None:
        logger.warning("Using dynamic multiscale args based on audio duration")
        return get_dynamic_multiscale_args(audio_duration_sec)
    else:
        # Fallback to medium scale if no duration provided
        logger.warning("Using default medium-scale multiscale args")
        return get_dynamic_multiscale_args(300)  # 5 minute default

def generate_multiscale_segments_from_model(audio_duration_ms, model=None):
    """Generate segments for each scale with proper alignment"""
    audio_duration_sec = audio_duration_ms / 1000.0
    multiscale_args_dict = load_multiscale_args(model, audio_duration_sec)
    scale_dict = multiscale_args_dict["scale_dict"]
    
    logger.info(f"Using scale_dict: {scale_dict}")
    logger.info(f"Audio duration: {audio_duration_ms:.2f} ms ({audio_duration_sec:.1f}s)")

    all_scales_segments = []
    
    # Handle both list and dict formats for scale_dict
    if isinstance(scale_dict, dict):
        scale_items = sorted(scale_dict.items())  # Sort by scale index
    else:
        scale_items = list(enumerate(scale_dict))  # Convert list to (index, value) pairs
    
    for scale_idx, (length_ms, hop_ms) in scale_items:
        segments = []
        start = 0.0
        while start < audio_duration_ms:
            end = min(start + length_ms, audio_duration_ms)
            segments.append([start / 1000.0, end / 1000.0])  # Convert to seconds
            if end >= audio_duration_ms:
                break
            start += hop_ms
        
        logger.info(f"Scale {scale_idx}: {len(segments)} segments (length={length_ms}ms, hop={hop_ms}ms)")
        all_scales_segments.append(segments)
    
    return all_scales_segments

def create_segment_tensors(all_scales_segments):
    """Create properly shaped segment tensors for MSDD model"""
    # Find the maximum number of segments across all scales
    max_segments = max(len(segments) for segments in all_scales_segments)
    num_scales = len(all_scales_segments)
    
    logger.info(f"Max segments: {max_segments}, Num scales: {num_scales}")
    
    # Create padded segment tensor: (batch, scales, max_segments, 2)
    ms_seg_timestamps = torch.zeros((1, num_scales, max_segments, 2), dtype=torch.float32)
    
    # Fill in the actual segments and pad with the last segment for shorter scales
    for scale_idx, segments in enumerate(all_scales_segments):
        for seg_idx, (start, end) in enumerate(segments):
            ms_seg_timestamps[0, scale_idx, seg_idx, 0] = start
            ms_seg_timestamps[0, scale_idx, seg_idx, 1] = end
        
        # Pad remaining slots with the last segment if needed
        if len(segments) < max_segments:
            last_seg = segments[-1]
            for seg_idx in range(len(segments), max_segments):
                ms_seg_timestamps[0, scale_idx, seg_idx, 0] = last_seg[0]
                ms_seg_timestamps[0, scale_idx, seg_idx, 1] = last_seg[1]
    
    # Create segment counts tensor: (batch, scales)
    # This should be cumulative counts, not individual counts
    individual_counts = [len(segments) for segments in all_scales_segments]
    cumulative_counts = []
    running_total = 0
    for count in individual_counts:
        running_total += count
        cumulative_counts.append(running_total)
    
    ms_seg_counts = torch.tensor([cumulative_counts], dtype=torch.int32)
    
    logger.info(f"ms_seg_timestamps shape: {ms_seg_timestamps.shape}")
    logger.info(f"ms_seg_counts shape: {ms_seg_counts.shape}")
    logger.info(f"Individual counts: {individual_counts}")
    logger.info(f"Cumulative counts: {cumulative_counts}")
    logger.info(f"ms_seg_counts values: {ms_seg_counts}")
    
    return ms_seg_timestamps, ms_seg_counts, max_segments

def create_model_inputs(all_scales_segments, feature_length):
    """Create all required model input tensors with correct shapes"""
    
    # Find the maximum number of segments across all scales
    max_segments = max(len(segments) for segments in all_scales_segments)
    num_scales = len(all_scales_segments)
    
    logger.info(f"Max segments: {max_segments}, Num scales: {num_scales}")
    
    # Create padded segment tensor: (batch, scales, max_segments, 2)
    ms_seg_timestamps = torch.zeros((1, num_scales, max_segments, 2), dtype=torch.float32)
    
    # Fill in the actual segments and pad with the last segment for shorter scales
    for scale_idx, segments in enumerate(all_scales_segments):
        for seg_idx, (start, end) in enumerate(segments):
            ms_seg_timestamps[0, scale_idx, seg_idx, 0] = start
            ms_seg_timestamps[0, scale_idx, seg_idx, 1] = end
        
        # Pad remaining slots with the last segment if needed
        if len(segments) < max_segments:
            last_seg = segments[-1]
            for seg_idx in range(len(segments), max_segments):
                ms_seg_timestamps[0, scale_idx, seg_idx, 0] = last_seg[0]
                ms_seg_timestamps[0, scale_idx, seg_idx, 1] = last_seg[1]
    
    # Create segment counts tensor: (batch, scales) - use actual counts, not cumulative
    ms_seg_counts = torch.tensor([[len(segments) for segments in all_scales_segments]], dtype=torch.int32)
    
    # Create cluster label index: (batch, max_segments)
    clus_label_index = torch.zeros((1, max_segments), dtype=torch.long)
    
    # Create scale mapping: (batch, scales, feature_length)
    scale_mapping = torch.zeros((1, num_scales, feature_length), dtype=torch.float32)
    
    # Fill scale mapping based on segment boundaries
    for scale_idx, segments in enumerate(all_scales_segments):
        for start_sec, end_sec in segments:
            # Convert seconds to frame indices (assuming 100 fps for mel features)
            start_frame = int(start_sec * 100)
            end_frame = int(end_sec * 100)
            start_frame = max(0, min(start_frame, feature_length - 1))
            end_frame = max(0, min(end_frame, feature_length))
            
            if start_frame < end_frame:
                scale_mapping[0, scale_idx, start_frame:end_frame] = 1.0
    
    # Create dummy targets for inference: (batch, max_segments, num_speakers)
    targets = torch.zeros((1, max_segments, 1), dtype=torch.float32)
    
    logger.info(f"ms_seg_timestamps shape: {ms_seg_timestamps.shape}")
    logger.info(f"ms_seg_counts shape: {ms_seg_counts.shape}")
    logger.info(f"ms_seg_counts values: {ms_seg_counts}")
    logger.info(f"clus_label_index shape: {clus_label_index.shape}")
    logger.info(f"scale_mapping shape: {scale_mapping.shape}")
    logger.info(f"targets shape: {targets.shape}")
    
    return ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets

def diarize_audio(audio_path, output_dir):
    """Main diarization function with comprehensive error handling"""
    
    # Clear GPU memory from any previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    logger.info("Loading model...")
    try:
        model = EncDecDiarLabelModel.from_pretrained("diar_msdd_telephonic")
        model.eval()
        device = next(model.parameters()).device
        logger.info(f"Model loaded on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    logger.info(f"Loading audio from: {audio_path}")
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.to(device)
        logger.info(f"Audio loaded: shape={waveform.shape}, sr=16000")
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        raise

    logger.info("Extracting features...")
    try:
        featurizer = FilterbankFeatures(
            sample_rate=16000,
            n_window_size=400,  # 25ms window
            n_window_stride=160,  # 10ms stride  
            n_fft=512
        ).to(device)
        
        with torch.no_grad():
            # Use positional arguments for FilterbankFeatures
            features, feature_length = featurizer(
                waveform, 
                torch.tensor([waveform.shape[1]], dtype=torch.long).to(device)
            )
        
        logger.info(f"Raw features shape: {features.shape}")
        
        # Model expects (batch, time) shape according to error message
        # Features come as (batch, freq, time), so we need to transpose and process
        if features.dim() == 3:
            # Transpose from (batch, freq, time) to (batch, time, freq)
            features = features.transpose(1, 2)  # Now (batch, time, freq)
            # Then average across frequency dimension to get (batch, time)
            features = features.mean(dim=-1)  # (batch, time)
        elif features.dim() == 2:
            # Already (batch, time) or (time, batch) - ensure correct order
            if features.shape[0] != 1:  # If not batch-first
                features = features.transpose(0, 1)
        
        feature_length = feature_length.cpu().item()
        logger.info(f"Features processed: shape={features.shape}, length={feature_length}")
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise

    # Generate multiscale segments
    audio_duration_ms = waveform.shape[1] / 16000 * 1000
    logger.info(f"Audio duration: {audio_duration_ms:.2f} ms")
    
    # Clear some GPU memory before processing segments
    torch.cuda.empty_cache()
    
    try:
        all_scales_segments = generate_multiscale_segments_from_model(audio_duration_ms, model)
        
        # Create model input tensors
        ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets = create_model_inputs(
            all_scales_segments, feature_length
        )
        
        # Move all tensors to device
        ms_seg_timestamps = ms_seg_timestamps.to(device)
        ms_seg_counts = ms_seg_counts.to(device)
        clus_label_index = clus_label_index.to(device)
        scale_mapping = scale_mapping.to(device)
        targets = targets.to(device)
        
        logger.info(f"GPU memory after tensor creation: {torch.cuda.memory_allocated()/1e9:.2f}GB used")
        
    except Exception as e:
        logger.error(f"Segment generation failed: {e}")
        raise

    logger.info("Running diarization inference...")
    try:
        # Ensure model has required attributes
        if not hasattr(model, 'emb_batch_size'):
            model.emb_batch_size = 0
            
        if not hasattr(model, 'multiscale_args_dict'):
            # Add the multiscale_args_dict we used for segment generation
            audio_duration_sec = waveform.shape[1] / 16000
            multiscale_args_dict = load_multiscale_args(model, audio_duration_sec)
            model.multiscale_args_dict = multiscale_args_dict
            logger.info("Added multiscale_args_dict to model")
        
        with torch.no_grad():
            # Try the model's diarize method instead of forward
            # Some MSDD models expect this interface
            if hasattr(model, 'diarize'):
                logger.info("Using model.diarize() method")
                diar_hyp = model.diarize(
                    features=features,
                    ms_seg_timestamps=ms_seg_timestamps,
                    ms_seg_counts=ms_seg_counts,
                    clus_label_index=clus_label_index,
                    scale_mapping=scale_mapping,
                    targets=targets,
                )
            else:
                logger.info("Using model.forward() method")
                diar_hyp, _, _ = model.forward(
                    features=features,
                    feature_length=torch.tensor([feature_length], dtype=torch.long).to(device),
                    ms_seg_timestamps=ms_seg_timestamps,
                    ms_seg_counts=ms_seg_counts,
                    clus_label_index=clus_label_index,
                    scale_mapping=scale_mapping,
                    targets=targets,
                )
        
        logger.info(f"Diarization completed. Output: {diar_hyp}")
        
    except Exception as e:
        logger.error(f"Model forward pass failed: {e}")
        logger.error(f"Features shape: {features.shape}")
        logger.error(f"Feature length: {feature_length}")
        logger.error(f"ms_seg_timestamps shape: {ms_seg_timestamps.shape}")
        logger.error(f"ms_seg_counts shape: {ms_seg_counts.shape}")
        raise

    # Save results
    output_path = Path(output_dir) / (Path(audio_path).stem + "_diarization.txt")
    try:
        with open(output_path, "w") as f:
            f.write(f"Audio file: {audio_path}\n")
            f.write(f"Duration: {audio_duration_ms/1000:.2f} seconds\n")
            f.write(f"Number of segments: {len(all_scales_segments[-1])}\n\n")
            
            # Use the finest scale (last one) for output
            base_segments = all_scales_segments[-1]
            
            if isinstance(diar_hyp, torch.Tensor):
                diar_hyp = diar_hyp.cpu().numpy()
            
            for i, label in enumerate(diar_hyp):
                if i < len(base_segments):
                    start_sec, end_sec = base_segments[i]
                    f.write(f"{start_sec:.2f}-{end_sec:.2f}s: Speaker {label}\n")
        
        logger.info(f"Diarization output saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <audio_file> <output_folder>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    output_folder = sys.argv[2]
    
    # Create output directory if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    try:
        diarize_audio(audio_file, output_folder)
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        sys.exit(1)