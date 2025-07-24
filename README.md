# 🎙️ GPU Transcript Pipeline

This pipeline transcribes and diarizes multi-speaker audio (like *Sesame Street* episodes) using:
- 🗣️ **WhisperX** for transcription
- 👥 **NVIDIA NeMo MSDD** for speaker diarization

---

## ⚠️ GPU WARNING

> These scripts **require a much stronger GPU than an NVIDIA RTX 3090**.

### ✅ You need:
- NVIDIA **A100**, **H100**, or higher
- At least **24 GB GPU VRAM**
- **CUDA 11.8+**
- Properly installed **PyTorch**, **Triton**, and **NeMo**

> ⚠️ Do **not** attempt this on a laptop or mid-range GPU — it will crash or produce invalid results.

See [`gpu_requirements.md`](gpu_requirements.md) for more details.

---

## 🧪 Pipeline Steps

1. **`step1_transcribe.py`**  
   → Runs WhisperX transcription and word alignment  
   → Outputs: `aligned_segments.json`

2. **`step2_diarize.py`**  
   → Performs multiscale speaker diarization using NeMo MSDD  
   → Outputs: `<audio_name>_diarization.txt`

3. **`step3_merge_output.py`**  
   → Merges transcribed words with speaker labels  
   → Adds scene tags and NER (e.g., "Elmo", "Hooper’s Store")  
   → Outputs:  
   - `transcription.md` (readable transcript)  
   - `transcription.json` (structured metadata)

---

## 📦 Environments

Use these files to install required packages:

- `Environments/whisperx_env_requirements.txt`
- `Environments/nemo_env_requirements.txt`

### 🛠️ To recreate:

```bash
python3 -m venv my_env
source my_env/bin/activate
pip install -r Environments/whisperx_env_requirements.txt  # or nemo_env_requirements.txt