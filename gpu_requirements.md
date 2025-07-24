# 🚨 EXTREME GPU REQUIRED

These scripts **will not run properly** on laptops or standard GPUs — including the NVIDIA **RTX 3090**.

---

## ✅ Minimum Required:

- NVIDIA **A100**, **H100**, or equivalent
- CUDA **11.8+**
- At least **24 GB GPU memory**
- Strong support for **Triton** and **PyTorch 2.x**

> ⚠️ Even RTX 3090 may **fail** on full diarization tasks.

---

## ⚙️ What the Scripts Do

- `step1_transcribe.py`: WhisperX transcription — **GPU required**
- `step2_diarize.py`: NeMo MSDD speaker diarization — **GPU absolutely required**
- `step3_merge_output.py`: Combines results — can run on **CPU**