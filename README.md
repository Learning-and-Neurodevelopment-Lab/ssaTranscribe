# Pipeline Overview

`ssaTranscribe` is an open-source, self-hosted pipeline that delivers audio-to-text speaker diarization, voice matching, multilingual vocabulary, and keyword detection in end-to-end transcriptions of episodic media.

Developed to support the expansion of the [_Sesame Street_ Archive (SSA)](https://www.sesamestreetarchive.io/), the primary goal of `ssaTranscribe` is to transcribe 4,397 original _Sesame Street_ episodes aired between 1969-2018, enabling a multimodal approach to scaling the SSA from ~5,000 to 35,000 labeled images across `face`, `number`, `place`, and `word` object categories.

By surfacing rich dialogue from the world's most longstanding children’s educational television series, `ssaTranscribe` identifies visually imaginative, stylistically diverse scenes that reflect children's experiences of wonder, play, and embodied learning—strategically curating the magical content behind the SSA.

## Directory Structure

This directory tree outlines the top-level folders and essential files included in `ssaTranscribe`:

```plaintext
ssaTranscribe/
├── data/                                        # Input and output data directories
│   ├── raw/                                     # Original unprocessed audio files
│   └── processed/                               # Audio clips segmented and ready for ASR
├── models/                                      # Pretrained model weights
│   └── asr_model.pt                             # Whisper or other ASR model file
├── scripts/                                     # Core logic for each pipeline step
│   ├── transcriptionwithspeakersv1.py           # Main script for processing audio to text
│   ├── diarize.py                               # Performs speaker diarization
│   └── postprocess.py                           # Formats and cleans transcripts
├── configs/                                     # YAML configuration files
│   └── default.yaml                             # Default settings for pipeline runs
├── README.md                                    # Project overview and usage
├── requirements.txt                             # Python package dependencies
└── run_pipeline.sh                              # End-to-end pipeline entry point
```

## Data Flow

This table maps the flow of data through the `ssaTranscribe` pipeline, from original _Sesame Street_ episode to readable, searchable output:

|     | Stage                  | Input Format    | Process (File / Script)                             | Output Format                     | Output Location        |
|-----| ---------------------- | --------------- | --------------------------------------------------- | --------------------------------- | ---------------------- |
|**1**| Audio Ingestion        | `.wav`, `.mp3`  | *Raw input in* `data/raw/`                          | (same)                            | `data/raw/`            |
|**2**| Segmentation / Prep    | `.wav`, `.mp3`  | `scripts/segment.py` *(if used)*                    | `.wav` (cleaned/trimmed)          | `data/processed/`      |
|**3**| Transcription (ASR)    | `.wav`          | `scripts/transcribe.py` using `models/asr_model.pt` | `.txt`, `.json`                   | `outputs/transcripts/` |
|**4**| Diarization (Optional) | `.wav`          | `scripts/diarize.py`                                | `.json` speaker tags              | `outputs/transcripts/` |
|**5**| Postprocessing         | `.json`, `.txt` | `scripts/postprocess.py`                            | `.srt`, cleaned `.json` or `.txt` | `outputs/transcripts/` |
|**6**| Configuration          | `.yaml`         | `configs/default.yaml`                              | (used throughout)                 | —                      |

## Implementation

### 💪 Compute

**Required GPU Type:**
- .

**Minimum CPU and RAM Specifications:**
- .

**Disk Space Requirements:**
- .

**Estimated Runtime Per Episode (With/Without GPU):**
- .

---

### 📩 Installation

**Python Version:**
- .

**Environment Setup Steps:**
- .

**Required Python Packages:**
- .

**Pretrained Model Download Instructions:**
- .

**System-Level Dependencies:**
- .

---

### ⚙️ Configuration

**Main Configuration File Location:**
- .

**User-Editable Parameters:**
- .

**Required Environment Variables:**
- .

**Command-Line Arguments, Flags:**
- .

---

### 🎬 Execution

**Full Pipeline Execution Command:**
- .

**Running Individual Pipeline Components:**
- .

**Expected Input and Output File Types:**
- .

**Default Output Directory and File Locations:**
- .
