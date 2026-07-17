# Separate-and-Detect: Unified Drum Transcription and Stem Generation via Latent Diffusion

# Abstract

Automatic Drum Transcription (ADT) is commonly formulated as a direct mapping from a music mixture to symbolic drum events. While effective for transcription, this formulation discards the acoustic stems that are useful for editing, remixing, and production. We revisit an alternative separate-and-detect formulation, where a drum source separation front end first produces five editable drum stems, and a fixed onset detector then converts each stem into symbolic events. The separator is built on a five-stem latent diffusion model that jointly generates kick, snare, toms, hi-hats, and cymbals in a compact VAE latent space. We further study two training-only auxiliary branches-an onset branch (OB) and a timbre branch (TB)-which shape the separator during learning but are discarded at inference. Trained on synthetic drum multitracks and evaluated on MDB Drums and ENST-Drums, the proposed pipeline consistently improves over a strong U-Net-based drum separation baseline in overall transcription F1. It also outperforms a representative end-to-end ADT system on kick and snare F1 under our evaluation protocol, while additionally providing separated audio stems. The ablation results show that OB gives the most stable transcription gains, whereas TB changes the trade-off between reconstruction, perceptual stem quality, and onset detection. These results suggest that generative drum demixing can serve not only as a source separation model, but also as a practical front end for interpretable drum transcription.

# Installation

This project supports two conda environments for different use cases:

## Environment Setup

### Environment 1: `musicldm_env`
- **Purpose**: Original environment for MusicLDM related tasks
- **Usage**: For running the original batch evaluation scripts
- **Python**: 3.9
- **Includes**: librosa, madmom, scipy, numpy, matplotlib
- **Optimized for**: batch processing and evaluation

### Environment 2: `onset_detect`
- **Purpose**: Onset Detection environment
- **Usage**: For running the single audio onset detection script
- **Python**: 3.10
- **Includes**: librosa, madmom, scipy, numpy
- **Optimized for**: single audio onset detection

### Quick Setup

Create environments from yml files:
```bash
# Create musicldm_env (for batch evaluation)
conda env create -f musicldm_env.yml

# Create onset_detect environment (for single audio detection)
conda env create -f onset_detect.yml
```

Activate environments:
```bash
# For batch evaluation and training
conda activate musicldm_env

# For single audio onset detection
conda activate onset_detect
```


### (Be careful !) Fix madmom Import Issue

Modify the madmom package to fix compatibility issues:

1. Navigate to the madmom processors file:
   ```
   <conda_env_path>/lib/python3.10/site-packages/madmom/processors.py 
   ```

2. Edit line 23:

   **Change from:**
   ```python
   from collections import MutableSequence
   ```
   
   **Change to:**
   ```python
   from collections.abc import MutableSequence
   ```

### Usage Examples

**Batch Evaluation (musicldm_env):**
```bash
conda activate musicldm_env
python integrated_train.py --config <config_path>
```

**Single Audio Detection (onset_detect):**
```bash
conda activate onset_detect
bash run_single_onset_detection.sh
```


# Data

We use the StemGMD and IDMT-SMT-Drums datasets in this project.

Please download them from the following links and organize them to match the structure under the `data` folder in this repository:

- StemGMD: https://zenodo.org/records/7860223
- IDMT-SMT-Drums: https://zenodo.org/records/7544164

After downloading, preprocess and arrange your data to mirror the examples under the `data` directory (all examples reside in `data`). This ensures the training and evaluation scripts can locate audio and annotations correctly.

# Training MSG-LD

After data and conda environments are installed properly, you will need to download components of MusicLDM that are used for MSG-LD too. For this please 

```
# Download hifigan-ckpt.ckpt
wget https://zenodo.org/record/10643148/files/hifigan-ckpt.ckpt

# Download vae-ckpt.ckpt
wget https://zenodo.org/record/10643148/files/vae-ckpt.ckpt

```

After placing these files in your preferred directory and updating their paths in the corresponding config, run the following to train MSG-LD:

```
python inference_train.py --config config/MSG-LD/integrated_musicldm.yaml
```

# Config YAML

Common configs under `config/MSG-LD/` and when to use them:

- `integrated_musicldm.yaml`
  - Purpose: Baseline MSG-LD (latent diffusion separator) without auxiliaries.
  - Use when: You want a simple baseline to compare against auxiliary branches.

- `integrated_musicldm_onset.yaml`
  - Purpose: Adds an onset auxiliary branch to encourage percussion-aware separation.
  - Use when: You want better alignment of percussive cues without the timbre auxiliary.

- `integrated_musicldm_onset_timbre.yaml`
  - Purpose: Adds both onset and timbre auxiliary branches (recommended for drums).
  - Use when: You want best downstream drum transcription with editable stems.

- `inference_musicldm_mdb_inference.yaml`
  - Purpose: Inference/evaluation on the MDB-Drums dataset.
  - Use when: Running separation (and configured evaluation) on MDB splits.
  - Make sure: Dataset roots, checkpoint paths, and output dirs are correct.

- `inference_musicldm_enst_inference.yaml`
  - Purpose: Inference/evaluation on the ENST-Drums dataset.
  - Use when: Running separation (and configured evaluation) on ENST splits.
  - Make sure: Dataset roots, checkpoint paths, and output dirs are correct.

Example (training with onset+timbre):
```bash
CUDA_VISIBLE_DEVICES=0 \
python integrated_train.py --config config/MSG-LD/integrated_musicldm_onset_timbre.yaml
```

# Inference

Two typical inference paths, mapped to the two environments.

1) Separation / dataset-level evaluation (musicldm_env)
```bash
conda activate musicldm_env

# MDB-Drums inference/eval (config controls dataset split/paths/checkpoints)
CUDA_VISIBLE_DEVICES=0 \
python integrated_train.py --config config/MSG-LD/inference_musicldm_mdb_inference.yaml --separate_only

# ENST-Drums inference/eval
CUDA_VISIBLE_DEVICES=0 \
python integrated_train.py --config config/MSG-LD/inference_musicldm_enst_inference.yaml --separate_only

# Single audio file separation (test inference) --> What ever music you like
CUDA_VISIBLE_DEVICES=0 \
python integrated_train.py --config config/MSG-LD/integrated_musicldm_test_inference.yaml --separate_only
```
Tips:
- Set `CUDA_VISIBLE_DEVICES` to select a GPU (optional).
- Verify paths inside the inference yaml(s): checkpoints, dataset roots, and output directories.
- For single audio separation, modify these paths in the config:
  - `data.params.path.valid_data`: Where to read separated stems from
  - `mdb_eval.demucs_input_root`: Input audio file(s) directory
  - `mdb_eval.demucs_output_root`: Demucs separation output directory

2) Single-audio onset transcription (onset_detect)
```bash
conda activate onset_detect

# Quick start (uses example path inside the script)
bash run_single_onset_detection.sh

# Or specify your own input/output
python single_audio_onset_detection.py \
  --input_audio /absolute/path/to/val_0/mix/YourSong.wav \
  --output_dir /absolute/path/to/transcription_results
```
Output transcripts are saved as `.txt`, one onset per line:
```
<timestamp_seconds>    <drum_type>
```

# Checkpoint release

All the model checkpoints will be released upon acceptance.

# Demo page

The demo page is available at:

[Demo Page](https://ddman1101.github.io/Separate-and-Detect-demo/)
