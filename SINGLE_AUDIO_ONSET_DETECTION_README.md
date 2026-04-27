# Single Audio Onset Detection Transcription Script

This script performs onset detection on specified audio files and outputs transcription results with a txt file.
## Features

- Uses optimized parameters for onset detection
- Outputs standardized transcription results (timestamp + drum type)
- Provides detailed statistics

## Usage

### 1. Direct Python Script Usage

First activate the conda environment:
```bash
conda activate onset_detect
```

**Note**: This script requires the `onset_detect` environment. If you don't have it, create it from the provided yml file:
```bash
conda env create -f onset_detect.yml
```

Then run the script:
```bash
python single_audio_onset_detection.py \
    --input_audio /path/to/your/audio.wav \
    --output_dir ./transcription_results
```

### 2. Using Example Script (Recommended)

```bash
bash run_single_onset_detection.sh
```

The example script will automatically activate the correct conda environment.

## Parameters

- `--input_audio`: Input audio file path (required)
- `--output_dir`: Output directory (optional, default: ./transcription_results)

## Input Requirements

The input audio file path must follow this structure:
```
/path/to/val_0/mix/your_audio.wav
```

The script will automatically look for corresponding stem files in the `val_0` directory:
- `val_0/stem_0/your_audio.wav` (kick)
- `val_0/stem_1/your_audio.wav` (snare)
- `val_0/stem_2/your_audio.wav` (toms)
- `val_0/stem_3/your_audio.wav` (hi_hats)
- `val_0/stem_4/your_audio.wav` (cymbals)

## Output Format

Transcription results are saved as `.txt` files with the following format:
```
timestamp    drum_type
0.123456    Kick
0.456789    Snare
0.789012    Hi_Hats
...
```

Where:
- `timestamp`: Time stamp (seconds)
- `drum_type`: Drum type (Kick, Snare, Toms, Hi_Hats, Cymbals)

## Example

### Input
```bash
python single_audio_onset_detection.py \
    --input_audio /PATH/TO/PROJECT/inference_outputs/val_0/mix/MusicDelta_Beatles_MIX_from_0.wav \
    --output_dir ./transcription_results
```

Input audio could give the "mix path"

### Output
```
==========================================
Single Audio Onset Detection Transcription
==========================================
Using example audio file: /PATH/TO/PROJECT/inference_outputs/val_0/mix/MusicDelta_Beatles_MIX_from_0.wav
Output directory: /PATH/TO/PROJECT/transcription_results

Processing audio file: /PATH/TO/PROJECT/inference_outputs/val_0/mix/MusicDelta_Beatles_MIX_from_0.wav
Found 5 stem files
  Processing kick: /PATH/TO/PROJECT/inference_outputs/val_0/stem_0/MusicDelta_Beatles_MIX_from_0.wav
    Detected 14 onsets
  Processing snare: /PATH/TO/PROJECT/inference_outputs/val_0/stem_1/MusicDelta_Beatles_MIX_from_0.wav
    Detected 9 onsets
  Processing toms: /PATH/TO/PROJECT/inference_outputs/val_0/stem_2/MusicDelta_Beatles_MIX_from_0.wav
    Detected 9 onsets
  Processing hi_hats: /PATH/TO/PROJECT/inference_outputs/val_0/stem_3/MusicDelta_Beatles_MIX_from_0.wav
    Detected 12 onsets
  Processing cymbals: /PATH/TO/PROJECT/inference_outputs/val_0/stem_4/MusicDelta_Beatles_MIX_from_0.wav
    Detected 0 onsets
Transcription results saved to: /PATH/TO/PROJECT/transcription_results/MusicDelta_Beatles_MIX_from_0_transcription.txt
Total detected 44 onsets

Drum type statistics:
  Hi_Hats: 12
  Kick: 14
  Snare: 9
  Toms: 9

==========================================
Transcription completed!
Results saved in: /PATH/TO/PROJECT/transcription_results

View transcription results:
cat /PATH/TO/PROJECT/transcription_results/*_transcription.txt
==========================================
```