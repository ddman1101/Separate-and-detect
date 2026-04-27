#!/bin/bash

# Single Audio Onset Detection Transcription Script
# Example usage script

echo "=========================================="
echo "Single Audio Onset Detection Transcription"
echo "=========================================="


# Set paths
SCRIPT_PATH="/PATH/TO/PROJECT/single_audio_onset_detection.py"
OUTPUT_DIR="/PATH/TO/PROJECT/transcription_results"

# Example audio file path (modify according to your actual path)
EXAMPLE_AUDIO="/PATH/TO/PROJECT/inference_outputs/val_0/mix/MusicDelta_Beatles_MIX_from_0.wav"

# Activate conda environment
# echo "Activating onset_detect environment..."
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate onset_detect

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found: $SCRIPT_PATH"
    exit 1
fi

# Check if example audio file exists
if [ ! -f "$EXAMPLE_AUDIO" ]; then
    echo "Warning: Example audio file not found: $EXAMPLE_AUDIO"
    echo "Please modify the EXAMPLE_AUDIO variable in the script to your actual audio file path"
    echo ""
    echo "Usage:"
    echo "python $SCRIPT_PATH --input_audio /path/to/your/audio.wav --output_dir $OUTPUT_DIR"
    exit 1
fi

echo "Using example audio file: $EXAMPLE_AUDIO"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Execute onset detection
python "$SCRIPT_PATH" \
    --input_audio "$EXAMPLE_AUDIO" \
    --output_dir "$OUTPUT_DIR"

# Check execution result
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Transcription completed!"
    echo "Results saved in: $OUTPUT_DIR"
    echo ""
    echo "View transcription results:"
    echo "cat $OUTPUT_DIR/*_transcription.txt"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Transcription failed! Please check error messages"
    echo "=========================================="
    exit 1
fi