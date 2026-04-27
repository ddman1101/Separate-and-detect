#!/usr/bin/env python3
"""
Single Audio Onset Detection Transcription Script
Based on mdb_onset_evaluation.py method, performs onset detection on specified audio file and outputs transcription results

Usage:
python single_audio_onset_detection.py --input_audio /path/to/audio.wav --output_dir /path/to/output

Example:
python single_audio_onset_detection.py \
    --input_audio /PATH/TO/PROJECT/inference_outputs/val_0/mix/MusicDelta_Beatles_MIX_from_0.wav \
    --output_dir ./transcription_results
"""

import os
import argparse
import librosa
import numpy as np
from pathlib import Path
from madmom.features.onsets import (
    CNNOnsetProcessor,
    OnsetPeakPickingProcessor,
    SpectralOnsetProcessor
)
from madmom.audio.signal import Signal
import scipy.signal
from collections import defaultdict

# 5-stem mapping
STEM_MAPPING = {
    'kick': 'Kick',
    'snare': 'Snare', 
    'toms': 'Toms',
    'hi_hats': 'Hi_Hats',
    'cymbals': 'Cymbals'
}

# Best parameters
DETECTION_PARAMS = {
    'kick': {
        'threshold': 0.7123620356542087,
        'min_separation': 0.12605714451279332,
        'cutoff': 5659.969709057025
    },
    'snare': {
        'threshold': 0.24425729153651546,
        'min_separation': 0.059295203501098154,
        'cutoff': 1248.559112730783
    },
    'toms': {
        'threshold': 0.5571172192068058,
        'min_separation': 0.19440705946102893,
        'cutoff_low': 299,
        'cutoff_high': 1185.8016508799908
    },
    'hi_hats': {
        'threshold': 0.08597749010989753,
        'min_separation': 0.06948347430929988,
        'cutoff': 517.9637529565225
    },
    'cymbals': {
        'threshold': 0.39457664043870916,
        'min_separation': 0.1796002881571787,
        'cutoff': 1115.370160765871
    }
}

def butter_lowpass(cutoff, sr, order=5):
    """Create low-pass filter coefficients"""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def butter_highpass(cutoff, sr, order=5):
    """Create high-pass filter coefficients"""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a

def apply_filter(audio, cutoff, sr, filter_type='highpass'):
    """Apply filter to audio signal"""
    if filter_type == 'highpass':
        b, a = butter_highpass(cutoff, sr)
    else:
        b, a = scipy.signal.butter(5, cutoff / (0.5 * sr), btype='lowpass', analog=False)
    filtered = scipy.signal.filtfilt(b, a, audio)
    return filtered

def detect_kick_onsets(audio_path, threshold=0.8, min_separation=0.08, cutoff=None):
    """Detect Kick onsets"""
    if not os.path.isfile(audio_path):
        return []
    
    # Load audio file
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Apply low-pass filter if cutoff frequency is specified
    if cutoff is not None and cutoff < sr/2:
        b, a = butter_lowpass(cutoff, sr)
        audio = scipy.signal.filtfilt(b, a, audio)
    
    # Add 1 second padding
    offset_sec = 1.0
    offset_samples = int(offset_sec * sr)
    padded_audio = np.concatenate([np.zeros(offset_samples), audio])
    padded_signal = Signal(padded_audio, sample_rate=sr)
    
    proc_CNN = CNNOnsetProcessor()  
    activation_cnn = proc_CNN(padded_signal)
    
    peak_picking = OnsetPeakPickingProcessor(threshold=threshold)
    detected_onsets_cnn_padded = peak_picking(activation_cnn)
    detected_onsets_cnn = detected_onsets_cnn_padded - offset_sec
    detected_onsets_cnn[detected_onsets_cnn < 0] = 0
    
    # Basic boundary filtering: remove onsets near audio start and end
    duration = len(audio) / sr
    boundary_threshold = 0.2  # 0.2 second boundary threshold
    
    filtered = []
    for onset in detected_onsets_cnn:
        # Filter boundary onsets
        if onset < boundary_threshold or onset > (duration - boundary_threshold):
            continue
        filtered.append(onset)
    
    return filtered

def detect_snare_onsets(audio_path, threshold=0.5, min_separation=0.12, cutoff=2832):
    """Detect Snare onsets"""
    if not os.path.isfile(audio_path):
        return []
    
    # Load audio file with specified sample rate
    sr = 16000
    audio, sr = librosa.load(audio_path, sr=sr)
    
    # Optimize cutoff frequency
    nyquist = sr / 2
    cutoff_freq = cutoff
    normal_cutoff = cutoff_freq / nyquist
    b, a = scipy.signal.butter(5, normal_cutoff, btype='low')
    audio = scipy.signal.filtfilt(b, a, audio)
    
    # Add 1 second padding
    offset_sec = 1.0
    offset_samples = int(offset_sec * sr)
    padded_audio = np.concatenate([np.zeros(offset_samples), audio])
    padded_signal = Signal(padded_audio, sample_rate=sr)
    
    proc_CNN = CNNOnsetProcessor()  
    activation_cnn = proc_CNN(padded_signal)
    
    peak_picking = OnsetPeakPickingProcessor(threshold=threshold,
                                           pre_avg=0.03, post_avg=0.03, 
                                           pre_max=0.03, post_max=0.03)
    detected_onsets_cnn_padded = peak_picking(activation_cnn)
    detected_onsets_cnn = detected_onsets_cnn_padded - offset_sec
    detected_onsets_cnn[detected_onsets_cnn < 0] = 0
    
    # Basic boundary filtering
    duration = len(audio) / sr
    boundary_threshold = 0.2
    
    filtered = []
    for onset in detected_onsets_cnn:
        if onset < boundary_threshold or onset > (duration - boundary_threshold):
            continue
        filtered.append(onset)
    
    return filtered

def detect_toms_onsets(audio_path, threshold=0.4, min_separation=0.2, cutoff_low=299, cutoff_high=2244):
    """Detect Toms onsets"""
    if not os.path.isfile(audio_path):
        return []
    
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Adjust band-pass filter
    if cutoff_low is not None and cutoff_low > 0:
        audio = apply_filter(audio, cutoff_low, sr, 'highpass')
    if cutoff_high is not None and cutoff_high > 0 and cutoff_high < sr / 2:
        audio = apply_filter(audio, cutoff_high, sr, 'lowpass')
    
    # Add 1 second padding
    offset_sec = 1.0
    offset_samples = int(offset_sec * sr)
    padded_audio = np.concatenate([np.zeros(offset_samples), audio])
    padded_signal = Signal(padded_audio, sample_rate=sr)
    
    proc_CNN = CNNOnsetProcessor()  
    activation_cnn = proc_CNN(padded_signal)
    
    peak_picking = OnsetPeakPickingProcessor(threshold=threshold,
                                           pre_avg=0.03, post_avg=0.03, 
                                           pre_max=0.03, post_max=0.03)
    detected_onsets_cnn_padded = peak_picking(activation_cnn)
    detected_onsets_cnn = detected_onsets_cnn_padded - offset_sec
    detected_onsets_cnn[detected_onsets_cnn < 0] = 0
    
    # Basic boundary filtering
    duration = len(audio) / sr
    boundary_threshold = 0.2
    
    filtered = []
    for onset in detected_onsets_cnn:
        if onset < boundary_threshold or onset > (duration - boundary_threshold):
            continue
        filtered.append(onset)
    
    return filtered

def detect_hi_hats_onsets(audio_path, threshold=0.4, min_separation=0.08, cutoff=600.5530232910016):
    """Detect Hi-Hats onsets"""
    if not os.path.isfile(audio_path):
        return []
    
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Adjust cutoff frequency
    cutoff_freq = cutoff
    audio = apply_filter(audio, cutoff_freq, sr, 'highpass')
    
    # Add 1 second padding
    offset_sec = 1.0
    offset_samples = int(offset_sec * sr)
    padded_audio = np.concatenate([np.zeros(offset_samples), audio])
    padded_signal = Signal(padded_audio, sample_rate=sr)
    
    proc_CNN = CNNOnsetProcessor()  
    activation_cnn = proc_CNN(padded_signal)
    
    peak_picking = OnsetPeakPickingProcessor(threshold=threshold, 
                                           pre_avg=0.02, post_avg=0.02, 
                                           pre_max=0.02, post_max=0.02)
    detected_onsets_cnn_padded = peak_picking(activation_cnn)
    detected_onsets_cnn = detected_onsets_cnn_padded - offset_sec
    detected_onsets_cnn[detected_onsets_cnn < 0] = 0
    
    filtered = []
    for onset in detected_onsets_cnn:
        if not filtered or (onset - filtered[-1] >= min_separation):
            filtered.append(onset)
    
    return filtered

def detect_cymbals_onsets(audio_path, threshold=0.6, min_separation=0.15, cutoff=1249.3631467527541):
    """Detect Cymbals onsets"""
    if not os.path.isfile(audio_path):
        return []
    
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # High-frequency filtering
    cutoff_freq = cutoff
    audio = apply_filter(audio, cutoff_freq, sr, 'highpass')
    
    # Add 1 second padding
    offset_sec = 1.0
    offset_samples = int(offset_sec * sr)
    padded_audio = np.concatenate([np.zeros(offset_samples), audio])
    padded_signal = Signal(padded_audio, sample_rate=sr)
    
    proc_CNN = CNNOnsetProcessor()  
    activation_cnn = proc_CNN(padded_signal)
    
    peak_picking = OnsetPeakPickingProcessor(threshold=threshold,
                                           pre_avg=0.03, post_avg=0.03, 
                                           pre_max=0.03, post_max=0.03)
    detected_onsets_cnn_padded = peak_picking(activation_cnn)
    detected_onsets_cnn = detected_onsets_cnn_padded - offset_sec
    detected_onsets_cnn[detected_onsets_cnn < 0] = 0
    
    # Basic boundary filtering
    duration = len(audio) / sr
    boundary_threshold = 0.2
    
    filtered = []
    for onset in detected_onsets_cnn:
        if onset < boundary_threshold or onset > (duration - boundary_threshold):
            continue
        filtered.append(onset)
    
    return filtered

def find_stem_files(input_audio_path):
    """Find corresponding stem files based on input audio path"""
    # Extract basic information from input path
    input_path = Path(input_audio_path)
    filename = input_path.stem  # e.g. MusicDelta_Beatles_MIX_from_0
    
    # Find val_0 directory
    val_0_dir = None
    current_path = input_path.parent
    while current_path != current_path.parent:  # until root directory
        if current_path.name == 'val_0':
            val_0_dir = current_path
            break
        current_path = current_path.parent
    
    if val_0_dir is None:
        raise ValueError(f"Cannot find val_0 directory in the path of {input_audio_path}")
    
    # Build stem file paths
    stem_files = {}
    stem_dirs = {
        'kick': 'stem_0',
        'snare': 'stem_1', 
        'toms': 'stem_2',
        'hi_hats': 'stem_3',
        'cymbals': 'stem_4'
    }
    
    for stem_name, stem_dir in stem_dirs.items():
        stem_file = val_0_dir / stem_dir / f"{filename}.wav"
        if stem_file.exists():
            stem_files[stem_name] = str(stem_file)
        else:
            print(f"Warning: {stem_name} file not found: {stem_file}")
    
    return stem_files

def detect_onsets_for_audio(input_audio_path, output_dir):
    """Perform onset detection on specified audio file"""
    print(f"Processing audio file: {input_audio_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find corresponding stem files
    try:
        stem_files = find_stem_files(input_audio_path)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    if not stem_files:
        print("Error: No stem files found")
        return
    
    print(f"Found {len(stem_files)} stem files")
    
    # Detection function mapping
    detection_functions = {
        'kick': detect_kick_onsets,
        'snare': detect_snare_onsets,
        'toms': detect_toms_onsets,
        'hi_hats': detect_hi_hats_onsets,
        'cymbals': detect_cymbals_onsets
    }
    
    # Collect all detection results
    all_onsets = []
    
    for stem_name, stem_file in stem_files.items():
        print(f"  Processing {stem_name}: {stem_file}")
        
        # Get parameters
        params = DETECTION_PARAMS[stem_name]
        
        # Detect onsets
        if stem_name == 'toms':
            onsets = detection_functions[stem_name](
                stem_file,
                threshold=params['threshold'],
                min_separation=params['min_separation'],
                cutoff_low=params['cutoff_low'],
                cutoff_high=params['cutoff_high']
            )
        else:
            onsets = detection_functions[stem_name](
                stem_file,
                threshold=params['threshold'],
                min_separation=params['min_separation'],
                cutoff=params['cutoff']
            )
        
        print(f"    Detected {len(onsets)} onsets")
        
        # Add to results
        for onset_time in onsets:
            all_onsets.append({
                'timestamp': onset_time,
                'drum_type': STEM_MAPPING[stem_name]
            })
    
    # Sort by time
    all_onsets.sort(key=lambda x: x['timestamp'])
    
    # Output transcription results
    output_filename = Path(input_audio_path).stem + "_transcription.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        for onset in all_onsets:
            f.write(f"{onset['timestamp']:.6f}\t{onset['drum_type']}\n")
    
    print(f"Transcription results saved to: {output_path}")
    print(f"Total detected {len(all_onsets)} onsets")
    
    # Display statistics
    drum_counts = defaultdict(int)
    for onset in all_onsets:
        drum_counts[onset['drum_type']] += 1
    
    print("\nDrum type statistics:")
    for drum_type, count in sorted(drum_counts.items()):
        print(f"  {drum_type}: {count}")

def main():
    parser = argparse.ArgumentParser(description='Single Audio Onset Detection Transcription')
    parser.add_argument('--input_audio', type=str, required=True,
                       help='Input audio file path (e.g., /path/to/MusicDelta_Beatles_MIX_from_0.wav)')
    parser.add_argument('--output_dir', type=str, default='./transcription_results',
                       help='Output directory (default: ./transcription_results)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input_audio):
        print(f"Error: Input file does not exist: {args.input_audio}")
        return
    
    # Execute onset detection
    detect_onsets_for_audio(args.input_audio, args.output_dir)

if __name__ == "__main__":
    main()