from collections import defaultdict
import pandas as pd
from typing import *
import math
import torchaudio
import torch
from pathlib import Path
import IPython.display as ipd
import numpy as np
from torchmetrics.functional.pairwise import pairwise_euclidean_distance
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

si_sdr = ScaleInvariantSignalDistortionRatio()
si_snr = ScaleInvariantSignalNoiseRatio()

def compute_mse(spectrogram1, spectrogram2):
    """
    Compute the Log-Spectral Distance (LSD) between two Mel spectrograms.

    Parameters:
    spectrogram1 (np.ndarray): The first Mel spectrogram (original).
    spectrogram2 (np.ndarray): The second Mel spectrogram (separated).

    Returns:
    float: The Log-Spectral Distance between the two spectrograms.
    """
    # Ensure the spectrograms have the same shape
    assert spectrogram1.shape == spectrogram2.shape, "Spectrograms must have the same shape."

    # Compute the log-amplitude spectrograms
    log_spectrogram1 = spectrogram1 #np.log1p(spectrogram1)
    log_spectrogram2 = spectrogram2 #np.log1p(spectrogram2)

    # Compute the squared difference between the log spectrograms
    squared_diff = np.square(log_spectrogram1 - log_spectrogram2)

    # Compute the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)

    # Compute the log-spectral distance
    # lsd = np.sqrt(mean_squared_diff)

    return mean_squared_diff


def load_mel_chunks2(chunk_folder: Path, stems: Sequence[str]) -> Tuple[Mapping[str, torch.Tensor], int]:
    separated_tracks = {}
    for idx, stem in enumerate(stems):
        subfolder = chunk_folder / f"stem_mel_{idx}"
        npy_files = sorted(subfolder.glob("*.npy"))
        if npy_files:
            separated_tracks[stem] = [np.load(file) for file in npy_files]  # Load the first file
        else:
            raise FileNotFoundError(f"No .npy files found in {subfolder}")

    return separated_tracks

def evaluate_separations(
    separation_path_mel,
    dataset_path_mel,
    ):

    separation_path_mel = Path(separation_path_mel)
    dataset_path_mel = Path(dataset_path_mel)
    
    si_sdr_dict = []
    mse_dict = []

    npy_files = sorted(separation_path_mel.glob("*.npy"))
    separated_track_mel = [np.load(file) for file in npy_files]  

    npy_files = sorted(dataset_path_mel.glob("*.npy"))
    original_track_mel = [np.load(file) for file in npy_files]  


    num_chunks = len(separated_track_mel)
    for i in range(num_chunks):
        o = original_track_mel[i]
        s = separated_track_mel[i]

        # Calculate MSE and SI-SDR
        mse_value = compute_mse(s, o).item()
        mse_dict.append(mse_value)
        
        o = torch.tensor(o).flatten()
        s = torch.tensor(s).flatten()
        si_sdr_value = si_sdr(s, o).item()
        si_sdr_dict.append(si_sdr_value)

    # Compute averages
    avg_mse = sum(mse_dict) / len(mse_dict)
    avg_si_sdr = sum(si_sdr_dict) / len(si_sdr_dict)

    return {"mse" : avg_mse, "si_sdr": avg_si_sdr}