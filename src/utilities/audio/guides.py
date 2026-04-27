#!/usr/bin/env python3
"""
Guide-map feature computation module.
Computes from mel-spectrogram: spectral flux, HPSS percussive mask, high-frequency band map, low-frequency band map.
"""

import numpy as np
import librosa
from scipy.ndimage import median_filter


def _minmax(x, eps=1e-8):
    """Normalize to 0~1 range"""
    vmin, vmax = np.min(x), np.max(x)
    return (x - vmin) / (vmax - vmin + eps)


def compute_guides_from_mel(
    mel_power,               # [M, T] linear mel power (power=2.0)
    sr=16000,
    hop_length=160,
    mel_fmin=0.0,
    mel_fmax=8000.0,
    F_med=17,                # percussive (frequency-direction) kernel
    T_med=9,                 # harmonic (time-direction) kernel
    p_mask=2,                # mask exponent
    hf_threshold=6000,       # high-frequency threshold (Hz)
    lf_threshold=200,        # low-frequency threshold (Hz)
    return_db_for_plot=False
):
    """
    Compute guide-map features from mel-spectrogram.
    
    Returns:
    guides_raw: np.ndarray [Cg, M, T], values in 0~1
    film_vec: np.ndarray [D_feat] (can use global average over Cg; feed to MLP in actual use)
    plots: optional dict with mel_db/hf_db/lf_db for plotting
    """
    M, T = mel_power.shape
    mel_db = librosa.power_to_db(mel_power + 1e-10, ref=np.max)
    
    # 1) Spectral flux (difference on dB + ReLU)
    mel_diff = np.diff(mel_db, axis=1, prepend=mel_db[:, :1])
    spectral_flux = np.maximum(mel_diff, 0.0)
    spectral_flux = _minmax(spectral_flux)  # [M,T]
    
    # 2) HPSS approximation: directional median filtering for percussive/harmonic
    P = median_filter(mel_power, size=(F_med, 1), mode='reflect')  # percussive-like
    H = median_filter(mel_power, size=(1, T_med), mode='reflect')  # harmonic-like
    Pp, Hp = P**p_mask, H**p_mask
    percussive_mask = Pp / (Pp + Hp + 1e-12)  # 0~1, [M,T]
    percussive_mask = percussive_mask.astype(np.float32)
    
    # 3) HF / LF band map (threshold by actual Hz)
    mel_hz = librosa.mel_frequencies(n_mels=M, fmin=mel_fmin, fmax=mel_fmax)
    hf_mask = (mel_hz >= hf_threshold).astype(float)[:, None]    # [M,1]
    lf_mask = (mel_hz <= lf_threshold).astype(float)[:, None]
    hf_band = mel_power * hf_mask  # [M,T] linear
    lf_band = mel_power * lf_mask
    
    # Normalize to 0~1 (either log or power; here we min-max the power)
    hf_map = _minmax(hf_band)
    lf_map = _minmax(lf_band)
    
    # 4) Combine all features
    guides_list = [
        spectral_flux.astype(np.float32),  # C0
        percussive_mask,                   # C1
        hf_map.astype(np.float32),         # C2
        lf_map.astype(np.float32),         # C3
    ]
    guides_raw = np.stack(guides_list, axis=0)  # [Cg, M, T]
    
    # Simple FiLM conditioning: global mean of each guide-map -> [Cg]
    film_vec = guides_raw.mean(axis=(1, 2)).astype(np.float32)  # [Cg]
    
    plots = None
    if return_db_for_plot:
        plots = {
            'mel_db': mel_db,
            'hf_db': librosa.power_to_db(hf_band + 1e-10, ref=np.max),
            'lf_db': librosa.power_to_db(lf_band + 1e-10, ref=np.max),
        }
    
    return guides_raw, film_vec, plots 