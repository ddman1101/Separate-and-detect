#!/usr/bin/env python3
"""
What is integrated_train.py ?
1. Can be used for training the MSG-LD with multi-dataset (StemGMD + IDMT)
2. Can be used for the MSG-LD checkpoint with MDB drums dataset's inference (Only source separation)
3. Can be used for the MSG-LD checkpoint with ENST-drums-mixed dataset's inference (Only source separation)

How to use ?
(Training) 1. python integrated_train.py --config config/MSG-LD/integrated_musicldm_multidataset.yaml
(Training) 2. python integrated_train.py --config config/MSG-LD/integrated_musicldm_onset.yaml
(Training) 3. python integrated_train.py --config config/MSG-LD/integrated_musicldm_onset_timbre.yaml
(Inference) 2. python integrated_train.py --config config/MSG-LD/integrated_musicldm_mdb_inference.yaml
(Inference) 3. python integrated_train.py --config config/MSG-LD/integrated_musicldm_enst_inference.yaml
"""

import sys
import os
import argparse
import yaml
import json
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
import shutil
import glob
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.strategies import DDPStrategy
import datetime
import time
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from latent_diffusion.models.musicldm import MusicLDM_multitask
from latent_diffusion.util import instantiate_from_config
from src.utilities.data.datamodule import DataModuleFromConfig
from utilities.tools import listdir_nohidden, get_restore_step

from matplotlib.colors import ListedColormap
from mir_eval import onset as onset_eval

import os, torch, torch.multiprocessing as mp
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
os.environ.setdefault("NUMBA_NUM_THREADS","1")
os.environ.setdefault("PYTORCH_MP_START_METHOD","spawn")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

def _idx_to_sec(idx, fps): 
    return np.asarray(idx, dtype=float) / float(fps)

def _mir_eval_scores(pred_idx, gt_idx, fps, windows=(0.2, 0.3, 0.5)):
    ref = _idx_to_sec(gt_idx, fps)
    est = _idx_to_sec(pred_idx, fps)
    out = {}
    if len(ref) == 0 and len(est) == 0:
        for w in windows:
            out[w] = dict(F1=0.0, P=0.0, R=0.0)
        return out
    for w in windows:
        f1, p, r = onset_eval.f_measure(ref, est, window=w)
        out[w] = dict(F1=float(f1), P=float(p), R=float(r))
    return out

def sweep_and_plot_topk_pianoroll(
    pred_onset, gt_onset, save_path,
    input_is_logits=False,
    fps=100,
    stem_names=('Kick','Snare','Toms','Hi-Hats','Cymbals'),
    min_ms=(70,60,60,35,45),
    quantiles=(0.90,0.92,0.94,0.96,0.97,0.98,0.985,0.99,0.992,0.994,0.996,0.998),
    locmax_win=5,
    smooth_win=1,
    tol_frames=2
):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    try:
        from matplotlib.colors import ListedColormap
        _BW_CMAP = ListedColormap(['white','black'])
    except Exception:
        _BW_CMAP = 'binary'

    # ---------- helpers ----------
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

    def _robust_z(x):
        med = np.median(x); mad = np.median(np.abs(x - med)) + 1e-8
        return (x - med) / mad

    def _sliding_max(x, k):
        k = k if k % 2 == 1 else k + 1
        pad = k // 2
        xpad = np.pad(x, (pad, pad), mode="edge")
        view = np.lib.stride_tricks.sliding_window_view(xpad, k)
        return view.max(axis=-1)

    def _local_max_mask(x, k):
        return x >= _sliding_max(x, k)

    def _apply_refractory(cand_idx, wait):
        out, last = [], -10**9
        for t in cand_idx:
            if t - last >= wait:
                out.append(t); last = t
        return np.asarray(out, dtype=int)

    def _events_from_curve(x, z, q, wait, k):
        thr = np.quantile(z, q)
        mask = (z > thr) & _local_max_mask(x, k)
        cand_idx = np.where(mask)[0]
        pred_idx = _apply_refractory(cand_idx, wait)
        return pred_idx, thr

    def _match_with_tolerance(pred_idx, gt_idx, tol):
        if len(gt_idx) == 0:
            return 0, len(pred_idx), 0
        used = np.zeros(len(gt_idx), dtype=bool)
        tp = 0
        for p in pred_idx:
            lo = np.searchsorted(gt_idx, p - tol, side='left')
            hi = np.searchsorted(gt_idx, p + tol, side='right')
            best_j, best_d = -1, 1e9
            for j in range(lo, hi):
                if not used[j]:
                    d = abs(gt_idx[j] - p)
                    if d < best_d:
                        best_d, best_j = d, j
            if best_j >= 0:
                used[best_j] = True
                tp += 1
        fp = len(pred_idx) - tp
        fn = len(gt_idx) - tp
        return tp, fp, fn

    def _metrics(tp, fp, fn, eps=1e-8):
        P = tp / (tp + fp + eps); R = tp / (tp + fn + eps)
        F1 = 2 * P * R / (P + R + eps)
        return P, R, F1

    def _roll_from_idx(idx, T):
        r = np.zeros(T, dtype=np.float32)
        if len(idx) > 0: r[np.clip(idx, 0, T-1)] = 1.0
        return r

    # ---------- data prep ----------
    pred_onset = _to_np(pred_onset).astype(np.float32)
    gt_onset   = _to_np(gt_onset).astype(np.float32)
    if pred_onset.ndim == 3:
        pred_onset = pred_onset[0]
    assert pred_onset.shape == gt_onset.shape and pred_onset.ndim == 2, \
        f"expect (S,T), got {pred_onset.shape} vs {gt_onset.shape}"

    S, T = gt_onset.shape
    probs = _sigmoid(pred_onset) if input_is_logits else pred_onset
    probs = np.clip(probs, 0.0, 1.0)

    if smooth_win and smooth_win > 1:
        k = np.ones(smooth_win, dtype=np.float32) / smooth_win
        probs = np.stack([np.convolve(probs[s], k, mode='same') for s in range(S)], axis=0)

    if isinstance(min_ms, (list, tuple, np.ndarray)):
        assert len(min_ms) == S, f"min_ms length {len(min_ms)} != stems {S}"
        waits = [max(1, int(round(m * fps / 1000.0))) for m in min_ms]
    else:
        w = max(1, int(round(min_ms * fps / 1000.0)))
        waits = [w] * S

    gt_idx_all = [np.where(gt_onset[s] > 0.5)[0] for s in range(S)]

    # ---------- sweep per stem & build full 5×T ----------
    best_list = []
    pred_full = np.zeros_like(gt_onset, dtype=np.float32)

    for s in range(S):
        x  = probs[s]
        z  = _robust_z(x)
        wait = waits[s]
        gt_idx = gt_idx_all[s]

        windows = (0.2, 0.3, 0.5)
        primary_window = 0.3 
        
        q_candidates = tuple(sorted(set(quantiles) | {1.0}))
        best = {"F1": -1.0, "pred_idx": np.array([], dtype=int)}
        for q in q_candidates:
            pred_idx, thr = _events_from_curve(x, z, q, wait, k=locmax_win)
            m_all = _mir_eval_scores(pred_idx, gt_idx, fps=fps, windows=windows)
            F1_key = m_all[primary_window]["F1"]     
            # When F1 is the same, prefer fewer peaks to avoid picking peaks on stems with no events
            if (F1_key > best["F1"] + 1e-6) or (abs(F1_key - best["F1"]) <= 1e-6 and len(pred_idx) < len(best["pred_idx"])):
                best = dict(
                    s=s, stem=(stem_names[s] if s < len(stem_names) else f"Stem{s}"),
                    q=float(q), thr_z=float(thr),
                    metrics=m_all,
                    pred_idx=np.asarray(pred_idx), gt_idx=np.asarray(gt_idx),
                    F1=float(F1_key)
                )
        

        best_list.append(best)
        pred_full[s] = _roll_from_idx(best["pred_idx"], T)

    print(f"==== Best per-stem (mir_eval.onset, windows={windows}, primary={primary_window}s) ====")
    for b in best_list:
        m = b["metrics"]
        line = (f"{b['stem']:8s} | q={b['q']:.3f} thr_z={b['thr_z']:.2f} | "
                f"F1@0.2={m[0.2]['F1']:.3f} P={m[0.2]['P']:.3f} R={m[0.2]['R']:.3f} | "
                f"F1@0.3={m[0.3]['F1']:.3f} P={m[0.3]['P']:.3f} R={m[0.3]['R']:.3f} | "
                f"F1@0.5={m[0.5]['F1']:.3f} P={m[0.5]['P']:.3f} R={m[0.5]['R']:.3f}")
        print(line)

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), squeeze=True)
    axes[0].imshow((gt_onset > 0.5).astype(np.float32), cmap=_BW_CMAP,
                   aspect='auto', vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_title('Ground Truth (S × T)')
    axes[1].imshow(pred_full, cmap=_BW_CMAP,
                   aspect='auto', vmin=0, vmax=1, interpolation='nearest')
    subtitle = " | ".join([f"{b['stem']}:F1={b['F1']:.3f},q={b['q']:.3f}" for b in best_list])
    axes[1].set_title(f'Prediction (per-stem best q, locmax={locmax_win}, wait={waits})\n{subtitle}')

    for ax in axes:
        ax.set_yticks(range(S))
        ax.set_yticklabels(stem_names[:S])
        ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Full pianoroll saved to: {save_path}")

    per_stem = []
    for b in best_list:
        m = b["metrics"][primary_window]
        per_stem.append({
            "stem": b["stem"],
            "F1": float(m["F1"]),
            "P": float(m["P"]),
            "R": float(m["R"])
        })
    macro_f1 = float(np.mean([x["F1"] for x in per_stem])) if per_stem else 0.0
    return {
        "per_stem": per_stem,
        "macro_F1": macro_f1,
        "primary_window": primary_window
    }

class IntegratedPianorollVisualizationCallback(Callback):
    """Integrated pianoroll visualization callback - supports original MSG-LD and onset_only modes"""
    
    def __init__(self, save_dir, sample_interval=1, thresholds=[0.5, 0.3, 0.1], num_samples=2):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sample_interval = sample_interval
        self.thresholds = thresholds
        self.stem_names = ['Kick', 'Snare', 'Toms', 'Hi-Hats', 'Cymbals']
        self.num_samples = num_samples
        self._reset_epoch_accum()
        # For whole-validation re-evaluation with a single best threshold per stem: collect all samples
        self._epoch_store = []  # each element: dict(pred=[S,T], gt=[S,T])

    def _reset_epoch_accum(self):
        self.accum = {
            "count": 0,
            "stem_sums": {name: {"F1": 0.0, "P": 0.0, "R": 0.0} for name in self.stem_names},
            "macro_f1_sum": 0.0
        }

    def on_validation_epoch_start(self, trainer, pl_module):
        self._reset_epoch_accum()
        self._epoch_store = []
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Called after each validation batch"""
        if batch_idx % self.sample_interval != 0:
            return
        
        # Check whether it's onset_only mode or onset prediction is enabled
        has_onset_prediction = (
            hasattr(pl_module, 'training_mode') and pl_module.training_mode == "onset_only"
        ) or (
            hasattr(pl_module, 'enable_onset_prediction') and pl_module.enable_onset_prediction
        )
        
        if not has_onset_prediction:
            return
        
        try:
            saved_any = False
            # Try to fetch track_id information from batch
            track_ids = None
            if isinstance(batch, dict):
                for key in ['track_id', 'track_ids', 'id', 'ids', 'filename', 'filenames']:
                    if key in batch:
                        track_ids = batch[key]
                        break
                
                print(f"\n=== Current Batch Info ===")
                print(f"Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'Not a dict'}")
                if 'track_id' in batch:
                    print(f"Track IDs in batch: {batch['track_id']}")
                if 'onset_json_path' in batch:
                    print(f"Onset JSON paths in batch: {batch['onset_json_path']}")
                print("=" * 50)
            
            # Get predictions
            pred_is_logits = True
            pred_onset_logits = None
            gt_onset = None
            if isinstance(outputs, dict):
                # 1) Directly provided pred_onset / gt_onset
                if 'pred_onset' in outputs:
                    pred_onset_logits = outputs['pred_onset']
                    gt_onset = outputs.get('gt_onset', None)
                    pred_is_logits = True
                # 2) In 'onset_data' (returned by validation_step)
                elif 'onset_data' in outputs and isinstance(outputs['onset_data'], dict):
                    onset_data = outputs['onset_data']
                    if 'pred_onset' in onset_data and 'gt_onset' in onset_data:
                        # Here pred_onset is already sigmoid probabilities
                        pred_onset_logits = onset_data['pred_onset']
                        gt_onset = onset_data['gt_onset']
                        pred_is_logits = False

            # Fallback: at least require GT
            if gt_onset is None and isinstance(batch, dict) and 'onset_pianoroll' in batch:
                gt_onset = batch['onset_pianoroll']

            # If still missing either, skip plotting
            if pred_onset_logits is None or gt_onset is None:
                return
            
            # Ensure tensors are on CPU
            if isinstance(pred_onset_logits, torch.Tensor):
                if pred_is_logits:
                    pred_onset = torch.sigmoid(pred_onset_logits)
                else:
                    pred_onset = pred_onset_logits
                pred_onset = pred_onset.detach().cpu().numpy()
                pred_onset_logits = pred_onset_logits.detach().cpu().numpy()
            if isinstance(gt_onset, torch.Tensor):
                gt_onset = gt_onset.detach().cpu().numpy()
            
            # Determine batch size and choose samples
            batch_size = pred_onset.shape[0]
            if batch_size > 0:
                num_samples_to_process = min(self.num_samples, batch_size)
                if batch_size >= self.num_samples:
                    sample_indices = list(range(num_samples_to_process))
                else:
                    sample_indices = list(range(batch_size))
                
                print(f"==== Processing {len(sample_indices)} samples from batch (batch_size={batch_size}) ====")
                
                # Process each selected sample
                for sample_idx in sample_indices:
                    # Get this sample's track_id
                    track_id_info = ""
                    if track_ids is not None:
                        try:
                            if isinstance(track_ids, (list, tuple)) and sample_idx < len(track_ids):
                                track_id = track_ids[sample_idx]
                            elif hasattr(track_ids, '__getitem__'):
                                track_id = track_ids[sample_idx]
                            else:
                                track_id = track_ids
                            
                            if isinstance(track_id, str):
                                track_id_info = f" (Track: {track_id})"
                            elif isinstance(track_id, (int, float)):
                                track_id_info = f" (Track: {int(track_id)})"
                            else:
                                track_id_info = f" (Track: {str(track_id)})"
                        except (IndexError, TypeError):
                            track_id_info = ""
                    
                    print(f"\n--- Sample {sample_idx}{track_id_info} ---")
                    
                    # Extract selected sample
                    pred_sample = pred_onset[sample_idx]  # [num_stems, time_steps]
                    pred_sample_logits = pred_onset_logits[sample_idx]
                    gt_sample = gt_onset[sample_idx]      # [num_stems, time_steps]
                    
                    # Print stats for each stem
                    print("==== Onset Probabilities per Stem ====")
                    for i, stem_name in enumerate(self.stem_names):
                        if i < pred_sample_logits.shape[0]:
                            stem_logits = pred_sample_logits[i, :].flatten()
                            stem_probs = pred_sample[i, :].flatten()
                            
                            print(f'{stem_name:8s} | Before sigmoid: min={float(stem_logits.min()):.3f} max={float(stem_logits.max()):.3f} '
                                  f'p25={float(np.quantile(stem_logits,0.25)):.3f} p50={float(np.quantile(stem_logits,0.5)):.3f} p75={float(np.quantile(stem_logits,0.75)):.3f}')
                            print(f'{stem_name:8s} | After sigmoid:  min={float(stem_probs.min()):.3f} max={float(stem_probs.max()):.3f} '
                                  f'p25={float(np.quantile(stem_probs,0.25)):.3f} p50={float(np.quantile(stem_probs,0.5)):.3f} p75={float(np.quantile(stem_probs,0.75)):.3f}')
                    
                    # Create filename part with track_id
                    track_id_str = ""
                    if track_ids is not None:
                        try:
                            if isinstance(track_ids, (list, tuple)) and sample_idx < len(track_ids):
                                track_id = track_ids[sample_idx]
                            elif hasattr(track_ids, '__getitem__'):
                                track_id = track_ids[sample_idx]
                            else:
                                track_id = track_ids
                            
                            if isinstance(track_id, str):
                                track_id_str = f"_track_{track_id}"
                            elif isinstance(track_id, (int, float)):
                                track_id_str = f"_track_{int(track_id)}"
                            else:
                                track_id_str = f"_track_{str(track_id)}"
                        except (IndexError, TypeError) as e:
                            print(f"Warning: Could not extract track_id for sample {sample_idx}: {e}")
                            track_id_str = ""
                    
                    # Build save path; append fname for identifying source
                    fname_str = ""
                    try:
                        if isinstance(batch, dict) and 'fname' in batch:
                            fname_raw = batch['fname']
                            if isinstance(fname_raw, (list, tuple)) and sample_idx < len(fname_raw):
                                fname_str = f"_{fname_raw[sample_idx]}"
                            elif isinstance(fname_raw, str):
                                fname_str = f"_{fname_raw}"
                    except Exception:
                        fname_str = ""
                    save_path = self.save_dir / f"integrated_pianoroll_epoch_{trainer.current_epoch}_batch_{batch_idx}_sample_{sample_idx}{track_id_str}{fname_str}.png"
                    print(f"save_path: {save_path}")
                    
                    # Plot comparison and print F1/P/R (printed inside sweep)
                    metrics = sweep_and_plot_topk_pianoroll(
                        pred_sample_logits, gt_sample,
                        save_path=save_path,
                        input_is_logits=pred_is_logits,
                        fps=100, tol_frames=2
                    )
                    saved_any = True

                    # Store for epoch-level re-evaluation with global best threshold per stem
                    try:
                        self._epoch_store.append({
                            "pred": pred_sample_logits.copy(),  # [S,T]
                            "gt": gt_sample.copy()              # [S,T]
                        })
                    except Exception:
                        pass

                    # Aggregate per-stem metrics for primary window
                    try:
                        self.accum["count"] += 1
                        self.accum["macro_f1_sum"] += float(metrics.get("macro_F1", 0.0))
                        for s in metrics.get("per_stem", []):
                            name = s["stem"]
                            if name in self.accum["stem_sums"]:
                                for k in ["F1","P","R"]:
                                    self.accum["stem_sums"][name][k] += float(s[k])
                    except Exception:
                        pass
        except Exception as e:
            print(f"Pianoroll visualization failed: {e}")
            import traceback
            traceback.print_exc()
        else:
            # Fallback: if nothing was saved this round, try copying images produced in validation_step
            if not saved_any:
                try:
                    if hasattr(pl_module, 'get_log_dir'):
                        src_dir = Path(pl_module.get_log_dir()) / 'onset_visualization_val'
                        if src_dir.exists():
                            for img in glob.glob(str(src_dir / '*.png')):
                                dst = self.save_dir / Path(img).name
                                if not dst.exists():
                                    shutil.copy2(img, dst)
                except Exception as e:
                    print(f"Fallback copy of onset_visualization_val failed: {e}")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Compute averages over the whole validation
        n = max(1, self.accum["count"])
        macro_f1_mean = self.accum["macro_f1_sum"] / n
        print(f"=== Validation mean (primary window) over {n} samples ===")
        print(f"Macro-F1: {macro_f1_mean:.3f}")
        for name in self.stem_names:
            s = self.accum["stem_sums"][name]
            print(f"{name:8s} | F1={s['F1']/n:.3f} P={s['P']/n:.3f} R={s['R']/n:.3f}")
        # Additionally: search a single best threshold per stem over the whole validation, then compute final Macro-F1
        try:
            if len(self._epoch_store) > 0:
                import numpy as np
                from copy import deepcopy
                fps = 100
                tol = 2
                S = self._epoch_store[0]["gt"].shape[0]
                # Stack all samples
                all_pred = np.concatenate([x["pred"][None] for x in self._epoch_store], axis=0)  # [N,S,T]
                all_gt   = np.concatenate([x["gt"][None] for x in self._epoch_store], axis=0)    # [N,S,T]

                def _mir_eval_scores_idx(pred_idx, gt_idx, fps, windows=(0.3,)):
                    from mir_eval import onset as onset_eval
                    ref = np.asarray(gt_idx, dtype=float) / float(fps)
                    est = np.asarray(pred_idx, dtype=float) / float(fps)
                    out = {}
                    for w in windows:
                        f1,p,r = onset_eval.f_measure(ref, est, window=w)
                        out[w] = dict(F1=float(f1),P=float(p),R=float(r))
                    return out

                def _local_max_mask(x, k=5):
                    pad = k//2
                    xpad = np.pad(x,(pad,pad),mode="edge")
                    view = np.lib.stride_tricks.sliding_window_view(xpad,k)
                    return x >= view.max(axis=-1)

                def _pred_idx_for_thr(x, z, thr, k=5, wait=1):
                    # Important: threshold thr is from z-score domain; compare on z,
                    # but local-max detection remains on the original probability x.
                    mask = (z>thr) & _local_max_mask(x,k)
                    cand = np.where(mask)[0]
                    # refractory
                    out=[]; last=-10**9
                    for t in cand:
                        if t-last>=wait:
                            out.append(t); last=t
                    return np.asarray(out,dtype=int)

                # For each stem, find a single best threshold (0.3s window)
                best_thr = [0.5]*S
                per_stem_metrics = []
                waits = [max(1,int(round(m*fps/1000.0))) for m in (70,60,60,35,45)] if S>=5 else [3]*S
                for s in range(S):
                    probs = 1/(1+np.exp(-all_pred[:,s,:]))  # logits->prob
                    gt = all_gt[:,s,:] > 0.5
                    T = probs.shape[-1]
                    # Scan the same threshold across all samples and aggregate
                    qs = np.linspace(0.80,0.999,25)
                    best=(None,-1.0,{})
                    z = (probs - np.median(probs,axis=1,keepdims=True)) / (np.median(np.abs(probs-np.median(probs,axis=1,keepdims=True)),axis=1,keepdims=True)+1e-8)
                    for q in qs:
                        # For each sample, get pred_idx and accumulate TP/FP/FN
                        TP=FP=FN=0
                        for i in range(all_pred.shape[0]):
                            thr=np.quantile(z[i],q)
                            pred_idx=_pred_idx_for_thr(probs[i], z[i], thr, k=5, wait=waits[s])
                            gt_idx=np.where(gt[i])[0]
                            # Matching with tolerance
                            used=np.zeros(len(gt_idx),dtype=bool)
                            tp=0
                            for p in pred_idx:
                                lo=np.searchsorted(gt_idx,p-tol,'left'); hi=np.searchsorted(gt_idx,p+tol,'right')
                                best_j=-1; best_d=1e9
                                for j in range(lo,hi):
                                    if not used[j]:
                                        d=abs(gt_idx[j]-p)
                                        if d<best_d:
                                            best_d=d; best_j=j
                                if best_j>=0:
                                    used[best_j]=True; tp+=1
                            fp=len(pred_idx)-tp; fn=len(gt_idx)-tp
                            TP+=tp; FP+=fp; FN+=fn
                        P=TP/(TP+FP+1e-8); R=TP/(TP+FN+1e-8); F1=2*P*R/(P+R+1e-8)
                        if F1>best[1]+1e-6:
                            best=(q,F1,{"F1":F1,"P":P,"R":R})
                    best_thr[s]=best[0] if best[0] is not None else 0.5
                    per_stem_metrics.append(best[2])
                macro_F1_final=float(np.mean([m["F1"] for m in per_stem_metrics])) if per_stem_metrics else 0.0
                print("=== Validation re-eval with single best threshold per stem (0.3s window) ===")
                print(f"Macro-F1(final): {macro_F1_final:.3f}")
                for s,name in enumerate(self.stem_names[:S]):
                    m=per_stem_metrics[s]
                    print(f"{name:8s} | F1={m['F1']:.3f} P={m['P']:.3f} R={m['R']:.3f} q*={best_thr[s]:.3f}")
        except Exception as e:
            print(f"Re-eval failed: {e}")
        
        # Save to JSON
        try:
            out = {
                "num_samples": n,
                "macro_F1": macro_f1_mean,
                "per_stem": {name: {k: v/n for k,v in self.accum["stem_sums"][name].items()} for name in self.stem_names}
            }
            with open(self.save_dir / f"mean_metrics_epoch_{trainer.current_epoch}.json", "w") as f:
                json.dump(out, f, indent=2)
        except Exception:
            pass

def load_config(config_path):
    """
    Load configuration file
    """
    if not config_path:
        # Use default config file
        config_path = "config/integrated_musicldm_onset.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Integrated training script")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to existing config file (optional)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size"
    )
    parser.add_argument(
        "--visualize_pianoroll",
        action="store_true",
        help="Enable pianoroll visualization"
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1,
        help="Visualization sampling interval"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development mode"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--separate_only",
        action="store_true",
        help="Separation only: optionally run demucs first, then use current checkpoint to generate separated outputs per batch without evaluation"
    )
    
    args = parser.parse_args()
    
    print(f"=== Start integrated training ===")
    
    # Set random seed
    seed_everything(42)
    
    # Load config
    config = load_config(args.config)
    
    # Update config
    config["trainer"]["max_epochs"] = args.epochs
    config["data"]["params"]["batch_size"] = args.batch_size
    config["model"]["params"]["batchsize"] = args.batch_size
    config["dev"] = args.dev
    
    print(f"Config:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Pianoroll visualization: {args.visualize_pianoroll}")
    if args.visualize_pianoroll:
        print(f"  - Sample interval: {args.sample_interval}")
    
    # Create data module
    print("\nCreating data module...")
    if not args.separate_only:
        data = instantiate_from_config(config["data"])
        data.prepare_data()
        data.setup()
    
    # Set log directory
    log_path = config["log_directory"]
    os.makedirs(log_path, exist_ok=True)
    
    # Generate experiment name
    random_seconds_shift = datetime.timedelta(seconds=np.random.randint(60))
    now = (datetime.datetime.now() - random_seconds_shift).strftime('%Y-%m-%dT%H-%M-%S')
    
    if args.dev:
        nowname = f"{now}_joint_training"
    else:
        nowname = f"{now}_joint_training"
    
    run_path = os.path.join(log_path, config["project_name"], nowname)
    os.makedirs(run_path, exist_ok=True)
    
    print(f"Experiment name: {nowname}")
    print(f"Run path: {run_path}")
    
    # Create logger (optional)
    logger = None
    if args.use_wandb:
        if args.dev:
            os.environ["WANDB_MODE"] = "offline"
        logger = WandbLogger(
            save_dir=run_path,
            project=config["project_name"],
            name=nowname,
            log_model=False,
            settings=wandb.Settings(start_method="thread")
        )
    
    # Create checkpoint callback (monitor metric depends on task)
    model_params = config.get("model", {}).get("params", {})
    unet_params = model_params.get("unet_config", {}).get("params", {})
    use_onset = bool(model_params.get("enable_onset_prediction", False)) or bool(unet_params.get("use_onset_branch", False))

    if use_onset:
        monitor_key = "val/onset_loss"
        monitor_mode = "min"
        filename_metric = "val_onset_loss"
    else:
        # If onset branch is disabled, use SDR_kick as monitor (higher is better)
        monitor_key = "val/stem_0/SDR"
        monitor_mode = "max"
        # Note: the metric in filename should use underscores
        filename_metric = "val_stem_0_SDR"

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_path, "checkpoints"),
        filename=f"integrated-{{epoch:02d}}-{{{filename_metric}:.4f}}",
        save_top_k=config["trainer"]["save_top_k"],
        monitor=monitor_key,
        mode=monitor_mode,
        save_last=True,  # save last checkpoint as last.ckpt
        save_on_train_epoch_end=False  # save only at validation end based on monitor
    )
    
    # Create callbacks list
    callbacks = [checkpoint_callback]
    
    # Add an extra callback to save a checkpoint every epoch
    every_epoch_callback = ModelCheckpoint(
        dirpath=os.path.join(run_path, "checkpoints"),
        filename="epoch-{epoch:02d}",
        save_top_k=-1,  # keep all epochs
        every_n_epochs=1,  # save every epoch
        save_last=True,
        save_on_train_epoch_end=True  # save at each epoch end
    )
    callbacks.append(every_epoch_callback)
    
    # If enabled, add pianoroll visualization callback
    if args.visualize_pianoroll:
        viz_callback = IntegratedPianorollVisualizationCallback(
            save_dir=os.path.join(run_path, "pianoroll_visualizations"),
            sample_interval=args.sample_interval,
            thresholds=[0.62,0.6,0.58,0.56,0.54],
            num_samples=2
        )
        callbacks.append(viz_callback)
        print(f"Integrated Pianoroll visualization callback added")

    # (Optional) MDB transcription evaluation (5-stem separation + onset detection). Paths can be provided via YAML or here.
    # stems_root should point to a folder structure of five stems. Each song has a subfolder with kick/snare/toms/hi_hats/cymbals.wav
    mdb_cfg = config.get('mdb_eval', {})
    if mdb_cfg:
        stems_root = mdb_cfg.get('stems_root', None)
        gt_onsets_root = mdb_cfg.get('gt_onsets_root', None)
        run_demucs = mdb_cfg.get('run_demucs', False)
        demucs_in = mdb_cfg.get('demucs_input_root', None)
        demucs_out = mdb_cfg.get('demucs_output_root', None)
        demucs_model = mdb_cfg.get('demucs_model', 'htdemucs')
        demucs_two = mdb_cfg.get('demucs_two_stems', 'drums')

    
    # Create model
    print("Creating model...")
    model = instantiate_from_config(config["model"])
    model.set_log_dir(run_path, config["project_name"], nowname)
    # Decide whether to resume from a specified checkpoint
    resume_from_checkpoint = config.get("trainer", {}).get("resume_from_checkpoint", None)
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Resume from checkpoint: {resume_from_checkpoint}")
    else:
        print("Train from scratch")
        resume_from_checkpoint = None
    # Create trainer
    trainer = Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        num_sanity_val_steps=0,
        logger=logger,
        limit_val_batches=config["trainer"]["limit_val_batches"],
        limit_train_batches=config["trainer"]["limit_train_batches"],
        val_check_interval=config["trainer"]["val_check_interval"],
        strategy=DDPStrategy(
            find_unused_parameters=False,
            static_graph=True,
            broadcast_buffers=False
        ) if len(config["trainer"]["devices"]) > 1 else None,
        callbacks=callbacks,
        precision=config["trainer"]["precision"],
        fast_dev_run=args.dev,
        # DataLoader related settings
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=True,
        # Shorter timeouts
        sync_batchnorm=False,
        # Avoid DataLoader issues
        replace_sampler_ddp=True,
    )
    if args.separate_only:
        # Separation-only: do not run evaluation
        # 1) Check if valid_data path exists, if not, run demucs first
        data = instantiate_from_config(config["data"])
        valid_data_path = data.path.get('valid_data')
        
        if valid_data_path and not os.path.exists(valid_data_path):
            print(f"[SeparateOnly] valid_data path not found: {valid_data_path}")
            print("[SeparateOnly] Running Demucs separation first...")
            
            mdb_cfg = config.get('mdb_eval', {}) or {}
            if mdb_cfg.get('run_demucs', False) and mdb_cfg.get('demucs_input_root') and mdb_cfg.get('demucs_output_root'):
                try:
                    Path(mdb_cfg['demucs_output_root']).mkdir(parents=True, exist_ok=True)
                    
                    # Prepare input file list (supports directory or single file)
                    demucs_in = str(mdb_cfg['demucs_input_root'])
                    inputs = []
                    if os.path.isdir(demucs_in):
                        exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
                        for root, _, files in os.walk(demucs_in):
                            for fn in files:
                                if fn.lower().endswith(exts):
                                    inputs.append(os.path.join(root, fn))
                        inputs.sort()
                    else:
                        inputs = [demucs_in]
                    
                    if not inputs:
                        print(f"[SeparateOnly] demucs inputs empty under: {demucs_in}")
                    else:
                        cmd = [
                            'demucs', '-n', str(mdb_cfg.get('demucs_model', 'htdemucs')),
                            '-o', str(mdb_cfg['demucs_output_root'])
                        ]
                        if mdb_cfg.get('demucs_two_stems'):
                            cmd += ['--two-stems', str(mdb_cfg.get('demucs_two_stems'))]
                        if mdb_cfg.get('demucs_device'):
                            cmd += ['--device', str(mdb_cfg['demucs_device'])]
                        cmd += inputs
                        print(f"[SeparateOnly] running demucs: {' '.join(cmd[:8])} ... (+{len(inputs)} files)")
                        result = subprocess.run(cmd, check=False)
                        if result.returncode == 0:
                            print("[SeparateOnly] Demucs separation completed successfully")
                        else:
                            print(f"[SeparateOnly] Demucs failed with return code: {result.returncode}")
                except Exception as e:
                    print(f"[SeparateOnly] demucs failed: {e}")
            else:
                print("[SeparateOnly] Demucs configuration not found or incomplete")
        else:
            print(f"[SeparateOnly] valid_data path exists: {valid_data_path}")

        # 2) Run validate once with current checkpoint, but only collect samples output during validation_step (target_*/val_* in log_dir)
        #    You can control data source via DataModule path options (e.g., set valid_data to demucs_output_root)
        data.prepare_data()
        data.setup()
        print("[SeparateOnly] start model separation (no eval)...")
        trainer.validate(model, data, ckpt_path=resume_from_checkpoint)
    else:
        print("\nStart training...")
        trainer.fit(model, data)
    
    print(f"\n=== Integrated training finished ===")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    
    if args.visualize_pianoroll:
        viz_dir = os.path.join(run_path, "pianoroll_visualizations")
        print(f"Pianoroll visualizations saved at: {viz_dir}")

if __name__ == "__main__":
    main() 