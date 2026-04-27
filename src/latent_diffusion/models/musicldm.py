"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import sys
import os
from typing import Any, Callable, List, Optional, Union
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
from latent_diffusion.modules.encoders.modules import CLAPResidualVQ
import wandb
from pathlib import Path
from utilities.sep_evaluation import evaluate_separations

# For onset predictor
import torch.nn.init as init
import math

import torch
import torch.nn.functional as F

# Onset detection in the end
# from ddc_onset import FRAME_RATE, compute_onset_salience, find_peaks

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

try:
    from audioldm_eval import EvaluationHelper
except ImportError:
    print("Warning: audioldm_eval not found, some evaluation features may not work")
    EvaluationHelper = None

from latent_diffusion.util import (
    log_txt_as_img,
    exists,
    default,
    ismap,
    isimage,
    mean_flat,
    count_params,
    instantiate_from_config,
)
from latent_diffusion.modules.ema import LitEma
from latent_diffusion.modules.distributions.distributions import (
    normal_kl,
    DiagonalGaussianDistribution,
)
from latent_encoder.autoencoder import (
    VQModelInterface,
    IdentityFirstStage,
    AutoencoderKL,
)
from latent_diffusion.modules.diffusionmodules.util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
)
from latent_diffusion.models.ddim import DDIMSampler
from latent_diffusion.models.plms import PLMSSampler
import soundfile as sf
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import json
import pandas as pd

from torchvision.ops import sigmoid_focal_loss

__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2

# For onset prediction
def binary_focal_loss(pred, target, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-8):
    """
    pred: 預測機率 (sigmoid後), shape = (batch, ...)
    target: 標籤 (0/1), shape = (batch, ...)
    """
    p = pred.clamp(min=eps, max=1-eps)
    ce_loss = F.binary_cross_entropy(p, target, reduction='none')
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def binary_focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-8):

    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.sigmoid(logits)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_term = (1 - p_t) ** gamma * targets + p_t ** gamma * (1 - targets)
    loss = alpha_t * focal_term * bce_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

STEM_JSON_MAP = {
    'kick': ["Bass Drum 1"],
    'snare': ["Acoustic Snare"],
    'toms': ["High Floor Tom", "Low-Mid Tom", "High Tom"],
    'hi-hats': ["Closed Hi Hat", "Open Hi Hat"],
    'cymbals': ["Crash Cymbal 1", "Ride Cymbal 1"]
}

def load_real_onset_json(track_ids=None, batch=None, split='validation'):
    """
    將一個 batch 的所有 track 對應的 *_onsets.json 讀進來。
    參數
    ----
    track_ids : str | list[str] | None
        可以直接給 list of track_id；若為 None 則從 batch 解析。
    batch : dict | None
        batch['fname'] 或 batch['track_id'] 必須有對應項目。
    split : str
        'train' | 'validation' | 'test'
    Returns
    -------
    List[dict]  # 長度 = batch_size
    """
    # ---------- 1. 取得 track_ids (list) ----------
    if track_ids is None:
        if batch is None:
            raise ValueError('track_ids 與 batch 不能同時為 None')

        if 'fname' in batch:                                  # DataLoader 回傳的欄位
            fnames = batch['fname']
            # 可能是 list / tuple / Tensor / 單一字串
            if isinstance(fnames, (list, tuple)):
                fnames = list(fnames)
            elif hasattr(fnames, 'tolist'):                   # torch.Tensor
                fnames = [str(x) for x in fnames.tolist()]
            else:
                fnames = [fnames]

            track_ids = []
            for f in fnames:
                root = Path(f).name.split('_from_')[0]        # "Track00032_from_0" → "Track00032"
                track_ids.append(root)
        elif 'track_id' in batch:
            tids = batch['track_id']
            track_ids = list(tids) if isinstance(tids, (list, tuple)) else [tids]
        else:
            raise KeyError('batch 中找不到 fname 或 track_id 欄位')
    else:
        # 若外部直接給單一字串也轉成 list
        track_ids = track_ids if isinstance(track_ids, (list, tuple)) else [track_ids]

    # ---------- 2. 讀 track_mapping.csv ----------
    mapping_csv = '/PATH/TO/PROJECT/data/StemGMD_org/track_mapping.csv'
    df = pd.read_csv(mapping_csv)

    # ---------- 3. 逐筆讀 onset json ----------
    onset_dicts = []
    for tid in track_ids:
        row = df[(df['track_id'] == tid) & (df['split'] == split)]
        if row.empty:
            raise FileNotFoundError(f'找不到 track_id={tid}, split={split} 的對應行')

        take_name = row.iloc[0]['take']
        json_path = Path(f'/PATH/TO/PROJECT/midi_onset_gt/{take_name}_onsets.json')
        if not json_path.exists():
            raise FileNotFoundError(json_path)

        with open(json_path) as fp:
            onset_dicts.append(json.load(fp))

    return onset_dicts


def _load_real_onset_json(track_id=None, batch=None, split=None):
    """
    載入真實的 onset JSON 文件
    如果沒有提供track_id，則從batch中提取
    split: 'train', 'validation', 'test' 或 None (自動推測)
    """
    import pandas as pd  # 添加pandas import
    
    if track_id is None and batch is not None:
        # 從batch中提取track_id - 修正：檢查fname字段
        if 'fname' in batch:
            # 格式: "Track00032_from_0" -> track_id="Track00032", segment=0
            fname = batch['fname'][0] if isinstance(batch['fname'], list) else batch['fname']
            if '_from_' in fname:
                track_id = fname.split('_from_')[0]  # 提取Track00032部分
                segment = fname.split('_from_')[1]   # 提取0部分
                print(f"從fname提取: track_id={track_id}, segment={segment}")
            else:
                track_id = fname.split('_')[0] if '_' in fname else fname
                segment = "未知"
                print(f"從fname提取: track_id={track_id}, segment={segment}")
        elif 'track_id' in batch:
            track_id = batch['track_id'][0] if isinstance(batch['track_id'], list) else batch['track_id']
        else:
            print("無法從batch中提取track_id")
            return None
    
    if track_id is None:
        print("未提供track_id，無法載入onset JSON")
        return None
    
    # 如果沒有指定split，嘗試推測
    if split is None:
        # 根據當前環境推測split
        # 這裡可以根據實際情況調整邏輯
        split = 'validation'  # 默認假設是validation
        # print(f"未指定split，使用默認值: {split}")
    
    # print(f"=== 載入真實 onset JSON 文件: {track_id} (split: {split}) ===")
    
    # 載入track_mapping.csv
    track_mapping_path = "/PATH/TO/PROJECT/data/StemGMD_org/track_mapping.csv"
    
    if not os.path.exists(track_mapping_path):
        # print(f"未找到 track_mapping.csv: {track_mapping_path}")
        return None
    
    try:
        df = pd.read_csv(track_mapping_path)
        # print(f"成功載入 track_mapping.csv，共 {len(df)} 行")
        
        # 根據split和track_id找到對應的行
        track_info = df[(df['track_id'] == track_id) & (df['split'] == split)]
        
        if track_info.empty:
            print(f"在track_mapping.csv中未找到track_id: {track_id} (split: {split})")
            print(f"可用的track_id範圍: {df['track_id'].min()} - {df['track_id'].max()}")
            print(f"可用的split: {df['split'].unique()}")
            print(f"此track_id在不同split中的記錄:")
            track_all_splits = df[df['track_id'] == track_id]
            if not track_all_splits.empty:
                for _, row in track_all_splits.iterrows():
                    print(f"  - split: {row['split']}, take: {row['take']}")
            else:
                print(f"  未找到track_id: {track_id} 的任何記錄")
            return None
        
        # 取第一個匹配的行
        row = track_info.iloc[0]
        take_name = row['take']  # 例如: "69_jazz_125_fill_4-4"
        stems_dir = row['stems_dir']  # 音頻文件路徑
        
        # print(f"找到track信息: {track_id} -> {take_name} (split: {split})")
        # print(f"音頻文件路徑: {stems_dir}")
        
        # 根據take_name找到對應的onset JSON文件
        onset_json_path = f"/PATH/TO/PROJECT/midi_onset_gt/{take_name}_onsets.json"
        
        if not os.path.exists(onset_json_path):
            # print(f"未找到 onset JSON 文件: {onset_json_path}")
            # print(f"請檢查以下路徑是否存在:")
            # print(f"  - {onset_json_path}")
            return None
        
        # print(f"找到 onset 文件: {onset_json_path}")
        
        # 載入JSON文件
        with open(onset_json_path, 'r') as f:
            onset_data = json.load(f)
        
        # print(f"載入的 onset 數據: {onset_data}")
        return onset_data
        
    except Exception as e:
        # print(f"載入 track_mapping.csv 時發生錯誤: {e}")
        return None

def convert_onset_json_to_pianoroll(onset_data, duration=10.24, frame_rate=100.0, start_offset_sec: float = 0.0):
    """
    將 onset JSON 數據轉換為 5 stems 的 pianoroll
    """
    # 轉換 JSON → pianoroll；支援以秒為單位的起始位移（segment offset）
    
    total_frames = int(duration * frame_rate)
    # print(f"總幀數: {total_frames} (duration={duration}s, frame_rate={frame_rate}Hz)")
    
    # 初始化 5 stems 的 pianoroll
    pianoroll = np.zeros((5, total_frames), dtype=np.float32)
    stem_names = ['kick', 'snare', 'toms', 'hi-hats', 'cymbals']
    
    onset_counts = {}
    
    for stem_idx, (stem_name, json_keys) in enumerate(STEM_JSON_MAP.items()):
        # print(f"處理 stem: {stem_name} -> {json_keys}")
        onset_counts[stem_name] = 0
        
        for json_key in json_keys:
            if json_key in onset_data:
                onset_times = onset_data[json_key]
                # print(f"  {json_key}: {onset_times}")
                
                for onset_time in onset_times:
                    # 以 segment 起點為 0 計算相對幀索引
                    rel_time = float(onset_time) - float(start_offset_sec)
                    frame_idx = int(rel_time * frame_rate)
                    if 0 <= frame_idx < total_frames:
                        pianoroll[stem_idx, frame_idx] = 1.0
                        onset_counts[stem_name] += 1
                        # print(f"  onset at {onset_time}s -> frame {frame_idx}")
    
    # print(f"生成的 pianoroll 形狀: {pianoroll.shape}")
    # print("每個 stem 的 onset 數量:")
    # for stem_name, count in onset_counts.items():
    #     print(f"  {stem_name}: {count}")
    
    return pianoroll

def batch_convert_onset_json(batch_json, duration=10.24, frame_rate=100.0, offsets_sec=None):
    if offsets_sec is None:
        offsets_sec = [0.0] * len(batch_json)
    pr_list = [convert_onset_json_to_pianoroll(js, duration, frame_rate, offsets_sec[i])
               for i, js in enumerate(batch_json)]
    return np.stack(pr_list, 0)

def get_onset_pianoroll(waveform, sr):
    """
    使用 librosa 快速 onset detection
    waveform: numpy array, shape (num_stems, T) 或 (T,)（單 stem）
    return: pianoroll, shape (num_stems, frames) 或 (frames,)
    """
    # 檢查輸入音頻是否包含 NaN 或 Inf
    if np.isnan(waveform).any() or np.isinf(waveform).any():
        print(f"Warning: Found NaN or Inf in waveform input to get_onset_pianoroll, replacing with zeros")
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
        waveform = np.clip(waveform, -1, 1)
    
    # print(f"[DEBUG] get_onset_pianoroll waveform shape: {waveform.shape}")
    if waveform.ndim == 1:
        # 單 stem
        # 使用 librosa 快速 onset detection
        try:
            onset_frames = librosa.onset.onset_detect(
                y=waveform, 
                sr=sr, 
                units='frames',
                hop_length=160,  # 與你的 STFT 設定一致
                backtrack=True,  # 回溯到實際 onset 位置
                pre_max=20,      # 優化參數
                post_max=20,
                pre_avg=100,
                post_avg=100,
                delta=0.2,       # 降低閾值，增加檢測靈敏度
                wait=0
            )
            
            # 轉成 pianoroll
            frame_length = int(len(waveform) / sr * sr / 160)  # 根據 hop_length 計算 frame 數
            pianoroll = np.zeros(frame_length, dtype=np.float32)
            pianoroll[onset_frames] = 1.0
            return pianoroll
        except Exception as e:
            print(f"Error in onset detection for single stem: {e}")
            # 返回零 pianoroll
            frame_length = int(len(waveform) / sr * sr / 160)
            return np.zeros(frame_length, dtype=np.float32)
        
    elif waveform.ndim == 2:
        # 多 stem
        num_stems = waveform.shape[0]
        pianorolls = []
        for i in range(num_stems):
            try:
                # 使用 librosa 快速 onset detection
                onset_frames = librosa.onset.onset_detect(
                    y=waveform[i], 
                    sr=sr, 
                    units='frames',
                    hop_length=160,
                    backtrack=True,
                    pre_max=20,
                    post_max=20,
                    pre_avg=100,
                    post_avg=100,
                    delta=0.2,
                    wait=0
                )
                
                # 轉成 pianoroll
                frame_length = int(len(waveform[i]) / sr * sr / 160)
                pianoroll = np.zeros(frame_length, dtype=np.float32)
                pianoroll[onset_frames] = 1.0
                pianorolls.append(pianoroll)
            except Exception as e:
                print(f"Error in onset detection for stem {i}: {e}")
                # 返回零 pianoroll
                frame_length = int(len(waveform[i]) / sr * sr / 160)
                pianorolls.append(np.zeros(frame_length, dtype=np.float32))
        return np.stack(pianorolls, axis=0)  # (num_stems, frames)
    else:
        raise ValueError("waveform shape 不正確")

def weighted_bce_with_logits(logits, target):
    """
    logits, target: [B, S, T]
    針對每個 (B,S) 計算正例權重 w_pos = neg/pos，並只放大正例的 BCE。
    """
    B, S, T = target.shape
    # 避免除 0
    pos = target.sum(dim=-1).clamp_min(1e-6)       # [B,S]
    neg = (T - pos).clamp_min(1.0)
    w_pos = (neg / pos).clamp(1.0, 100.0)          # [B,S]

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')  # [B,S,T]
    # 只放大正例的位置
    weights = torch.where(target > 0, w_pos.unsqueeze(-1), torch.ones_like(target))
    return (bce * weights).mean()

def soft_f1_loss(logits, target, eps=1e-7):
    """
    與 F1 對齊的軟指標 (Dice/F1)。對每個 (B,S) 計算後再平均。
    """
    p = torch.sigmoid(logits)
    tp = (p * target).sum(dim=-1)
    fp = (p * (1 - target)).sum(dim=-1)
    fn = ((1 - p) * target).sum(dim=-1)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)         # [B,S]
    return 1.0 - f1.mean()

def _align_lengths_1d(pred, target):
    """確保 pred 與 target 長度一致，裁到最小長度。接受 tensor 或 ndarray，回傳 torch.Tensor 1D。"""
    import torch as _torch
    if not isinstance(pred, _torch.Tensor):
        pred = _torch.tensor(pred)
    if not isinstance(target, _torch.Tensor):
        target = _torch.tensor(target)
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    L = min(pred.numel(), target.numel())
    return pred[:L], target[:L]

def compute_sdr(pred, target, eps: float = 1e-8):
    """Scale-dependent SDR (dB): 10*log10(||s||^2 / ||s - s_hat||^2). Accepts torch.Tensor or np.ndarray (1D)."""
    import torch as _torch
    pred, target = _align_lengths_1d(pred, target)
    noise = target - pred
    num = _torch.sum(target ** 2)
    den = _torch.sum(noise ** 2) + eps
    val = 10.0 * _torch.log10((num + eps) / den)
    return float(val.detach().cpu().item())

def compute_si_sdr(pred, target, eps: float = 1e-8):
    """Scale-invariant SDR (dB) following Le Roux et al. Accepts torch.Tensor or np.ndarray (1D)."""
    import torch as _torch
    x_hat, s = _align_lengths_1d(pred, target)
    if _torch.sum(s ** 2) < eps:
        return float('-inf')
    alpha = _torch.dot(x_hat, s) / (_torch.dot(s, s) + eps)
    s_target = alpha * s
    e_noise = x_hat - s_target
    val = 10.0 * _torch.log10((_torch.sum(s_target ** 2) + eps) / (_torch.sum(e_noise ** 2) + eps))
    return float(val.detach().cpu().item())

def _safe_load_audio(path: str, target_sr: int = 16000):
    try:
        import soundfile as _sf
        import numpy as _np
        audio, sr = _sf.read(path, always_2d=False)
        if sr != target_sr:
            try:
                import librosa as _librosa
                audio = _librosa.resample(audio.astype(_np.float32), orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except Exception:
                pass
        if audio is None:
            return None
        if audio.ndim > 1:
            audio = audio[..., 0]
        audio = _np.asarray(audio, dtype=_np.float32)
        audio = _np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        max_abs = float(_np.max(_np.abs(audio))) if audio.size > 0 else 0.0
        if not _np.isfinite(max_abs) or max_abs <= 1e-12:
            return audio
        if max_abs > 1.0:
            audio = audio / max_abs
        audio = _np.clip(audio, -1.0, 1.0)
        return audio
    except Exception:
        return None

def _robust_eval_audio_folder(pred_dir: Path, gt_dir: Path):
    import numpy as _np
    from pathlib import Path as _Path
    pred_dir = _Path(pred_dir)
    gt_dir = _Path(gt_dir)
    pred_files = {p.name for p in pred_dir.glob("*.wav")}
    gt_files = {p.name for p in gt_dir.glob("*.wav")}
    common = sorted(list(pred_files & gt_files))
    if len(common) == 0:
        return None
    sdr_list = []
    sisdr_list = []
    for name in common:
        yp = _safe_load_audio(str(pred_dir / name))
        yt = _safe_load_audio(str(gt_dir / name))
        if yp is None or yt is None:
            continue
        L = min(len(yp), len(yt))
        if L <= 0:
            continue
        yp = yp[:L]
        yt = yt[:L]
        try:
            sdr_val = float(compute_sdr(yp, yt))
            sisdr_val = float(compute_si_sdr(yp, yt))
            if _np.isfinite(sdr_val):
                sdr_list.append(sdr_val)
            if _np.isfinite(sisdr_val):
                sisdr_list.append(sisdr_val)
        except Exception:
            continue
    if len(sdr_list) == 0 and len(sisdr_list) == 0:
        return None
    out = {}
    if len(sdr_list) > 0:
        out["sdr"] = float(_np.mean(sdr_list))
    if len(sisdr_list) > 0:
        out["si_sdr"] = float(_np.mean(sisdr_list))
    out["num_files"] = len(common)
    return out

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        unet_config,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        latent_t_size=256,
        latent_f_size=16,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
    ):
        print(f"[DEBUG] DDPM.__init__ conditioning_key: {conditioning_key}")
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
        ], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        self.state = None
        print(
            f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode"
        )
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key

        self.latent_t_size = latent_t_size
        self.latent_f_size = latent_f_size

        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(
                ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet
            )

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        else:
            self.logvar = nn.Parameter(self.logvar, requires_grad=False)

        self.logger_save_dir = None
        self.logger_project = None
        self.logger_version = None
        self.label_indices_total = None
        # To avoid the system cannot find metric value for checkpoint
        self.metrics_buffer = {
            "val/kullback_leibler_divergence_sigmoid": 15.0,
            "val/kullback_leibler_divergence_softmax": 10.0,
            "val/psnr": 0.0,
            "val/ssim": 0.0,
            "val/inception_score_mean": 1.0,
            "val/inception_score_std": 0.0,
            "val/kernel_inception_distance_mean": 0.0,
            "val/kernel_inception_distance_std": 0.0,
            "val/frechet_inception_distance": 133.0,
            "val/frechet_audio_distance": 32.0,
        }
        self.initial_learning_rate = None
        self.test_data_subset_path = None
        self._onset_pred_list = []
        self._onset_gt_list = []
        
    def get_log_dir(self):
        if (
            self.logger_save_dir is None
            and self.logger_project is None
            and self.logger_version is None
        ):
            return os.path.join(
                self.logger.save_dir, self.logger._project, self.logger.version
            )
        else:
            return os.path.join(
                self.logger_save_dir, self.logger_project, self.logger_version
            )

    def set_log_dir(self, save_dir, project, version):
        self.logger_save_dir = save_dir
        self.logger_project = project
        self.logger_version = version

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            self.load_state_dict(sd, strict=False)
            if not only_model
            else self.model.load_state_dict(sd, strict=False)
        )
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (
            (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))).contiguous()
        )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="Sampling t",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                clip_denoised=self.clip_denoised,
            )
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        shape = (batch_size, channels, self.latent_t_size, self.latent_f_size)
        channels = self.channels
        return self.p_sample_loop(shape, return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(
                f"Paramterization {self.parameterization} not yet supported"
            )

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = "train" if self.training else "val"

        loss_dict.update({f"{log_prefix}/loss_simple": loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f"{log_prefix}/loss_vlb": loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f"{log_prefix}/loss": loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        # print("Here 0")
        if k == 'text':
            # print("Here 1")
            return list(batch['text'])
        elif k == 'fname':
            return batch['fname']
        elif 'fbank' in k: # == 'fbank' or k == 'fbank_1' or k == 'fbank_2':
            return batch[k].unsqueeze(1).to(memory_format=torch.contiguous_format).float()
        else:
            return batch[k].to(memory_format=torch.contiguous_format).float()
        
    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        # x = batch[self.first_stage_key].to(memory_format=torch.contiguous_format).float()
        loss, loss_dict = self(x)
        return loss, loss_dict

    def warmup_step(self):
        if self.initial_learning_rate is None:
            self.initial_learning_rate = self.learning_rate

        # Only the first parameter group
        if self.global_step <= 1000:
            if self.global_step == 0:
                print(
                    "Warming up learning rate start with %s"
                    % self.initial_learning_rate
                )
            self.trainer.optimizers[0].param_groups[0]["lr"] = (
                self.global_step / 1000
            ) * self.initial_learning_rate
        else:
            # TODO set learning rate here
            self.trainer.optimizers[0].param_groups[0][
                "lr"
            ] = self.initial_learning_rate


    def training_step(self, batch, batch_idx):

        assert self.training, "training step must be in training stage"
        self.warmup_step()

        if (
            self.state is None
            and len(self.trainer.optimizers[0].state_dict()["state"].keys()) > 0
        ):
            self.state = (
                self.trainer.optimizers[0].state_dict()["state"][0]["exp_avg"].clone()
            )
        elif self.state is not None and batch_idx % 1000 == 0:
            assert (
                torch.sum(
                    torch.abs(
                        self.state
                        - self.trainer.optimizers[0].state_dict()["state"][0]["exp_avg"]
                    )
                )
                > 1e-7
            ), "Optimizer is not working"

        # if len(self.metrics_buffer.keys()) > 0:
        #     for k in self.metrics_buffer.keys():
        #         self.log(
        #             k,
        #             self.metrics_buffer[k],
        #             prog_bar=False,
        #             logger=True,
        #             on_step=True,
        #             on_epoch=False,
        #         )
        #         print(k, self.metrics_buffer[k])
        #     self.metrics_buffer = {}

        loss, loss_dict = self.shared_step(batch)

        self.log_dict(
            {k: float(v) for k, v in loss_dict.items()},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            "global_step",
            float(self.global_step),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "lr_abs",
            float(lr),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        
        # 額外補上 onset log
        if 'onset_loss' in loss_dict:
            self.log('train/onset_loss', loss_dict['onset_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if 'train/loss_onset' in loss_dict:
            self.log('train/loss_onset_step', loss_dict['train/loss_onset'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if 'onset_focal_loss_end' in loss_dict:
            self.log('onset_focal_loss_end_step', loss_dict['onset_focal_loss_end'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if 'train/loss' in loss_dict:
            self.log('train/loss_step', loss_dict['train/loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if 'train/loss_simple' in loss_dict:
            self.log('train/loss_simple_step', loss_dict['train/loss_simple'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if 'train/loss_vlb' in loss_dict:
            self.log('train/loss_vlb_step', loss_dict['train/loss_vlb'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if 'onset_f1' in loss_dict:
            self.log('train/onset_f1', loss_dict['onset_f1'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if 'onset_acc' in loss_dict:
            self.log('train/onset_acc', loss_dict['onset_acc'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # timbre 的日誌已在 p_losses 中記錄，避免重複衝突

        return loss

    def on_validation_epoch_start(self) -> None:
        # # Use text as condition during validation
        # if self.global_rank == 0:
        #     if self.model.conditioning_key is not None:
        #         if self.cond_stage_key_orig == "waveform":
        #             self.cond_stage_key = "text"
        #             self.cond_stage_model.embed_mode = "text"
        # 初始化每個 stem 的 SDR / SI-SDR 累積容器
        try:
            self._sdr_per_stem = [[] for _ in range(self.num_stems)]
            self._sisdr_per_stem = [[] for _ in range(self.num_stems)]
        except Exception:
            pass
        return super().on_validation_epoch_start()

    # def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
    #     self.cond_stage_key = "waveform"
    #     self.cond_stage_model.embed_mode = "audio"
    #     return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        assert not self.training, "Validation/Test must not be in training stage"
        # 檢查是否為 timbre_only 模式
        training_mode = getattr(self, 'training_mode', 'all')
        is_timbre_only = training_mode == "timbre_only"

        # 收集 onset prediction 所需的資料
        onset_data = None
        gt_onset = None  # 確保後續路徑可以引用
        if self.use_onset_prediction and hasattr(self, 'onset_predictor'):
            # 使用真實的 JSON 文件生成 GT onset - 修正：傳入batch參數和split參數
            # onset_json_data = load_real_onset_json(batch=batch, split='validation')
            batch_json = load_real_onset_json(batch=batch, split='validation')  # List[dict]
            if batch_json is not None:
            # if onset_json_data is not None:
                # 解析每個樣本的 segment 起點秒數，從 fname 的 "_from_" 之後取得
                offsets = []
                fnames = batch.get('fname', [])
                if not isinstance(fnames, (list, tuple)):
                    fnames = [fnames]
                for f in fnames:
                    try:
                        base = Path(f).name
                        if '_from_' in base:
                            sec = float(base.split('_from_')[1])
                        else:
                            sec = 0.0
                    except Exception:
                        sec = 0.0
                    offsets.append(sec)
                gt_np = batch_convert_onset_json(batch_json, duration=10.24, frame_rate=100.0, offsets_sec=offsets)
                gt_onset = torch.as_tensor(gt_np, dtype=torch.float32, device=self.device)
                
                # gt_onset_pianoroll = convert_onset_json_to_pianoroll(onset_json_data, duration=10.0, frame_rate=102.4)
                # 轉換為 batch 格式 [B, num_stems, T]

                # ====== 可選：導引圖可視化 ======
                try:
                    if getattr(self, 'enable_guide_features', False) and getattr(self, 'enable_guide_visualization', False):
                        guides = batch.get('guides', None)
                        if guides is not None and (isinstance(guides, torch.Tensor) or isinstance(guides, np.ndarray)):
                            if (batch_idx % max(1, int(getattr(self, 'guide_visualization_interval', 200)))) == 0:
                                self._save_guide_images(guides, batch, batch_idx)
                except Exception:
                    pass
                # gt_onset = torch.tensor(gt_onset_pianoroll, device=self.device, dtype=torch.float32)
                # print(f"Ground Truth onset shape: {gt_onset.shape}")
                # print(f"GT 範圍: [{gt_onset.min():.4f}, {gt_onset.max():.4f}]")
                # print(f"GT 中非零 onset 數量: {(gt_onset > 0.5).sum().item()}")
            else:
                # 如果無法載入 JSON，提供詳細的錯誤信息
                # print("=== VALIDATION ONSET JSON 載入失敗 ===")
                
                # 提取當前樣本的信息
                track_id = "未知"
                segment = "未知"
                audio_path = "未知"
                time_segment = "未知"
                
                if 'fname' in batch:
                    fname = batch['fname'][0] if isinstance(batch['fname'], list) else batch['fname']
                    if '_from_' in fname:
                        track_id = fname.split('_from_')[0]
                        segment = fname.split('_from_')[1]
                        # 計算時間段：segment是從0開始的索引，每個segment是10秒
                        start_time = int(segment) * 10
                        end_time = start_time + 10
                        time_segment = f"{start_time}s - {end_time}s"
                    else:
                        track_id = fname.split('_')[0] if '_' in fname else fname
                        segment = "未知"
                    # print(f"Track ID: {track_id}")
                    # print(f"Segment: {segment}")
                    # print(f"文件名: {fname}")
                    # print(f"時間段: {time_segment}")
                elif 'fnames' in batch:
                    fname = batch['fnames'][0] if isinstance(batch['fnames'], list) else batch['fnames']
                    track_id = fname.split('_')[0] if '_' in fname else fname
                    # print(f"Track ID: {track_id}")
                    # print(f"文件名: {fname}")
                
                if 'audio_path' in batch:
                    audio_path = batch['audio_path'][0] if isinstance(batch['audio_path'], list) else batch['audio_path']
                    # print(f"音頻文件路徑: {audio_path}")
                
                if 'start_time' in batch and 'end_time' in batch:
                    start_time = batch['start_time'][0] if isinstance(batch['start_time'], list) else batch['start_time']
                    end_time = batch['end_time'][0] if isinstance(batch['end_time'], list) else batch['end_time']
                    time_segment = f"{start_time:.2f}s - {end_time:.2f}s"
                    # print(f"時間段: {time_segment}")
                
                # 如果無法載入 JSON，使用 batch 中的數據作為備用
                if 'onset_pianoroll' in batch:
                    gt_onset = batch['onset_pianoroll']
                    if not isinstance(gt_onset, torch.Tensor):
                        gt_onset = torch.tensor(gt_onset, device=self.device, dtype=torch.float32)
                    else:
                        gt_onset = gt_onset.clone().detach().to(device=self.device, dtype=torch.float32)
                    # print("使用batch中的onset_pianoroll作為備用")
                else:
                    gt_onset = None
                    # print("無法找到任何onset數據，跳過onset prediction")

        # 後備：即便沒有 onset_predictor，也嘗試從 batch 取 GT，供 UNet 分支可視化
        if gt_onset is None and isinstance(batch, dict) and 'onset_pianoroll' in batch:
            tmp_gt = batch['onset_pianoroll']
            if not isinstance(tmp_gt, torch.Tensor):
                gt_onset = torch.tensor(tmp_gt, device=self.device, dtype=torch.float32)
            else:
                gt_onset = tmp_gt.clone().detach().to(device=self.device, dtype=torch.float32)

        # 只在非 timbre_only 模式下執行 DDIM sampling
        if not is_timbre_only and self.global_rank == 0:

            name = self.get_validation_folder_name()

            stems_to_inpaint = self.model._trainer.datamodule.config.get('path', {}).get('stems_to_inpaint', None)
            stems = self.model._trainer.datamodule.config.get('path', {}).get('stems', [])

            if stems_to_inpaint is not None:

                stemidx_to_inpaint = [i for i,s in enumerate(stems) if s in stems_to_inpaint]

                self.inpainting(
                    [batch],
                    ddim_steps=self.evaluation_params["ddim_sampling_steps"],
                    ddim_eta=1.0,
                    x_T=None,
                    n_gen=self.evaluation_params["n_candidates_per_samples"],
                    unconditional_guidance_scale=self.evaluation_params[
                        "unconditional_guidance_scale"
                    ],
                    unconditional_conditioning=None,
                    name=name,
                    use_plms=False,
                    stemidx_to_inpaint = stemidx_to_inpaint,
                )

            else:

                # 使用 generate_sample 并获取 samples
                samples = self.generate_sample(
                    [batch],
                    name=name,
                    unconditional_guidance_scale=self.evaluation_params[
                        "unconditional_guidance_scale"
                    ],
                    ddim_steps=self.evaluation_params["ddim_sampling_steps"],
                    n_gen=self.evaluation_params["n_candidates_per_samples"],
                    return_samples=True,  # 返回 samples 而不是保存路径
                )
                
                # 使用获取的 samples 进行 onset prediction

                if self.use_onset_prediction and hasattr(self, 'onset_predictor') and gt_onset is not None:
                    # print("[DEBUG] (checkpoint 2-3)")
                    pred_onset = self.onset_predictor(samples)
                    pred_onset = torch.sigmoid(pred_onset)
                    
                    # 對齊長度
                    if pred_onset.shape[-1] != gt_onset.shape[-1]:
                        if pred_onset.shape[-1] > gt_onset.shape[-1]:
                            pred_onset = pred_onset[:, :, :gt_onset.shape[-1]]
                        else:
                            pad_shape = list(pred_onset.shape)
                            pad_shape[-1] = gt_onset.shape[-1] - pred_onset.shape[-1]
                            padding = torch.zeros(*pad_shape, device=self.device)
                            pred_onset = torch.cat([pred_onset, padding], dim=-1)
                    
                    # 若 pred_onset 維度多於3，壓縮到 [B, num_stems, T]
                    while pred_onset.dim() > 3:
                        pred_onset = pred_onset.mean(dim=2)
                    
                    
                    # ===== 簡化可視化：每個 epoch 只隨機選擇一個樣本 =====
                    batch_size = pred_onset.shape[0]
                    if batch_size > 0:
                        # 隨機選擇一個樣本進行可視化
                        import random
                        random.seed(self.current_epoch)  # 確保每個 epoch 選擇相同的樣本
                        sample_idx = random.randint(0, batch_size - 1)
                        print("===========================================")
                        print("pred_onset.shape : ", pred_onset.shape)
                        print("gt_onset.shape : ", gt_onset.shape)
                        print("===========================================")
                        # 獲取當前樣本的預測和ground truth
                        pred_sample = pred_onset[sample_idx].detach().cpu().numpy()  # [num_stems, time_steps]
                        gt_sample = gt_onset[sample_idx].detach().cpu().numpy()  # [num_stems, time_steps]
                        
                        # 使用0.5 threshold將prediction轉換為0,1值
                        pred_sample_binary = (pred_sample > 0.5).astype(np.float32)
                        
                        # 從batch中提取track_id用於命名
                        track_id = "unknown"
                        if 'fname' in batch:
                            fname = batch['fname'][0] if isinstance(batch['fname'], list) else batch['fname']
                            if '_from_' in fname:
                                track_id = fname.split('_from_')[0]  # 提取Track00032部分
                        
                        # 創建保存路徑
                        save_dir = os.path.join(self.get_log_dir(), "onset_visualization_val")
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # 保存組合圖，使用track_id命名
                        combined_path = os.path.join(save_dir, 
                            f"onset_combined_val_epoch_{self.current_epoch}_sample_{sample_idx}_{track_id}.png")
                        plot_onset_pianoroll_combined(
                            pred_sample_binary, gt_sample, combined_path, 
                            self.current_epoch, batch_idx, sample_idx
                        )
                    
                    onset_data = {
                        'pred_onset': pred_onset,
                        'gt_onset': gt_onset,
                        'batch_idx': batch_idx
                    }
                    self._onset_pred_list.append(pred_onset.cpu())
                    self._onset_gt_list.append(gt_onset.cpu())
                # 若沒有獨立 onset_predictor，但使用 UNet 分支，亦提供可視化所需資料
                elif (getattr(self, 'enable_onset_prediction', False) or getattr(self, 'use_onset_prediction', False)) and (getattr(self, '_mt_last_onset', None) is not None) and (gt_onset is not None):
                    pred_onset = torch.sigmoid(self._mt_last_onset)
                    # 壓到 [B,S,T]
                    while pred_onset.dim() > 3:
                        pred_onset = pred_onset.mean(dim=2)
                    # 對齊長度
                    if pred_onset.shape[-1] != gt_onset.shape[-1]:
                        T = min(pred_onset.shape[-1], gt_onset.shape[-1])
                        pred_onset = pred_onset[..., :T]
                        gt_onset = gt_onset[..., :T]
                    # 每個 epoch 粗略存一張圖
                    try:
                        batch_size = pred_onset.shape[0]
                        if batch_size > 0:
                            import random
                            random.seed(self.current_epoch)
                            sample_idx = random.randint(0, batch_size - 1)
                            pred_sample = pred_onset[sample_idx].detach().cpu().numpy()
                            gt_sample = gt_onset[sample_idx].detach().cpu().numpy()
                            pred_sample_binary = (pred_sample > 0.5).astype(np.float32)
                            track_id = "unknown"
                            if 'fname' in batch:
                                fname = batch['fname'][0] if isinstance(batch['fname'], list) else batch['fname']
                                if '_from_' in fname:
                                    track_id = fname.split('_from_')[0]
                            save_dir = os.path.join(self.get_log_dir(), "onset_visualization_val")
                            os.makedirs(save_dir, exist_ok=True)
                            combined_path = os.path.join(save_dir, f"onset_combined_val_epoch_{self.current_epoch}_sample_{sample_idx}_{track_id}.png")
                            plot_onset_pianoroll_combined(
                                pred_sample_binary, gt_sample, combined_path,
                                self.current_epoch, batch_idx, sample_idx
                            )
                    except Exception:
                        pass
                    onset_data = {
                        'pred_onset': pred_onset,  # 注意：callback 會視為概率
                        'gt_onset': gt_onset,
                        'batch_idx': batch_idx
                    }
                    try:
                        self._onset_pred_list.append(pred_onset.detach().cpu())
                        self._onset_gt_list.append(gt_onset.detach().cpu())
                    except Exception:
                        pass

        # 計算 loss（包括 timbre_only 模式）
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {
                key + "_ema": loss_dict_ema[key] for key in loss_dict_ema
            }

        # 返回 onset 資料供 epoch_end 使用
        return {
            'loss_dict_no_ema': loss_dict_no_ema,
            'loss_dict_ema': loss_dict_ema,
            'onset_data': onset_data
        }

    def get_validation_folder_name(self):
        return "val_%s" % (self.global_step)

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        # ====== Onset F1 評估（在 epoch 結束時計算） ======

        if self.use_onset_prediction and hasattr(self, 'onset_predictor'):
            # 收集所有 validation steps 的 onset 資料

            if len(self._onset_pred_list) > 0 and len(self._onset_gt_list):
                pred_onset = torch.cat(self._onset_pred_list, dim=0)
                gt_onset = torch.cat(self._onset_gt_list, dim=0)
                # Debug 資訊
                # print(f"[DEBUG] pred_onset range: [{pred_onset.min().item():.4f}, {pred_onset.max().item():.4f}], mean: {pred_onset.mean().item():.4f}")
                # print(f"[DEBUG] gt_onset range: [{gt_onset.min().item():.4f}, {gt_onset.max().item():.4f}], mean: {gt_onset.mean().item():.4f}")
                
                total_batch, num_stems, T = gt_onset.shape
                
                # 樂器名稱映射
                instrument_names = {
                    0: "Kick",
                    1: "Snare", 
                    2: "Toms",
                    3: "Hi-Hats",
                    4: "Cymbals"
                }
                
                # 多種閾值策略
                threshold_strategies = {
                    # 'mean': pred_onset.mean().item(),
                    # 'median': pred_onset.median().item(),
                    # 'adaptive': max(0.1, min(0.9, pred_onset.mean().item())),
                    'fixed_0.1': 0.1,
                    'fixed_0.3': 0.3,
                    'fixed_0.5': 0.5,
                    'fixed_0.7': 0.7
                }
                
                # 對每種閾值策略計算指標
                for strategy_name, threshold in threshold_strategies.items():
                    print(f"\n[VAL] Using threshold strategy: {strategy_name} (threshold={threshold:.4f})")
                    
                    pred_bin = (pred_onset > threshold).float()
                    gt_bin = (gt_onset > 0.5).float()  # GT 閾值固定為 0.5
                    
                    # 計算混淆矩陣
                    tp = (pred_bin * gt_bin).sum()
                    fp = (pred_bin * (1 - gt_bin)).sum()
                    fn = ((1 - pred_bin) * gt_bin).sum()
                    tn = ((1 - pred_bin) * (1 - gt_bin)).sum()
                    
                    # 計算指標
                    precision = (tp / (tp + fp + 1e-8)).item()
                    recall = (tp / (tp + fn + 1e-8)).item()
                    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
                    accuracy = ((tp + tn) / (tp + tn + fp + fn + 1e-8)).item()
                    
                    # 計算 Micro 和 Macro 平均
                    # Micro: 先合併所有類別，再計算指標
                    micro_precision = precision
                    micro_recall = recall
                    micro_f1 = f1_score
                    
                    # Macro: 每個類別分別計算，再平均
                    macro_precisions = []
                    macro_recalls = []
                    macro_f1s = []
                    
                    for stem_idx in range(num_stems):
                        pred_stem = pred_bin[:, stem_idx, :]
                        gt_stem = gt_bin[:, stem_idx, :]
                        
                        tp_stem = (pred_stem * gt_stem).sum()
                        fp_stem = (pred_stem * (1 - gt_stem)).sum()
                        fn_stem = ((1 - pred_stem) * gt_stem).sum()
                        
                        prec_stem = (tp_stem / (tp_stem + fp_stem + 1e-8)).item()
                        rec_stem = (tp_stem / (tp_stem + fn_stem + 1e-8)).item()
                        f1_stem = (2 * prec_stem * rec_stem / (prec_stem + rec_stem + 1e-8))
                        
                        macro_precisions.append(prec_stem)
                        macro_recalls.append(rec_stem)
                        macro_f1s.append(f1_stem)
                    
                    macro_precision = np.mean(macro_precisions)
                    macro_recall = np.mean(macro_recalls)
                    macro_f1 = np.mean(macro_f1s)
                    
                    # 記錄指標
                    print(f"[VAL] {strategy_name} - Micro: F1={micro_f1:.4f}, P={micro_precision:.4f}, R={micro_recall:.4f}")
                    print(f"[VAL] {strategy_name} - Macro: F1={macro_f1:.4f}, P={macro_precision:.4f}, R={macro_recall:.4f}")
                    print(f"[VAL] {strategy_name} - Accuracy: {accuracy:.4f}")
                    
                    # 記錄到 wandb
                    self.log(f"val/onset_micro_f1_{strategy_name}", micro_f1, prog_bar=False, on_step=False, on_epoch=True)
                    self.log(f"val/onset_micro_precision_{strategy_name}", micro_precision, prog_bar=False, on_step=False, on_epoch=True)
                    self.log(f"val/onset_micro_recall_{strategy_name}", micro_recall, prog_bar=False, on_step=False, on_epoch=True)
                    
                    self.log(f"val/onset_macro_f1_{strategy_name}", macro_f1, prog_bar=False, on_step=False, on_epoch=True)
                    self.log(f"val/onset_macro_precision_{strategy_name}", macro_precision, prog_bar=False, on_step=False, on_epoch=True)
                    self.log(f"val/onset_macro_recall_{strategy_name}", macro_recall, prog_bar=False, on_step=False, on_epoch=True)
                    
                    self.log(f"val/onset_accuracy_{strategy_name}", accuracy, prog_bar=False, on_step=False, on_epoch=True)
                    
                    # 記錄混淆矩陣元素
                    self.log(f"val/onset_tp_{strategy_name}", tp.item(), prog_bar=False, on_step=False, on_epoch=True)
                    self.log(f"val/onset_fp_{strategy_name}", fp.item(), prog_bar=False, on_step=False, on_epoch=True)
                    self.log(f"val/onset_fn_{strategy_name}", fn.item(), prog_bar=False, on_step=False, on_epoch=True)
                    self.log(f"val/onset_tn_{strategy_name}", tn.item(), prog_bar=False, on_step=False, on_epoch=True)
                    
                    # 每個 stem 的詳細指標
                    for stem_idx in range(num_stems):
                        instrument_name = instrument_names.get(stem_idx, f"Stem_{stem_idx}")
                        print(f"[VAL] {strategy_name} - {instrument_name}: F1={macro_f1s[stem_idx]:.4f}, P={macro_precisions[stem_idx]:.4f}, R={macro_recalls[stem_idx]:.4f}")
                        
                        self.log(f"val/onset_{strategy_name}_f1_{instrument_name.lower()}", macro_f1s[stem_idx], prog_bar=False, on_step=False, on_epoch=True)
                        self.log(f"val/onset_{strategy_name}_precision_{instrument_name.lower()}", macro_precisions[stem_idx], prog_bar=False, on_step=False, on_epoch=True)
                        self.log(f"val/onset_{strategy_name}_recall_{instrument_name.lower()}", macro_recalls[stem_idx], prog_bar=False, on_step=False, on_epoch=True)
                
                # 選擇最佳閾值策略（基於 macro F1）
                best_strategy = max(threshold_strategies.keys(), 
                                  key=lambda x: self.trainer.logged_metrics.get(f"val/onset_macro_f1_{x}", 0))
                best_f1 = self.trainer.logged_metrics.get(f"val/onset_macro_f1_{best_strategy}", 0)
                
                print(f"\n[VAL] Best threshold strategy: {best_strategy} (F1={best_f1:.4f})")
                # self.log("val/onset_best_strategy", best_strategy, prog_bar=False, on_step=False, on_epoch=True)
                self.log("val/onset_best_f1", best_f1, prog_bar=True, on_step=False, on_epoch=True)
                
                # 計算 onset loss
                # onset_loss = torch.nn.functional.binary_cross_entropy(pred_onset, gt_onset, reduction='mean').item()
                onset_loss = binary_focal_loss(pred_onset, gt_onset, alpha=0.25, gamma=2.0, reduction='mean').item()
                self.log("val/onset_loss", onset_loss, prog_bar=False, on_step=False, on_epoch=True)
                print(f"[VAL] Onset Loss: {onset_loss:.4f}")

                self._onset_pred_list.clear()
                self._onset_gt_list.clear()
                
        # ====== End Onset F1 ======

        if self.global_rank == 0:
            self.test_data_subset_path = os.path.join(self.get_log_dir(), "target_%s" % (self.global_step))
    
            if self.test_data_subset_path is not None:
                from audioldm_eval import EvaluationHelper

                print(
                    "Evaluate model output based on the data savee in: %s"
                    % self.test_data_subset_path
                )
                device = self.device #torch.device(f"cuda:{0}")
                name = self.get_validation_folder_name()
                waveform_save_path = os.path.join(self.get_log_dir(), name)
                if (
                    os.path.exists(waveform_save_path)
                    and len(os.listdir(waveform_save_path)) > 0
                ):
                    # evaluator = EvaluationHelper(16000, device)
                    # metrics = evaluator.main(
                    #     waveform_save_path,
                    #     self.test_data_subset_path,
                    # )

                    # self.metrics_buffer = {
                    #     ("val/" + k): float(v) for k, v in metrics.items()
                    # }
                    dir1 = Path(waveform_save_path)
                    dir2 = Path(self.test_data_subset_path)

                    # Get set of folder names in each directory
                    dir1_folders = {folder.name for folder in dir1.iterdir() if folder.is_dir()}
                    dir2_folders = {folder.name for folder in dir2.iterdir() if folder.is_dir()}

                    # Find the intersection of folder names existing in both directories
                    # matching_folders = dir1_folders & dir2_folders

                    # Evaluation toggles
                    eval_params = getattr(self, 'evaluation_params', {}) or {}
                    enable_audio_eval = bool(eval_params.get('enable_audio_eval', False))
                    enable_mel_eval = bool(eval_params.get('enable_mel_eval', True))

                    if enable_audio_eval:
                        # Find the intersection of folder names existing in both directories, excluding those with "mel" in their names
                        matching_folders = {folder for folder in (dir1_folders & dir2_folders) if "mel" not in folder.lower()}

                        # Iterate through matching folders and perform operations
                        for folder_name in matching_folders:
                            folder1 = dir1 / folder_name
                            folder2 = dir2 / folder_name

                            # 以穩健評測取代外部工具：逐檔讀 wav → 對齊 → 計 SDR/SI-SDR
                            try:
                                r = _robust_eval_audio_folder(folder1, folder2)
                                if r is None:
                                    continue
                                buf = {}
                                if "sdr" in r:
                                    buf[f"val/{folder_name}/sdr"] = float(r["sdr"])
                                if "si_sdr" in r:
                                    buf[f"val/{folder_name}/si_sdr"] = float(r["si_sdr"])
                                buf[f"val/{folder_name}/num_files"] = int(r.get("num_files", 0))
                                for k, v in buf.items():
                                    self.log(k, v, prog_bar=False, logger=True, on_step=False, on_epoch=True)
                            except Exception:
                                continue

                    if enable_mel_eval:
                        # Find the intersection of folder names existing in both directories, only those with "mel" in their names
                        matching_folders = {folder for folder in (dir1_folders & dir2_folders) if "mel" in folder.lower()}
                        # Iterate through matching folders and perform operations
                        for folder_name in matching_folders:
                            folder1 = dir1 / folder_name
                            folder2 = dir2 / folder_name

                            print("\nNow evaliating:", folder_name)

                            try:
                                results_mse = evaluate_separations(folder1, folder2)

                                self.metrics_buffer = {
                                    (f"val/{folder_name}/" + k): float(v.item() if hasattr(v, 'item') else v) for k, v in results_mse.items()
                                }
                            except Exception as e:
                                print(f"Error during evaluation of {folder_name}: {e}")
                                print("Skipping this evaluation and continuing...")
                                continue





                else:
                    print(
                        "The target folder for evaluation does not exist: %s"
                        % waveform_save_path
                    )

        self.cond_stage_key = self.cond_stage_key_orig
        if self.cond_stage_model is not None:
            self.cond_stage_model.embed_mode = self.cond_stage_model.embed_mode_orig
        # ====== 關閉每 stem 的 SDR / SI-SDR 聚合與記錄 ======
        # （保留容器但不做任何 logging，以加速驗證）

        return super().on_validation_epoch_end()

    def on_train_batch_end(self, *args, **kwargs):
        # Does this affect speed?
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(
                    batch_size=N, return_intermediates=True
                )

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def _save_guide_images(self, guides, batch, batch_idx):
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            import numpy as np
            import os
        except Exception:
            return

        if isinstance(guides, torch.Tensor):
            g = guides.detach().cpu().float().numpy()
        else:
            g = np.asarray(guides)

        fnames = batch.get('fname', None)
        if isinstance(fnames, (list, tuple)):
            fnames = list(fnames)
        elif isinstance(fnames, str):
            fnames = [fnames]

        if g.ndim < 4:
            return
        B, Cg = g.shape[0], g.shape[1]
        max_samples = int(getattr(self, 'guide_visualization_max_samples', 2))
        num = min(B, max_samples)

        save_dir = os.path.join(self.get_log_dir(), f"guides_batch_{batch_idx}")
        os.makedirs(save_dir, exist_ok=True)

        for i in range(num):
            fig, axes = plt.subplots(1, Cg, figsize=(3*Cg, 3), squeeze=False)
            for c in range(Cg):
                ax = axes[0, c]
                img = g[i, c]
                ax.imshow(img, origin='lower', aspect='auto', cmap='magma')
                ax.axis('off')
                ax.set_title(f"C{c}")
            base = f"sample_{i}"
            if fnames and i < len(fnames):
                base = str(fnames[i])
            out = os.path.join(save_dir, f"{base}.png")
            plt.tight_layout()
            try:
                plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
            except Exception:
                pass
            plt.close(fig)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class MusicLDM(DDPM):
    """main class"""

    def __init__(
        self,
        first_stage_config,
        cond_stage_config=None,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        batchsize=None,
        evaluation_params={},
        scale_by_std=False,
        base_learning_rate=None,
        latent_mixup=0.,
        num_stems=None,
        seperate_stem_z = False,
        use_silence_weight = False,
        tau = 3.0,
        *args,
        **kwargs,
    ):
        self.num_stems = num_stems
        self.learning_rate = base_learning_rate
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        self.evaluation_params = evaluation_params
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        # ====== 初始化可選的 onset 相關旗標，避免屬性缺失 ======
        # 支援兩種命名：use_onset_prediction / enable_onset_prediction
        self.use_onset_prediction = kwargs.get('use_onset_prediction', kwargs.get('enable_onset_prediction', False))
        self.use_onset_prediction_end = kwargs.get('use_onset_prediction_end', False)
        self.onset_loss_weight = kwargs.get('onset_loss_weight', 0.0)
        self.onset_warmup_steps = kwargs.get('onset_warmup_steps', 0)
        # 路徑類屬性（若其他模組需要）
        self.onset_data_path = kwargs.get('onset_data_path', None)
        self.track_mapping_path = kwargs.get('track_mapping_path', None)
        self.timbre_data_path = kwargs.get('timbre_data_path', None)
        
        # ====== 導引圖特徵配置 ======
        self.enable_guide_features = kwargs.get('enable_guide_features', False)
        self.guide_features_config = kwargs.get('guide_features', {})
        # 暫存配置參數，在 super().__init__() 後創建模組
        self._guide_config = {
            'use_guide_indices': self.guide_features_config.get('use_guide_indices', None),
            'guide_channels': int(self.guide_features_config.get('guide_channels', 4)),
            'film_dim': int(self.guide_features_config.get('film_dim', 32)),
            'latent_channels': int(self.guide_features_config.get('latent_channels', 0))
        }
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])

        parent_kwargs = kwargs.copy()
        # 移除多任務相關的參數，這些參數不應該傳遞給 DDPM 父類
        parent_kwargs.pop('onset_output_frames', None)
        parent_kwargs.pop('timbre_feature_dim', None)
        parent_kwargs.pop('use_onset_branch', None)
        parent_kwargs.pop('use_timbre_branch', None)
        parent_kwargs.pop('onset_branch_channels', None)
        parent_kwargs.pop('timbre_branch_channels', None)
        parent_kwargs.pop('num_stems', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('onset_warmup_steps', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('onset_data_path', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('track_mapping_path', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('timbre_data_path', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('training_mode', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('use_timbre_prediction', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('timbre_loss_weight', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('latent_loss_weight', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('enable_latent_diffusion', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('enable_onset_prediction', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('enable_timbre_prediction', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('use_onset_prediction_end', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('use_onset_prediction', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('onset_loss_weight', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('use_noise', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('use_random_timesteps', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('fixed_timesteps', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('fixed_timestep', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('z_channels', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('enable_guide_features', None)  # 這個參數也不應該傳遞給父類
        parent_kwargs.pop('guide_features', None)  # 這個參數也不應該傳遞給父類
        # 導引圖可視化參數也不應傳給父類
        parent_kwargs.pop('enable_guide_visualization', None)
        parent_kwargs.pop('guide_visualization_interval', None)
        parent_kwargs.pop('guide_visualization_max_samples', None)

        # 僅將已過濾的參數傳給父類，避免將多任務相關參數傳至 DDPM.__init__
        # 允許 YAML 控制是否可視化導引特徵（從 kwargs 取值，但不傳給父類）
        self.enable_guide_visualization = bool(kwargs.pop('enable_guide_visualization', False))
        self.guide_visualization_interval = int(kwargs.pop('guide_visualization_interval', 200))
        self.guide_visualization_max_samples = int(kwargs.pop('guide_visualization_max_samples', 2))

        super().__init__(conditioning_key=conditioning_key, **parent_kwargs)
        
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.cond_stage_key_orig = cond_stage_key
        self.latent_mixup = latent_mixup

        print(f'Use the Latent MixUP of {self.latent_mixup}')

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        
        # ====== 在 super().__init__() 後創建導引圖相關模組 ======
        if self.enable_guide_features:
            # 計算輸入維度
            use_guide_indices = self._guide_config['use_guide_indices']
            _film_in_dim = len(use_guide_indices) if (isinstance(use_guide_indices, list) and len(use_guide_indices) > 0) else self._guide_config['guide_channels']
            
            # FiLM MLP（供 C0/C1 全域向量使用）
            self.film_mlp = torch.nn.Sequential(
                torch.nn.Linear(_film_in_dim, self._guide_config['film_dim']),
                torch.nn.SiLU(),
                torch.nn.Linear(self._guide_config['film_dim'], self._guide_config['film_dim']),
            )
            
            # 1x1 conv 將導引圖壓到與 latent 相容的通道數（僅當需要 concat 時）
            if self._guide_config['latent_channels'] > 0:
                self.guide_conv = torch.nn.Conv2d(_film_in_dim, self._guide_config['latent_channels'], kernel_size=1)
            else:
                self.guide_conv = None
        else:
            self.film_mlp = None
            self.guide_conv = None
        
        # 清理臨時配置
        delattr(self, '_guide_config')

        # Patch VAE functions into cond_stage_model
        #####################
        if 'target' in cond_stage_config and cond_stage_config['target'] == 'latent_diffusion.modules.encoders.modules.Patch_Cond_Model':
            self.cond_stage_model.encode_first_stage = self.encode_first_stage
            self.cond_stage_model.get_first_stage_encoding = self.get_first_stage_encoding
            self.cond_stage_model.num_stems = self.num_stems
            self.cond_stage_model.device = self.first_stage_model.get_device
        #####################

        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        self.z_channels = first_stage_config["params"]["ddconfig"]["z_channels"]

        self.seperate_stem_z = seperate_stem_z
        self.use_silence_weight = use_silence_weight
        self.tau = tau

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print("Diffusion model optimizing logvar")
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert "target" in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    def make_cond_schedule(
        self,
    ):
        self.cond_ids = torch.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=torch.long,
        )
        ids = torch.round(
            torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)
        ).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if (
            self.scale_factor == 1
            and self.scale_by_std
            and self.current_epoch == 0
            and self.global_step == 0
            and batch_idx == 0
            and not self.restarted_from_ckpt
        ):
            # assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)

            #### 
            x = self.adapt_fbank_for_VAE_encoder(x)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            z=self.adapt_latent_for_LDM(z)

            del self.scale_factor
            self.register_buffer("scale_factor", 1.0 / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(
            given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s
        )

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        model = instantiate_from_config(config)
        self.cond_stage_model = model
        if model is not None:
            self.cond_stage_model = self.cond_stage_model.to(self.device)

    def _get_denoise_row_from_list(
        self, samples, desc="", force_no_decoder_quantization=False
    ):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(
                self.decode_first_stage(
                    zd.to(self.device), force_not_quantize=force_no_decoder_quantization
                )
            )
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, "n b c h w -> b n c h w")
        denoise_grid = rearrange(denoise_grid, "b n c h w -> (b n) c h w")
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(
                self.cond_stage_model.encode
            ):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                if len(c) == 1 and self.cond_stage_model.embed_mode == "text":
                    c = self.cond_stage_model([c[0], c[0]])
                    c = c[0:1]
                elif isinstance(c, (np.ndarray, torch.Tensor)) and len(c.shape) == 1:
                    c = self.cond_stage_model(c.unsqueeze(0))
                else:
                    c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(
            torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1
        )[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(
            weighting,
            self.split_input_params["clip_min_weight"],
            self.split_input_params["clip_max_weight"],
        )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(
                L_weighting,
                self.split_input_params["clip_min_tie_weight"],
                self.split_input_params["clip_max_tie_weight"],
            )

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(
        self, x, kernel_size, stride, uf=1, df=1
    ):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(
                kernel_size=kernel_size, dilation=1, padding=0, stride=stride
            )
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(
                kernel_size[0], kernel_size[1], Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(
                kernel_size=kernel_size, dilation=1, padding=0, stride=stride
            )
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                dilation=1,
                padding=0,
                stride=(stride[0] * uf, stride[1] * uf),
            )
            fold = torch.nn.Fold(
                output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2
            )

            weighting = self.get_weighting(
                kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(
                1, 1, h * uf, w * uf
            )  # normalizes the overlap
            weighting = weighting.view(
                (1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx)
            )

        elif df > 1 and uf == 1:
            fold_params = dict(
                kernel_size=kernel_size, dilation=1, padding=0, stride=stride
            )
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(
                kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                dilation=1,
                padding=0,
                stride=(stride[0] // df, stride[1] // df),
            )
            fold = torch.nn.Fold(
                output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2
            )

            weighting = self.get_weighting(
                kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device
            ).to(x.dtype)
            normalization = fold(weighting).view(
                1, 1, h // df, w // df
            )  # normalizes the overlap
            weighting = weighting.view(
                (1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx)
            )

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    def adapt_fbank_for_VAE_encoder(self, tensor):
        # Assuming tensor shape is [batch_size, 1, num_channels, ...]
        batch_size, _, num_channels, *dims = tensor.shape

        # Check if num_channels matches the model's num_stems
        assert num_channels == self.num_stems, f"num_channels ({num_channels}) does not match model's num_stems ({self.num_stems})"
        
        # Reshape tensor to merge batch_size and num_channels for batch processing
        # The new shape will be [batch_size * num_channels, 1, ...] where "..." represents the original last two dimensions
        tensor_reshaped = tensor.view(batch_size * num_channels, 1, *dims)

        return tensor_reshaped

    def adapt_latent_for_LDM(self, tensor):
        # Now, dynamically calculate the new shape for z to ensure batch_size remains unchanged
        new_height, new_width = tensor.shape[-2:]
        
        # # Check if num_channels matches the model's num_stems
        assert new_height == self.latent_t_size, f"latent_t_size ({new_height}) does not match model's latent_t_size ({self.latent_t_size})"
        assert new_width == self.latent_f_size, f"latent_t_size ({new_height}) does not match model's latent_t_size ({self.latent_f_size})"

        # new_num_channels = self.num_stems #* self.z_channels

        # # Calculate new number of channels based on the total number of elements in z divided by (batch_size * height * width)
        # total_elements = tensor.numel()
        # batch_size = total_elements // (new_num_channels * new_height * new_width)

        tensor_reshaped = tensor.view(-1, self.num_stems, self.z_channels, new_height, new_width)

        return tensor_reshaped

    def adapt_latent_for_VAE_decoder(self, tensor):
        # Assume tensor shape is [batch_size, new_channel_size, 256, 16]
        batch_size, new_stem, new_cahnnel_size, height, width = tensor.shape
        
        
        # Calculate the new batch size, keeping the total amount of data constant
        # The total number of elements is divided by the product of the old_channel_size, height, and width
        # updated_batch_size = batch_size * (new_cahnnel_size // self.z_channels)
        
        # Reshape tensor to [batch_size_updated, old_channel_size, 256, 16]
        tensor_reshaped = tensor.view(-1,  self.z_channels, height, width)
        
        return tensor_reshaped


    @torch.no_grad()
    def get_input(
        self,
        batch,
        k,
        return_first_stage_encode=True,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        bs=None
    ):
        if self.training and self.latent_mixup > 0:
            # doing the mixup
            x = super().get_input(batch, k)
            x1 = super().get_input(batch, k + '_1')
            x2 = super().get_input(batch, k + '_2')
            select_idx = torch.where(torch.rand(x.size(0)) < self.latent_mixup)[0]
            x1 = x1[select_idx]
            x2 = x2[select_idx]

            if return_first_stage_encode:
                encoder_posterior = self.encode_first_stage(x1)
                z1 = self.get_first_stage_encoding(encoder_posterior).detach()
                encoder_posterior = self.encode_first_stage(x2)
                z2 = self.get_first_stage_encoding(encoder_posterior).detach()
                encoder_posterior = self.encode_first_stage(x)
                z = self.get_first_stage_encoding(encoder_posterior).detach()
                p = torch.from_numpy(np.random.beta(5,5, x1.size(0)))
                p = p[:,None,None,None].to(self.device)

                nz = p * z1 + (1 - p) * z2
                nz = nz.float()
                z = z.float()
                nx = self.decode_first_stage(nz).detach()
                z[select_idx] = nz
                x[select_idx] = nx
                x.to(self.device)
                z.to(self.device)
            else:
                z = None

            if self.model.conditioning_key is not None:
                if cond_key is None:
                    cond_key = self.cond_stage_key

                if cond_key == 'waveform':
                    xc = super().get_input(batch, cond_key).cpu()
                    nxc = torch.from_numpy(self.mel_spectrogram_to_waveform(nx, save=False)).squeeze(1)
                    if nxc.size(-1) != xc.size(-1):
                        nxc = nxc[:, :int(xc.size(-1) * 0.9)]
                        nxc = torch.nn.functional.pad(nxc, (0, xc.size(-1) - nxc.size(-1)), 'constant', 0.)
                    xc[select_idx] = nxc
                    xc = xc.detach()
                    xc.requires_grad = False
                    xc.to(self.device)

                if not self.cond_stage_trainable or force_c_encode:
                    if isinstance(xc, dict) or isinstance(xc, list):
                        c = self.get_learned_conditioning(xc)
                    else:
                        c = self.get_learned_conditioning(xc.to(self.device))
                else:
                    c = xc
                if bs is not None:
                    c = c[:bs]
            else:
                raise f'Need a condition'
        else:

            x = super().get_input(batch, k)

            if bs is not None:
                x = x[:bs]

            x = x.to(self.device)

            if return_first_stage_encode:
            
                if k == "fbank_stems":
                    # adapt multichannel before processing
                    x_reshaped = self.adapt_fbank_for_VAE_encoder(x)

                    encoder_posterior = self.encode_first_stage(x_reshaped)
                    z = self.get_first_stage_encoding(encoder_posterior).detach()

                    z = self.adapt_latent_for_LDM(z)

                elif k == "fbank":

                    encoder_posterior = self.encode_first_stage(x)
                    z = self.get_first_stage_encoding(encoder_posterior).detach()
                else:
                    raise NotImplementedError
            else:
                z = None

            if self.model.conditioning_key is not None:
                if cond_key is None:
                    cond_key = self.cond_stage_key
                if cond_key != self.first_stage_key:
                    # [bs, 1, 527]
                    xc = super().get_input(batch, cond_key)
                    if type(xc) == torch.Tensor:
                        xc = xc.to(self.device)
                else:
                    xc = x

                if not self.cond_stage_trainable or force_c_encode:
                    if isinstance(xc, dict) or isinstance(xc, list):
                        c = self.get_learned_conditioning(xc)
                    else:
                        c = self.get_learned_conditioning(xc.to(self.device))
                else:
                    c = xc
                if bs is not None:
                    c = c[:bs]

            else:
                c = None
                xc = None
                if self.use_positional_encodings:
                    pos_x, pos_y = self.compute_latent_shifts(batch)
                    c = {"pos_x": pos_x, "pos_y": pos_y}
        
        # ====== 處理導引圖特徵 ======
        guides = None
        film_vec = None
        if self.enable_guide_features and isinstance(batch, dict) and 'guides' in batch and 'film_vec' in batch:
            try:
                # 獲取 guides 和 film_vec
                guides = batch['guides'].to(self.device)  # [B, Cg, M, T]
                film_vec = batch['film_vec'].to(self.device)  # [B, Cg]

                # 根據 use_guide_indices 篩選通道
                use_indices = self.guide_features_config.get('use_guide_indices', None)
                if use_indices is not None:
                    if isinstance(use_indices, list) and len(use_indices) > 0:
                        indices_tensor = torch.tensor(use_indices, device=self.device, dtype=torch.long)
                        guides = torch.index_select(guides, 1, indices_tensor) # 篩選 guides
                        film_vec = torch.index_select(film_vec, 1, indices_tensor) # 同步篩選 film_vec
                        Cg = guides.shape[1] # 更新 Cg 為篩選後的通道數
                    else:
                        print(f"警告：use_guide_indices 格式不正確或為空: {use_indices}，將使用所有導引通道。")

                # 將 guides 適配到 latent 尺寸
                if z is not None and guides is not None:
                    B, Cg_filtered, M, T = guides.shape # Cg_filtered 是篩選後的通道數
                    
                    # 安全地獲取 z 的維度
                    if z.dim() == 5:
                        # z 是 5D: [batch_size, num_stems, z_channels, height, width]
                        _, _, _, H_lat, W_lat = z.shape
                        concat_dim = 2  # 在 stem 維度後拼接
                    elif z.dim() == 4:
                        # z 是 4D: [batch_size, z_channels, height, width]
                        _, _, H_lat, W_lat = z.shape
                        concat_dim = 1  # 在通道維度拼接
                    else:
                        print(f"警告：z 的維度不支援：{z.shape}")
                        guides = None
                        film_vec = None
                        # 跳過後續處理
                        pass
                    
                    # 讀取導引通道數設定（使用初始化時的 guide_conv）
                    Dg = int(self.guide_features_config.get('latent_channels', 0))
                    if Dg > 0 and (hasattr(self, 'guide_conv') and self.guide_conv is not None):
                        # 使用 adaptive pooling 將 guides 下採樣到 latent 尺寸
                        guides_resized = torch.nn.functional.adaptive_avg_pool2d(
                            guides, (H_lat, W_lat)
                        )  # [B, Cg_filtered, H_lat, W_lat]
                        # 直接使用已於 __init__ 建立的 1x1 conv
                        g_lat = self.guide_conv.to(self.device)(guides_resized)  # [B, Dg, H_lat, W_lat]
                        # 將 g_lat 與 z 在適當的維度上拼接
                        if z.dim() == 5:
                            num_stems = z.shape[1]
                            g_lat = g_lat.unsqueeze(1).expand(-1, num_stems, -1, -1, -1)  # [B, num_stems, Dg, H_lat, W_lat]
                        z = torch.cat([z, g_lat], dim=concat_dim)
                    else:
                        # 不做通道拼接，僅保留 film_vec 給 FiLM
                        guides = None
                        
            except Exception as e:
                print(f"處理導引圖特徵失敗: {e}")
                import traceback
                traceback.print_exc()
                guides = None
                film_vec = None
        
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        
        # 添加 guides 和 film_vec 到輸出（訓練/推論一致回傳；呼叫端需僅取前兩個作為 z,c）
        if guides is not None:
            out.append(guides)
        if film_vec is not None:
            out.append(film_vec)
        
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1.0 / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(
                    z, ks, stride, uf=uf
                )

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view(
                    (z.shape[0], -1, ks[0], ks[1], z.shape[-1])
                )  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [
                        self.first_stage_model.decode(
                            z[:, :, :, :, i],
                            force_not_quantize=predict_cids or force_not_quantize,
                        )
                        for i in range(z.shape[-1])
                    ]
                else:

                    output_list = [
                        self.first_stage_model.decode(z[:, :, :, :, i])
                        for i in range(z.shape[-1])
                    ]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(
                        z, force_not_quantize=predict_cids or force_not_quantize
                    )
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(
                    z, force_not_quantize=predict_cids or force_not_quantize
                )
            else:
                return self.first_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(
        self, z, predict_cids=False, force_not_quantize=False
    ):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, "b h w c -> b c h w").contiguous()

        z = 1.0 / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(
                    z, ks, stride, uf=uf
                )

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view(
                    (z.shape[0], -1, ks[0], ks[1], z.shape[-1])
                )  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [
                        self.first_stage_model.decode(
                            z[:, :, :, :, i],
                            force_not_quantize=predict_cids or force_not_quantize,
                        )
                        for i in range(z.shape[-1])
                    ]
                else:

                    output_list = [
                        self.first_stage_model.decode(z[:, :, :, :, i])
                        for i in range(z.shape[-1])
                    ]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(
                        z, force_not_quantize=predict_cids or force_not_quantize
                    )
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(
                    z, force_not_quantize=predict_cids or force_not_quantize
                )
            else:
                return self.first_stage_model.decode(z)

    def mel_spectrogram_to_waveform(
        self, mel, savepath=".", bs=None, name="outwav", save=True
    ):
        # Mel: [bs, 1, t-steps, fbins]
        if len(mel.size()) == 4:
            mel = mel.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = self.first_stage_model.vocoder(mel)
        waveform = waveform.cpu().detach().numpy()
        if save:
            self.save_waveform(waveform, savepath, name)
        return waveform

    def save_waveform(self, waveform, savepath, name="outwav"):
        # 修正非有限值並限制振幅，避免後續評測報錯
        try:
            import numpy as _np
            waveform = _np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
            max_abs = _np.max(_np.abs(waveform)) if waveform.size > 0 else 1.0
            if not _np.isfinite(max_abs) or max_abs == 0:
                max_abs = 1.0
            # 可選：正規化到 [-1,1]
            if max_abs > 1.0:
                waveform = waveform / max_abs
            waveform = _np.clip(waveform, -1.0, 1.0)
        except Exception:
            pass
        for i in range(waveform.shape[0]):
            if type(name) is str:
                path = os.path.join(
                    savepath, "%s_%s_%s.wav" % (self.global_step, i, name)
                )
            elif type(name) is list:
                path = os.path.join(
                    savepath,
                    "%s.wav"
                    % (
                        os.path.basename(name[i])
                        if (not ".wav" in name[i])
                        else os.path.basename(name[i]).split(".")[0]
                    ),
                )
            else:
                raise NotImplementedError
            sf.write(path, waveform[i, 0], samplerate=16000)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params["original_image_size"] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(
                    x, ks, stride, df=df
                )
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view(
                    (z.shape[0], -1, ks[0], ks[1], z.shape[-1])
                )  # (bn, nc, ks[0], ks[1], L )

                output_list = [
                    self.first_stage_model.encode(z[:, :, :, :, i])
                    for i in range(z.shape[-1])
                ]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        # 獲取輸入，可能包含 guides 和 film_vec
        inputs = self.get_input(batch, self.first_stage_key, return_first_stage_outputs=False)
        x = inputs[0]  # z
        c = inputs[1]  # condition
        
        # 檢查是否有額外的特徵
        guides = None
        film_vec = None
        
        # 安全地提取 guides 和 film_vec
        try:
            if len(inputs) > 2:
                guides = inputs[2] if len(inputs) > 2 else None
            if len(inputs) > 3:
                film_vec = inputs[3] if len(inputs) > 3 else None
        except Exception as e:
            print(f"處理導引圖特徵失敗: {e}")
            guides = None
            film_vec = None
        
        loss = self(x, c, batch=batch, guides=guides, film_vec=film_vec, **kwargs)
        return loss

    def forward(self, x, c, *args, guides=None, film_vec=None, **kwargs):
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            # if self.shorten_cond_schedule:  # TODO: drop this option
            #     tc = self.cond_ids[t].to(self.device)
            #     c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        
        # 將 guides 和 film_vec 傳遞給 p_losses
        loss = self.p_losses(x, c, t, guides=guides, film_vec=film_vec, *args, **kwargs)
        return loss

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, return_ids=False, guides=None, film_vec=None):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            if self.model.conditioning_key == "concat":
                key = "c_concat"
            elif self.model.conditioning_key == "crossattn":
                key = "c_crossattn"
            else:
                key = "c_film"

            cond = {key: cond}
        
        # 處理 film_vec（如果有的話）
        if film_vec is not None and hasattr(self, 'enable_guide_features') and self.enable_guide_features:
            # 創建一個小的 MLP 來處理 film_vec，輸出 gamma 和 beta
            if not hasattr(self, 'film_mlp'):
                Cg = film_vec.shape[1] if len(film_vec.shape) > 1 else 4
                # 從配置中獲取參數，如果沒有則使用默認值
                extra_film_condition_dim = self.guide_features_config.get('film_dim', 32)
                z_channels = getattr(self, 'z_channels', 8)  # 獲取 VAE 的 z_channels
                self.film_mlp = torch.nn.Sequential(
                    torch.nn.Linear(Cg, extra_film_condition_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(extra_film_condition_dim, 2 * z_channels)  # gamma + beta
                ).to(self.device)
                # 零初始化 FiLM 的最後一層，避免初始隨機放大
                with torch.no_grad():
                    self.film_mlp[-1].weight.zero_()
                    self.film_mlp[-1].bias.zero_()
            
            # 處理 film_vec，輸出 gamma 和 beta
            film_params = self.film_mlp(film_vec)  # [B, 2*z_channels]
            
            # 將 film_params 添加到條件中作為額外的 FiLM 條件
            # 注意：這裡不改變原本的 mix concat 邏輯，只是添加額外的 FiLM 條件
            cond['c_film'] = [film_params]
        
        # print(f"cond: {cond.shape}")
        # print(f'x_noisy: {x_noisy.shape}')
        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def denoise_one_step(self, x, t, e_t): 
        # parameters       
        b_t = extract_into_tensor(self._buffers["betas"], t, x.shape)
        a_t = 1 - b_t
        sqrt_one_minus_alphas_cumprod_t = extract_into_tensor(self._buffers["sqrt_one_minus_alphas_cumprod"], t, x.shape)
        # sqrt_one_minus_at = torch.sqrt(1.0 - a_t)
        # # denoising
        # pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # return pred_x0 
        return (x - e_t * b_t / sqrt_one_minus_alphas_cumprod_t) / a_t.sqrt()


    def p_losses(self, x_start, cond, t, noise=None, guides=None, film_vec=None, *args, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # print(f"x_start: {x_start.shape}")
        # print(f"cond: {cond.shape}")
        # print(f"t: {t.shape}")
        # print(f"noise: {noise.shape}")
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond, guides=guides, film_vec=film_vec)

        loss_dict = {}
        prefix = "train" if self.training else "val"

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()
        # print(model_output.size(), target.size())
        if self.use_silence_weight:
            batch = kwargs.pop('batch', None)  # Extract 'batch' from 'kwargs'
            
            z_mask = torch.nn.functional.interpolate(batch['fbank_stems'], size=(256, 16), mode='nearest')

            z_mask = torch.exp( - z_mask / self.tau)

            loss_simple = self.get_loss(model_output, target, mean=False).mean([2])
            loss_simple = loss_simple * z_mask
            loss_simple = loss_simple.mean([1, 2, 3])
        else:
            loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])



        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb



        #*******************************************************************
        ## loss term that pushes stems far from each other in z space
        if self.seperate_stem_z:
            x_t_1 = self.denoise_one_step(x_noisy, t, model_output)
            seperation_loss = self.channel_separation_loss(x_t_1)
            loss_dict.update({f"{prefix}/loss_separation": seperation_loss})
            loss += 0.00001 * seperation_loss
        #*******************************************************************

        loss_dict.update({f"{prefix}/loss": loss})
        return loss, loss_dict


    def channel_separation_loss(self,z):
        bs, num_channels, _, _, _ = z.shape
        loss = 0.0
        # Iterate over pairs of channels
        for i in range(num_channels):
            for j in range(i + 1, num_channels):
                # Compute the squared Euclidean distance between channels i and j
                diff = z[:, i] - z[:, j]
                distance_squared = (diff ** 2).sum(dim=[1, 2, 3])  # Sum over all dimensions except the batch dimension
                loss += distance_squared.mean()  # Average over the batch dimension

        # The negative of the sum of distances (because we want to maximize the distance)
        return -loss

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(
                self, model_out, x, t, c, **corrector_kwargs
            )

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
    ):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_codebook_ids=return_codebook_ids,
            quantize_denoised=quantize_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
        )
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (
            (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))).contiguous()
        )

        if return_codebook_ids:
            return model_mean + nonzero_mask * (
                0.5 * model_log_variance
            ).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                x0,
            )
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        # if return_codebook_ids:
        #     return model_mean, logits.argmax(dim=1)
        # if return_x0:
        #     return model_mean, x0
        # else:
        #     return model_mean

    @torch.no_grad()
    def progressive_denoising(
        self,
        cond,
        shape,
        verbose=True,
        callback=None,
        quantize_denoised=False,
        img_callback=None,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        batch_size=None,
        x_T=None,
        start_T=None,
        log_every_t=None,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(
                reversed(range(0, timesteps)),
                desc="Progressive Generation",
                total=timesteps,
            )
            if verbose
            else reversed(range(0, timesteps))
        )
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
                return_x0=True,
                temperature=temperature[i],
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
            )
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
    ):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))
            img = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
            )
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_size=16,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        shape=None,
        **kwargs,
    ):
        if shape is None:
            shape = (batch_size, self.channels, self.latent_t_size, self.latent_f_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )
        return self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            timesteps=timesteps,
            quantize_denoised=quantize_denoised,
            mask=mask,
            x0=x0,
            **kwargs,
        )

    @torch.no_grad()
    def sample_log(
        self,
        cond,
        batch_size,
        ddim,
        ddim_steps,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_plms=False,
        mask=None,
        **kwargs,
    ):

        if mask is not None:
            shape = (self.channels, self.z_channels, mask.size()[-2], mask.size()[-1])
        else:
            shape = (self.channels, self.z_channels, self.latent_t_size, self.latent_f_size)
        # print(f"shape: {shape}")
        # print(f"cond: {cond.shape}")
        intermediate = None
        if ddim and not use_plms:
            print("Use ddim sampler")

            ddim_sampler = DDIMSampler(self)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                mask=mask,
                **kwargs,
            )
        elif use_plms:
            print("Use plms sampler")
            plms_sampler = PLMSSampler(self)
            samples, intermediates = plms_sampler.sample(
                ddim_steps,
                batch_size,
                shape,
                cond,
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                mask=mask,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )

        else:
            print("Use DDPM sampler")
            samples, intermediates = self.sample(
                cond=cond,
                batch_size=batch_size,
                return_intermediates=True,
                unconditional_guidance_scale=unconditional_guidance_scale,
                mask=mask,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs,
            )

        return samples, intermediate

    @torch.no_grad()
    def generate_long_sample(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        generate_duration=60,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name="waveform",
        use_plms=False,
        **kwargs,
    ):
        assert n_gen == 1
        assert x_T is None

        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        def find_best_waveform_alignment(waveform1, waveform2, margin=1000):
            # [2, 1, 163872]
            diff = 32768
            best_offset = None
            for i in range(2, margin):
                waveform_distance = np.mean(
                    np.abs(waveform1[..., i:] - waveform2[..., :-i])
                )
                if waveform_distance < diff:
                    best_offset = i
                    diff = waveform_distance
            for i in range(2, margin):
                waveform_distance = np.mean(
                    np.abs(waveform2[..., i:] - waveform1[..., :-i])
                )
                if waveform_distance < diff:
                    best_offset = -i
                    diff = waveform_distance
            return best_offset

        use_ddim = ddim_steps is not None
        waveform_save_path = os.path.join(self.get_log_dir(), name)
        os.makedirs(waveform_save_path, exist_ok=True)
        print("Waveform save path: ", waveform_save_path)

        with self.ema_scope("Plotting"):
            for batch in batchs:
                inputs = self.get_input(
                    batch,
                    self.first_stage_key,
                    return_first_stage_outputs=False,
                    force_c_encode=True,
                    return_original_cond=False,
                    bs=None,
                )
                # 兼容導引圖可選返回：只取前兩個作為 (z, c)
                z = inputs[0]
                c = inputs[1] if len(inputs) > 1 else None
                text = super().get_input(batch, "text")

                # Generate multiple samples
                batch_size = z.shape[0]
                c = torch.cat([c], dim=0)

                if unconditional_guidance_scale != 1.0:
                    unconditional_conditioning = (
                        self.cond_stage_model.get_unconditional_condition(batch_size)
                    )

                fnames = list(super().get_input(batch, "fname"))

                waveform = None
                waveform_segment_length = None
                mel_segment_length = None

                while True:
                    if waveform is None:
                        # [2, 8, 256, 16]
                        samples, _ = self.sample_log(
                            cond=c,
                            batch_size=batch_size,
                            x_T=x_T,
                            ddim=use_ddim,
                            ddim_steps=ddim_steps,
                            eta=ddim_eta,
                            unconditional_guidance_scale=unconditional_guidance_scale,
                            unconditional_conditioning=unconditional_conditioning,
                            use_plms=use_plms,
                        )

                        # [2, 1, 1024, 64]
                        mel = self.decode_first_stage(samples)

                        # [2, 1, 163872] np.array
                        waveform = self.mel_spectrogram_to_waveform(
                            mel,
                            savepath=waveform_save_path,
                            bs=None,
                            name=fnames,
                            save=False,
                        )
                        mel_segment_length = mel.size(-2)
                        waveform_segment_length = waveform.shape[-1]
                    else:
                        _, h, w = samples.shape[0], samples.shape[2], samples.shape[3]

                        mask = torch.ones(batch_size, h, w).to(self.device)
                        mask[:, 3 * (h // 16) :, :] = 0
                        mask = mask[:, None, ...]

                        rolled_sample = torch.roll(samples, shifts=(h // 4), dims=2)

                        samples, _ = self.sample_log(
                            cond=c,
                            batch_size=batch_size,
                            x_T=x_T,
                            ddim=use_ddim,
                            ddim_steps=ddim_steps,
                            eta=ddim_eta,
                            unconditional_guidance_scale=unconditional_guidance_scale,
                            unconditional_conditioning=unconditional_conditioning,
                            mask=mask,
                            use_plms=use_plms,
                            x0=rolled_sample,
                        )

                        # [2, 1, 1024, 64]
                        mel_continuation = self.decode_first_stage(samples)

                        # [2, 1, 163872] np.array
                        waveform_continuation = self.mel_spectrogram_to_waveform(
                            mel_continuation,
                            savepath=waveform_save_path,
                            bs=None,
                            name=fnames,
                            save=False,
                        )

                        margin_waveform = waveform[
                            ..., -(waveform_segment_length // 4) :
                        ]
                        offset = find_best_waveform_alignment(
                            margin_waveform,
                            waveform_continuation[..., : margin_waveform.shape[-1]],
                        )
                        print("Concatenation offset is %s" % offset)
                        waveform = np.concatenate(
                            [
                                waveform[
                                    ..., : -(waveform_segment_length // 4) + 2 * offset
                                ],
                                waveform_continuation,
                            ],
                            axis=-1,
                        )
                        self.save_waveform(waveform, waveform_save_path, name=fnames)
                        if waveform.shape[-1] / 16000 > generate_duration:
                            break

        return waveform_save_path

    @torch.no_grad()
    def generate_sample(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name="waveform",
        use_plms=False,
        **kwargs,
    ):
        # Generate n_gen times and select the best
        # Batch: audio, text, fnames
        assert x_T is None
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        waveform_save_path = os.path.join(self.get_log_dir(), name)
        os.makedirs(waveform_save_path, exist_ok=True)
        # print("\nWaveform save path: ", waveform_save_path)              

        wavefor_target_save_path = os.path.join(self.get_log_dir(), "target_%s" % (self.global_step))
        os.makedirs(wavefor_target_save_path, exist_ok=True)
        # print("\nWaveform target save path: ", wavefor_target_save_path)      

        # if (
        #     "audiocaps" in waveform_save_path
        #     and len(os.listdir(waveform_save_path)) >= 964
        # ):
        #     print("The evaluation has already been done at %s" % waveform_save_path)
        #     return waveform_save_path

        with self.ema_scope("Plotting"):
            for batch in batchs:
                inputs = self.get_input(
                    batch,
                    self.first_stage_key,
                    return_first_stage_outputs=False,
                    force_c_encode=True,
                    return_original_cond=False,
                    bs=None,
                )
                z = inputs[0]
                c = inputs[1] if len(inputs) > 1 else None

                if self.cond_stage_model is not None:

                    # Generate multiple samples
                    batch_size = z.shape[0] * n_gen

                    if self.cond_stage_model.embed_mode == "text":
                        text = super().get_input(batch, "text")
                       
                        if c is not None:
                            c = torch.cat([c] * n_gen, dim=0)
                        text = text * n_gen
                    elif self.cond_stage_model.embed_mode == "audio":
                        text = super().get_input(batch, "waveform")

                        if c is not None:
                            c = torch.cat([c] * n_gen, dim=0)
                        text = torch.cat([text] * n_gen, dim=0)

                    if unconditional_guidance_scale != 1.0:
                        unconditional_conditioning = (
                            self.cond_stage_model.get_unconditional_condition(batch_size)
                        )
                else:
                    batch_size = z.shape[0]
                    text = None

                fnames = list(super().get_input(batch, "fname"))
                # print("================")
                # print(f"c: {c.shape}")
                # print(f"batch_size: {batch_size}")
                # print(f"x_T: {x_T}")
                # print("================")
                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    use_plms=use_plms,
                )

                samples = self.adapt_latent_for_VAE_decoder(samples)
                mel = self.decode_first_stage(samples)

                waveform = self.mel_spectrogram_to_waveform(
                    mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
                )

                # Convert dtype to float32 if it is float16
                if waveform.dtype == 'float16':
                    waveform = waveform.astype('float32')
                    

                # downmix to songs for comparison
                waveform_reshaped = waveform.reshape(batch_size, self.num_stems, waveform.shape[-1])
                mix = waveform_reshaped.sum(axis=1)

                waveform = np.nan_to_num(waveform)
                waveform = np.clip(waveform, -1, 1)
                
                mix = np.nan_to_num(mix)
                mix = np.clip(mix, -1, 1)


                if self.model.conditioning_key is not None:
                    if self.cond_stage_model.embed_mode == "text": # TODO maybe make similar for audio (???)
                        similarity = self.cond_stage_model.cos_similarity(
                            torch.FloatTensor(mix).squeeze(1), text
                        )

                        best_index = []
                        for i in range(z.shape[0]):
                            candidates = similarity[i :: z.shape[0]]
                            max_index = torch.argmax(candidates).item()
                            best_index.append(i + max_index * z.shape[0])
                            # print("Similarity between generated audio and text", similarity)
                            # print("Choose the following indexes:", best_index)
                    else:
                        best_index = torch.arange(z.shape[0])
                else:
                    best_index = torch.arange(z.shape[0])

                # chose best scored mixes
                mix = mix[best_index]

                # chose coresponding stems audios and mels:
                selected_wavs = []
                selected_mels = []
                for start_index in best_index:

                    actual_start_index = start_index * self.num_stems

                    # Ensure the selection does not exceed array bounds
                    selected_slice = waveform[actual_start_index:actual_start_index + self.num_stems]
                    selected_wavs.append(selected_slice)

                    selected_slice = mel[actual_start_index:actual_start_index + self.num_stems].cpu().detach().numpy()
                    selected_mels.append(selected_slice)

                    
                waveform = np.concatenate(selected_wavs, axis=0)[:,0,:]
                waveform = waveform.reshape(z.shape[0], self.num_stems, waveform.shape[-1]) # back to batch size and multicahnnel

                # test_names =  [str(number) for number in range(4)]                
                # self.save_waveform(waveform[1][:, np.newaxis, :], "/home/karchkhadze/MusicLDM-Ext/test_folder", name=test_names)


                mel = np.concatenate(selected_mels, axis=0)[:,0,:]
                mel = mel.reshape(z.shape[0], self.num_stems, mel.shape[-2], mel.shape[-1]) # back to batch size and multicahnnel

                ############################# saving audios for metrics ##################################
                # save mixes
                    #generated
                generated_mix_dir = os.path.join(waveform_save_path, "mix")
                os.makedirs(generated_mix_dir, exist_ok=True)
                if mix.ndim == 1:
                    mix = mix[np.newaxis, :] 
                self.save_waveform(mix[:, np.newaxis, :], generated_mix_dir, name=fnames)
                    #target
                target_mix_dir = os.path.join(wavefor_target_save_path, "mix")
                os.makedirs(target_mix_dir, exist_ok=True)
                target_mix = super().get_input(batch, 'waveform')
                self.save_waveform(target_mix.unsqueeze(1).cpu().detach(), target_mix_dir, name=fnames)

                # save stems
                target_waveforms = super().get_input(batch, 'waveform_stems')
                for i in range(self.num_stems):
                    
                    # generated
                    generated_stem_dir = os.path.join(os.path.join(waveform_save_path, "stem_"+str(i)))
                    os.makedirs(generated_stem_dir, exist_ok=True)                    
                    self.save_waveform(waveform[:,i,:][:, np.newaxis, :], generated_stem_dir, name=fnames)
                    # mel
                    generated_stem_mel_dir = os.path.join(os.path.join(waveform_save_path, "stem_mel_"+str(i)))
                    os.makedirs(generated_stem_mel_dir, exist_ok=True) 
                    for j in range(mel.shape[0]):
                        file_path =  os.path.join(generated_stem_mel_dir,fnames[j]+".npy")
                        np.save(file_path, mel[j,i,:])


                    # target
                    target_stem_dir = os.path.join(os.path.join(wavefor_target_save_path, "stem_"+str(i)))
                    os.makedirs(target_stem_dir, exist_ok=True)                    
                    self.save_waveform(target_waveforms[:,i,:].unsqueeze(1).cpu().detach(), target_stem_dir, name=fnames)
                    # mel
                    target_stem_mel_dir = os.path.join(os.path.join(wavefor_target_save_path, "stem_mel_"+str(i)))
                    os.makedirs(target_stem_mel_dir, exist_ok=True) 
                    for j in range(mel.shape[0]):
                        file_path =  os.path.join(target_stem_mel_dir,fnames[j]+".npy")
                        np.save(file_path, batch['fbank_stems'].cpu().numpy()[j,i,:])



                ###################################### logging ##############################################     
                if self.logger is not None:
                    # create new list
                    log_data_batch = mel, waveform, target_waveforms, mix, target_mix, fnames, batch
                    self.log_images_audios(log_data_batch)

                # ===== 計算每個 stem 的 SDR / SI-SDR 並紀錄（不再用 try/except 吃錯誤） =====
                B = waveform.shape[0]
                S = waveform.shape[1]
                for i in range(B):
                    for s_idx in range(S):
                        pred_wav = waveform[i, s_idx]
                        gt_wav = target_waveforms[i, s_idx].cpu().numpy()
                        sdr_val = compute_sdr(pred_wav, gt_wav)
                        sisdr_val = compute_si_sdr(pred_wav, gt_wav)
                        print(f"[val] file={fnames[i]} stem={s_idx} SDR={sdr_val:.2f} dB, SI-SDR={sisdr_val:.2f} dB")
                        self.log(f"val/stem_{s_idx}/SDR", sdr_val, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                        self.log(f"val/stem_{s_idx}/SI-SDR", sisdr_val, on_step=True, on_epoch=True, prog_bar=False, logger=True)
                        if hasattr(self, '_sdr_per_stem') and s_idx < len(self._sdr_per_stem):
                            self._sdr_per_stem[s_idx].append(sdr_val)
                        if hasattr(self, '_sisdr_per_stem') and s_idx < len(self._sisdr_per_stem):
                            self._sisdr_per_stem[s_idx].append(sisdr_val)

        return waveform_save_path

    def tensor2numpy(self, tensor):
        return tensor.cpu().detach().numpy()
    
    def log_images_audios(self, log_data_batch):
        mel, waveform, target_waveforms, mix, target_mix, fnames, batch = log_data_batch

        # Use get to safely access "text" from batch, defaulting to a list of empty string if not found
        text = batch.get("text", [""] * mel.shape[0])

        # get target mel
        target_mel = self.tensor2numpy(batch['fbank_stems'])   

        name = "val"

        ### logginh spectrograms ###
        for i in range(mel.shape[0]):
            self.logger.log_image(
                "Mel_specs %s" % name,
                [np.concatenate([np.flipud(target_mel[i,j].T) for j in range(target_mel[i].shape[0])], axis=0), 
                 np.concatenate([np.flipud(mel[i,j].T) for j in range(mel[i].shape[0])], axis=0) ],

                caption=["target_fbank_%s" % fnames[i]+text[i], "generated_%s" %fnames[i]+text[i]],
            )

            ### logging audios ###

            log_dict = {}

            log_dict ["target_%s"% name] =  wandb.Audio(
                        self.tensor2numpy(target_mix)[i], caption= f"Full Song: {fnames[i]} {text[i]}", sample_rate=16000,)
            log_dict ["generated_%s"% name] =wandb.Audio(
                        mix[i], caption= f"Full Song: {fnames[i]} {text[i]}", sample_rate=16000,)

            for k in range(self.num_stems):
                log_dict[f"{name}_target_stem{k}"] = wandb.Audio(
                            self.tensor2numpy(target_waveforms)[i,k], caption= f"Stem {k}: {fnames[i]} {text[i]}", sample_rate=16000,)
                log_dict[f"{name}_generated_stem{k}"] = wandb.Audio(
                            waveform[i,k], caption= f"Stem {k}: {fnames[i]} {text[i]}" , sample_rate=16000,)

            # self.logger.experiment.log(
            #     {
            #         "target_%s"
            #         % name: wandb.Audio(
            #             self.tensor2numpy(target_mix)[i], caption= f"Full Song: {fnames[i]} {text[i]}", sample_rate=16000,
            #         ),
            #         "generated_%s"
            #         % name: wandb.Audio(
            #             mix[i], caption= f"Full Song: {fnames[i]} {text[i]}", sample_rate=16000,
            #         ),
            #     }
            # )

            # for k in range(self.num_stems):
            #     self.logger.experiment.log(
            #         {
            #             f"{name}_target_stem{k}": wandb.Audio(
            #                 self.tensor2numpy(target_waveforms)[i,k], caption= f"Stem {k}: {fnames[i]} {text[i]}", sample_rate=16000,
            #             ),
            #             f"{name}_generated_stem{k}": wandb.Audio(
            #                 waveform[i,k], caption= f"Stem {k}: {fnames[i]} {text[i]}" , sample_rate=16000,
            #             ),
            #         }
            #     )


            # Log all audio files together
            self.logger.experiment.log(log_dict)




    @torch.no_grad()
    def audio_continuation(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name="waveform",
        use_plms=False,
        **kwargs,
    ):
        assert x_T is None
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        waveform_save_path = os.path.join(self.get_log_dir(), name)
        os.makedirs(waveform_save_path, exist_ok=True)
        print("Waveform save path: ", waveform_save_path)
        with self.ema_scope("Plotting Inpaint"):
            for batch in batchs:
                inputs = self.get_input(
                    batch,
                    self.first_stage_key,
                    return_first_stage_outputs=False,
                    force_c_encode=True,
                    return_original_cond=False,
                    bs=None,
                )
                z = inputs[0]
                c = inputs[1] if len(inputs) > 1 else None
                text = super().get_input(batch, "text")

                # Generate multiple samples
                batch_size = z.shape[0] * n_gen
                c = torch.cat([c] * n_gen, dim=0)
                text = text * n_gen

                if unconditional_guidance_scale != 1.0:
                    unconditional_conditioning = (
                        self.cond_stage_model.get_unconditional_condition(batch_size)
                    )

                fnames = list(super().get_input(batch, "fname"))

                _, h, w = z.shape[0], z.shape[2], z.shape[3]

                mask = torch.ones(batch_size, h * 2, w).to(self.device)
                mask[:, h:, :] = 0
                mask = mask[:, None, ...]

                z = torch.cat([z, torch.zeros_like(z)], dim=2)
                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    mask=mask,
                    use_plms=use_plms,
                    x0=torch.cat([z] * n_gen, dim=0),
                )

                mel = self.decode_first_stage(samples)

                waveform = self.mel_spectrogram_to_waveform(
                    mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
                )

                similarity = self.cond_stage_model.cos_similarity(
                    torch.FloatTensor(waveform).squeeze(1), text
                )

                best_index = []
                for i in range(z.shape[0]):
                    candidates = similarity[i :: z.shape[0]]
                    max_index = torch.argmax(candidates).item()
                    best_index.append(i + max_index * z.shape[0])

                waveform = waveform[best_index]

                print("Similarity between generated audio and text", similarity)
                print("Choose the following indexes:", best_index)

                self.save_waveform(waveform, waveform_save_path, name=fnames)

    @torch.no_grad()
    def generate_inpaint_mask(self, z, stemidx_to_inpaint: List[int]):
        mask = torch.ones_like(z)
        for stem_idx in stemidx_to_inpaint:
            # channel_start = stem_idx * 8  # Calculate the start channel for the instrument
            # channel_end = channel_start + 8  # Calculate the end channel for the instrument
            mask[:, stem_idx, :, :, :] = 0.0  # Mask the channels for the instrument
        return mask


    @torch.no_grad()
    def inpainting(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name="waveform",
        use_plms=False,
        **kwargs,
    ):
        assert x_T is None
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        waveform_save_path = os.path.join(self.get_log_dir(), name)
        os.makedirs(waveform_save_path, exist_ok=True)
        print("Waveform save path: ", waveform_save_path)

        wavefor_target_save_path = os.path.join(self.get_log_dir(), "target_%s" % (self.global_step))
        os.makedirs(wavefor_target_save_path, exist_ok=True)
        print("\nWaveform target save path: ", wavefor_target_save_path)      



        with self.ema_scope("Plotting Inpaint"):
            for batch in batchs:
                inputs = self.get_input(
                    batch,
                    self.first_stage_key,
                    return_first_stage_outputs=False,
                    force_c_encode=True,
                    return_original_cond=False,
                    bs=None,
                )
                z = inputs[0]
                c = inputs[1] if len(inputs) > 1 else None
                # text = super().get_input(batch, "text")

                # # Generate multiple samples
                # batch_size = z.shape[0] * n_gen
                # c = torch.cat([c] * n_gen, dim=0)
                # text = text * n_gen

                # if unconditional_guidance_scale != 1.0:
                #     unconditional_conditioning = (
                #         self.cond_stage_model.get_unconditional_condition(batch_size)
                #     )

                if self.cond_stage_model is not None:

                    # Generate multiple samples
                    batch_size = z.shape[0] * n_gen

                    if self.cond_stage_model.embed_mode == "text":
                        text = super().get_input(batch, "text")
                       
                        if c is not None:
                            c = torch.cat([c] * n_gen, dim=0)
                        text = text * n_gen
                    elif self.cond_stage_model.embed_mode == "audio":
                        text = super().get_input(batch, "waveform")

                        if c is not None:
                            c = torch.cat([c] * n_gen, dim=0)
                        text = torch.cat([text] * n_gen, dim=0)

                    if unconditional_guidance_scale != 1.0:
                        unconditional_conditioning = (
                            self.cond_stage_model.get_unconditional_condition(batch_size)
                        )
                else:
                    batch_size = z.shape[0]
                    text = None


                fnames = list(super().get_input(batch, "fname"))

                # _, h, w = z.shape[0], z.shape[2], z.shape[3]

                # mask = torch.ones(batch_size, h, w).to(self.device)
                # mask[:, h // 4 : 3 * (h // 4), :] = 0
                # mask = mask[:, None, ...]

                mask = self.generate_inpaint_mask(z, kwargs["stemidx_to_inpaint"])


                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    mask=mask,
                    use_plms=use_plms,
                    x0=torch.cat([z] * n_gen, dim=0),
                )

                # mel = self.decode_first_stage(samples)

                # waveform = self.mel_spectrogram_to_waveform(
                #     mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
                # )

                # similarity = self.cond_stage_model.cos_similarity(
                #     torch.FloatTensor(waveform).squeeze(1), text
                # )

                # best_index = []
                # for i in range(z.shape[0]):
                #     candidates = similarity[i :: z.shape[0]]
                #     max_index = torch.argmax(candidates).item()
                #     best_index.append(i + max_index * z.shape[0])

                # waveform = waveform[best_index]

                # print("Similarity between generated audio and text", similarity)
                # print("Choose the following indexes:", best_index)

                # self.save_waveform(waveform, waveform_save_path, name=fnames)

                samples = self.adapt_latent_for_VAE_decoder(samples)
                mel = self.decode_first_stage(samples)

                waveform = self.mel_spectrogram_to_waveform(
                    mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
                )

                waveform = np.nan_to_num(waveform)
                waveform = np.clip(waveform, -1, 1)

                # downmix to songs for comparison
                waveform_reshaped = waveform.reshape(batch_size, self.num_stems, waveform.shape[-1])
                
                target_waveforms = super().get_input(batch, 'waveform_stems')

                waveform_reshaped = waveform_reshaped[:, :, :target_waveforms.shape[-1]] # trancate generated wvaeform because vocoder egenretas audio 32 samples longer than we need :))
                # Replace waveforms for all stems not in kwargs["stemidx_to_inpaint"]
                for idx in range(self.num_stems):
                    if idx not in kwargs["stemidx_to_inpaint"]:        
                        waveform_reshaped[:, idx, :] = target_waveforms[:, idx, :].cpu().numpy()
                waveform = waveform_reshaped.reshape(batch_size * self.num_stems, 1, waveform_reshaped.shape[-1])

                mix = waveform_reshaped.sum(axis=1)
               
                mix = np.nan_to_num(mix)
                mix = np.clip(mix, -1, 1)


                if self.model.conditioning_key is not None:
                    if self.cond_stage_model.embed_mode == "text": # TODO maybe make similar for audio (???)
                        similarity = self.cond_stage_model.cos_similarity(
                            torch.FloatTensor(mix).squeeze(1), text
                        )

                        best_index = []
                        for i in range(z.shape[0]):
                            candidates = similarity[i :: z.shape[0]]
                            max_index = torch.argmax(candidates).item()
                            best_index.append(i + max_index * z.shape[0])
                            # print("Similarity between generated audio and text", similarity)
                            # print("Choose the following indexes:", best_index)
                    else:
                        best_index = torch.arange(z.shape[0])
                else:
                    best_index = torch.arange(z.shape[0])

                # chose best scored mixes
                mix = mix[best_index]

                # chose coresponding stems audios and mels:
                selected_wavs = []
                selected_mels = []
                for start_index in best_index:

                    actual_start_index = start_index * self.num_stems

                    # Ensure the selection does not exceed array bounds
                    selected_slice = waveform[actual_start_index:actual_start_index + self.num_stems]
                    selected_wavs.append(selected_slice)

                    selected_slice = mel[actual_start_index:actual_start_index + self.num_stems].cpu().detach().numpy()
                    selected_mels.append(selected_slice)

                    
                waveform = np.concatenate(selected_wavs, axis=0)[:,0,:]
                waveform = waveform.reshape(z.shape[0], self.num_stems, waveform.shape[-1]) # back to batch size and multicahnnel

                # test_names =  [str(number) for number in range(4)]                
                # self.save_waveform(waveform[1][:, np.newaxis, :], "/home/karchkhadze/MusicLDM-Ext/test_folder", name=test_names)


                mel = np.concatenate(selected_mels, axis=0)[:,0,:]
                mel = mel.reshape(z.shape[0], self.num_stems, mel.shape[-2], mel.shape[-1]) # back to batch size and multicahnnel

                ############################# saving audios for metrics ##################################
                # save mixes
                    #generated
                generated_mix_dir = os.path.join(waveform_save_path, "mix")
                os.makedirs(generated_mix_dir, exist_ok=True)
                self.save_waveform(mix[:, np.newaxis, :], generated_mix_dir, name=fnames)
                    #target
                target_mix_dir = os.path.join(wavefor_target_save_path, "mix")
                os.makedirs(target_mix_dir, exist_ok=True)
                target_mix = super().get_input(batch, 'waveform')
                self.save_waveform(target_mix.unsqueeze(1).cpu().detach(), target_mix_dir, name=fnames)

                # save stems
                # target_waveforms = super().get_input(batch, 'waveform_stems')
                # for i in range(self.num_stems):
                    
                #     # generated
                #     generated_stem_dir = os.path.join(os.path.join(waveform_save_path, "stem_"+str(i)))
                #     os.makedirs(generated_stem_dir, exist_ok=True)                    
                #     self.save_waveform(waveform[:,i,:][:, np.newaxis, :], generated_stem_dir, name=fnames)

                #     # target
                #     target_stem_dir = os.path.join(os.path.join(wavefor_target_save_path, "stem_"+str(i)))
                #     os.makedirs(target_stem_dir, exist_ok=True)                    
                #     self.save_waveform(target_waveforms[:,i,:].unsqueeze(1).cpu().detach(), target_stem_dir, name=fnames)


                # for i in range(z.shape[0]):
                    
                # generated
                generated_stem_dir = os.path.join(os.path.join(waveform_save_path, "stem_"+"_".join(map(str, kwargs["stemidx_to_inpaint"]))))
                os.makedirs(generated_stem_dir, exist_ok=True)  
                generated_stems = waveform[:,kwargs["stemidx_to_inpaint"],:][:, np.newaxis, :].sum(-2)                  
                self.save_waveform(generated_stems, generated_stem_dir, name=fnames)

                # target
                target_stem_dir = os.path.join(os.path.join(wavefor_target_save_path, "stem_"+"_".join(map(str, kwargs["stemidx_to_inpaint"]))))
                os.makedirs(target_stem_dir, exist_ok=True)    
                target_stems = target_waveforms[:,kwargs["stemidx_to_inpaint"],:].unsqueeze(1).sum(-2).cpu().detach()          
                self.save_waveform(target_stems, target_stem_dir, name=fnames)
                ###################################### logging ##############################################     
                if self.logger is not None:
                    # create new list
                    log_data_batch = mel, generated_stems, target_stems, mix, target_mix, fnames, batch
                    self.log_images_audios_inpaint(log_data_batch)

    def log_images_audios_inpaint(self, log_data_batch):
        mel, waveform, target_waveforms, mix, target_mix, fnames, batch = log_data_batch

        # Use get to safely access "text" from batch, defaulting to a list of empty string if not found
        text = batch.get("text", [""] * mel.shape[0])

        # get target mel
        target_mel = self.tensor2numpy(batch['fbank_stems'])   

        name = "val"

        ### logginh spectrograms ###
        for i in range(mel.shape[0]):
            self.logger.log_image(
                "Mel_specs %s" % name,
                [np.concatenate([np.flipud(target_mel[i,j].T) for j in range(target_mel[i].shape[0])], axis=0), 
                 np.concatenate([np.flipud(mel[i,j].T) for j in range(mel[i].shape[0])], axis=0) ],

                caption=["target_fbank_%s" % fnames[i]+text[i], "generated_%s" %fnames[i]+text[i]],
            )

            ### logging audios ###

            log_dict = {}

            log_dict ["target_%s"% name] =  wandb.Audio(
                        self.tensor2numpy(target_mix)[i], caption= f"Full Song: {fnames[i]} {text[i]}", sample_rate=16000,)
            log_dict ["generated_%s"% name] =wandb.Audio(
                        mix[i], caption= f"Full Song: {fnames[i]} {text[i]}", sample_rate=16000,)

            # for k in range(self.num_stems):
            log_dict[f"{name}_target_stem"] = wandb.Audio(
                        self.tensor2numpy(target_waveforms)[i,0], caption= f"Stem: {fnames[i]} {text[i]}", sample_rate=16000,)
            log_dict[f"{name}_generated_stem"] = wandb.Audio(
                        waveform[i,0], caption= f"Stem: {fnames[i]} {text[i]}" , sample_rate=16000,)

            # Log all audio files together
            self.logger.experiment.log(log_dict)


    @torch.no_grad()
    def inpainting_half(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name="waveform",
        use_plms=False,
        **kwargs,
    ):
        assert x_T is None
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        waveform_save_path = os.path.join(self.get_log_dir(), name)
        os.makedirs(waveform_save_path, exist_ok=True)
        print("Waveform save path: ", waveform_save_path)
        with self.ema_scope("Plotting Inpaint"):
            for batch in batchs:
                z, c = self.get_input(
                    batch,
                    self.first_stage_key,
                    return_first_stage_outputs=False,
                    force_c_encode=True,
                    return_original_cond=False,
                    bs=None,
                )
                text = super().get_input(batch, "text")

                # Generate multiple samples
                batch_size = z.shape[0] * n_gen
                c = torch.cat([c] * n_gen, dim=0)
                text = text * n_gen

                if unconditional_guidance_scale != 1.0:
                    unconditional_conditioning = (
                        self.cond_stage_model.get_unconditional_condition(batch_size)
                    )

                fnames = list(super().get_input(batch, "fname"))

                _, h, w = z.shape[0], z.shape[2], z.shape[3]

                mask = torch.ones(batch_size, h, w).to(self.device)
                mask[:, int(h * 0.325) :, :] = 0
                mask = mask[:, None, ...]

                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    mask=mask,
                    use_plms=use_plms,
                    x0=torch.cat([z] * n_gen, dim=0),
                )

                mel = self.decode_first_stage(samples)

                waveform = self.mel_spectrogram_to_waveform(
                    mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
                )

                similarity = self.cond_stage_model.cos_similarity(
                    torch.FloatTensor(waveform).squeeze(1), text
                )

                best_index = []
                for i in range(z.shape[0]):
                    candidates = similarity[i :: z.shape[0]]
                    max_index = torch.argmax(candidates).item()
                    best_index.append(i + max_index * z.shape[0])

                waveform = waveform[best_index]

                print("Similarity between generated audio and text", similarity)
                print("Choose the following indexes:", best_index)

                self.save_waveform(waveform, waveform_save_path, name=fnames)

    @torch.no_grad()
    def super_resolution(
        self,
        batchs,
        ddim_steps=200,
        ddim_eta=1.0,
        x_T=None,
        n_gen=1,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        name="waveform",
        use_plms=False,
        **kwargs,
    ):
        assert x_T is None
        try:
            batchs = iter(batchs)
        except TypeError:
            raise ValueError("The first input argument should be an iterable object")

        if use_plms:
            assert ddim_steps is not None

        use_ddim = ddim_steps is not None
        waveform_save_path = os.path.join(self.get_log_dir(), name)
        os.makedirs(waveform_save_path, exist_ok=True)
        print("Waveform save path: ", waveform_save_path)
        with self.ema_scope("Plotting Inpaint"):
            for batch in batchs:
                z, c = self.get_input(
                    batch,
                    self.first_stage_key,
                    return_first_stage_outputs=False,
                    force_c_encode=True,
                    return_original_cond=False,
                    bs=None,
                )
                text = super().get_input(batch, "text")

                # Generate multiple samples
                batch_size = z.shape[0] * n_gen
                c = torch.cat([c] * n_gen, dim=0)
                text = text * n_gen

                if unconditional_guidance_scale != 1.0:
                    unconditional_conditioning = (
                        self.cond_stage_model.get_unconditional_condition(batch_size)
                    )

                fnames = list(super().get_input(batch, "fname"))

                _, h, w = z.shape[0], z.shape[2], z.shape[3]

                mask = torch.ones(batch_size, h, w).to(self.device)
                mask[:, :, 3 * (w // 4) :] = 0
                mask = mask[:, None, ...]

                samples, _ = self.sample_log(
                    cond=c,
                    batch_size=batch_size,
                    x_T=x_T,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    mask=mask,
                    use_plms=use_plms,
                    x0=torch.cat([z] * n_gen, dim=0),
                )

                mel = self.decode_first_stage(samples)

                waveform = self.mel_spectrogram_to_waveform(
                    mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
                )

                similarity = self.cond_stage_model.cos_similarity(
                    torch.FloatTensor(waveform).squeeze(1), text
                )

                best_index = []
                for i in range(z.shape[0]):
                    candidates = similarity[i :: z.shape[0]]
                    max_index = torch.argmax(candidates).item()
                    best_index.append(i + max_index * z.shape[0])

                waveform = waveform[best_index]

                print("Similarity between generated audio and text", similarity)
                print("Choose the following indexes:", best_index)

                self.save_waveform(waveform, waveform_save_path, name=fnames)


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        # debug disabled
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [
            None,
            "concat",
            "crossattn",
            "hybrid",
            "adm",
            "film",
        ]

    def forward(
        self, x, t, c_concat: list = None, c_crossattn: list = None, c_film: list = None
    ):
        x = x.contiguous()
        t = t.contiguous()
        # print("conditioning_key:", self.conditioning_key)
        # Debug：前幾步觀察輸入與 cond 型態/維度
        # debug disabled
        if self.conditioning_key is None:
            # 若外部仍提供了 c_film，則一併傳入 y
            if c_film is not None and len(c_film) > 0:
                out = self.diffusion_model(x, t, y=c_film[0].squeeze(1) if c_film[0].dim()==3 else c_film[0])
            else:
                out = self.diffusion_model(x, t)
        elif self.conditioning_key == "concat":
            # 修正：根據x的維度決定concat的維度
            if len(x.shape) == 5:
                # 5D: [batch, stems, z_channels, t, f] -> 在z_channels維度concat
                # print(x.shape, c_concat.shape)
                xc = torch.cat([x] + c_concat, dim=2)
            else:
                # print(f"x.shape: {x.shape}")
                # print(f"type c_concat: {type(c_concat)}")
                # print(f"len(c_concat): {len(c_concat)}")
                # print(f"c_concat.shape: {c_concat[0].shape}")
                # 4D: [batch, z_channels, t, f] -> 在z_channels維度concat
                xc = torch.cat([x] + c_concat, dim=1)
            if c_film is not None and len(c_film) > 0:
                out = self.diffusion_model(xc, t, y=c_film[0].squeeze(1) if c_film[0].dim()==3 else c_film[0])
            else:
                out = self.diffusion_model(xc, t)
        elif self.conditioning_key == "crossattn":
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == "hybrid":
            # 修正：根據x的維度決定concat的維度
            if len(x.shape) == 5:
                # 5D: [batch, stems, z_channels, t, f] -> 在z_channels維度concat
                xc = torch.cat([x] + c_concat, dim=2)
            else:
                # 4D: [batch, z_channels, t, f] -> 在z_channels維度concat
                xc = torch.cat([x] + c_concat, dim=1)
                cc = torch.cat(c_crossattn, 1)
            if c_film is not None and len(c_film) > 0:
                out = self.diffusion_model(xc, t, context=cc, y=c_film[0].squeeze(1) if c_film[0].dim()==3 else c_film[0])
            else:
                cc = torch.cat(c_crossattn, 1)
                out = self.diffusion_model(xc, t, context=cc)
        elif (
            self.conditioning_key == "film"
        ):  # The condition is assumed to be a global token, which wil pass through a linear layer and added with the time embedding for the FILM
            cc = c_film[0].squeeze(1)  # only has one token
            out = self.diffusion_model(x, t, y=cc)
        elif self.conditioning_key == "adm":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        # Debug：僅前幾步印出輸出資訊（包含是否為 tuple）
        # debug disabled

        return out

class CLAPResidualVQWrapper(pl.LightningModule):
    def __init__(self, clap_rvq_config, clap_model):
        super().__init__()
        self.claprvq = CLAPResidualVQ(**clap_rvq_config, clap_wrapper=clap_model)
        self.clap_rvq_config = clap_rvq_config

    def training_step(self, x):
        print(self.training, self.claprvq.training, self.claprvq.rvq.training)
        loss, _, _ = self.claprvq(x, is_text = self.clap_rvq_config['data_type'] == 'text')
        self.log('loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, x):
        self.eval()
        with torch.no_grad():
            loss, _, _ = self.claprvq(x, is_text = self.clap_rvq_config['data_type'] == 'text')
            self.log('valid_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(None, lr = 0.)
        return optimizer


class MusicLDM_multitask(MusicLDM):
    """最小差異版本：僅在 MusicLDM 基礎上增加 onset/timbre 兩個多任務分支的控制與損失。
    其餘訓練/取樣/資料流與 MusicLDM 完全一致。
    """

    def __init__(self, *args, enable_onset_prediction=False, enable_timbre_prediction=False,
                 onset_loss_weight=1.0, timbre_loss_weight=1.0, onset_warmup_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        # 多任務旗標與權重
        self.enable_onset_prediction = enable_onset_prediction
        self.enable_timbre_prediction = enable_timbre_prediction
        self.onset_loss_weight = onset_loss_weight
        self.timbre_loss_weight = timbre_loss_weight
        self.onset_warmup_steps = onset_warmup_steps or 0
        # 暫存 UNet 分支輸出
        self._mt_last_onset = None
        self._mt_last_timbre = None

    def apply_model(self, x_noisy, t, cond, return_ids=False, guides=None, film_vec=None):
        # 直接仿照父類處理 cond，但不截斷 UNet 的 tuple 輸出，方便多任務分支取用
        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            if self.model.conditioning_key == "concat":
                key = "c_concat"
            elif self.model.conditioning_key == "crossattn":
                key = "c_crossattn"
            else:
                key = "c_film"
            cond = {key: cond}

        # 處理 FiLM 條件（來自 C0、C1 的全域平均）
        if hasattr(self, 'enable_guide_features') and self.enable_guide_features and film_vec is not None:
            # 創建一個小的 MLP 來處理 film_vec，輸出 gamma 和 beta
            if not hasattr(self, 'film_mlp'):
                Cg = film_vec.shape[1] if len(film_vec.shape) > 1 else 4
                # 從配置中獲取參數，如果沒有則使用默認值
                extra_film_condition_dim = self.guide_features_config.get('film_dim', 32)
                z_channels = getattr(self, 'z_channels', 8)  # 獲取 VAE 的 z_channels
                self.film_mlp = torch.nn.Sequential(
                    torch.nn.Linear(Cg, extra_film_condition_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(extra_film_condition_dim, 2 * z_channels)  # gamma + beta
                ).to(self.device)
                # 零初始化 FiLM 的最後一層，避免初始隨機放大
                with torch.no_grad():
                    self.film_mlp[-1].weight.zero_()
                    self.film_mlp[-1].bias.zero_()
            
            # 處理 film_vec，輸出 gamma 和 beta
            film_params = self.film_mlp(film_vec)  # [B, 2*z_channels]
            
            # 將 film_params 添加到條件中作為額外的 FiLM 條件
            # 注意：這裡不改變原本的 mix concat 邏輯，只是添加額外的 FiLM 條件
            cond['c_film'] = [film_params]

        x_recon = self.model(x_noisy, t, **cond)

        # 重置暫存
        self._mt_last_onset = None
        self._mt_last_timbre = None

        # 僅在 rank0 且前幾步印出一次偵錯資訊
        try:
            _rank0 = (not hasattr(self, "global_rank")) or (self.global_rank == 0)
            _gstep = int(getattr(self, "global_step", 0))
        except Exception:
            _rank0, _gstep = True, 0

        if isinstance(x_recon, tuple):
            main_out = x_recon[0]
            # 依輸出形狀自動對應 onset/timbre，支援兩分支同時開啟
            if len(x_recon) >= 2:
                extras = x_recon[1:]
                onset_T = None
                timbre_D = None
                try:
                    if hasattr(self, 'model') and hasattr(self.model, 'diffusion_model'):
                        onset_T = getattr(self.model.diffusion_model, 'onset_output_frames', None)
                        timbre_D = getattr(self.model.diffusion_model, 'timbre_feature_dim', None)
                except Exception:
                    pass

                # 優先依據最後一維長度做匹配（onset→T=1024，timbre→D=7）
                for e in extras:
                    try:
                        last = e.shape[-1]
                    except Exception:
                        last = None
                    if self.enable_onset_prediction and onset_T is not None and last == onset_T and self._mt_last_onset is None:
                        self._mt_last_onset = e
                        continue
                    if self.enable_timbre_prediction and timbre_D is not None and last == timbre_D and self._mt_last_timbre is None:
                        self._mt_last_timbre = e

                # 後備：若仍未識別，依開關判斷或預設將單一路徑視為 onset
                if len(extras) == 1 and self._mt_last_onset is None and self._mt_last_timbre is None:
                    if self.enable_timbre_prediction and not self.enable_onset_prediction:
                        self._mt_last_timbre = extras[0]
                    elif self.enable_onset_prediction and not self.enable_timbre_prediction:
                        self._mt_last_onset = extras[0]
                    else:
                        self._mt_last_onset = extras[0]

            # debug disabled

            # 與父類一致：預設回傳主輸出（除非顯式要求 return_ids）
            if not return_ids:
                return main_out
            return x_recon
        else:
            # debug disabled
            return x_recon

    def p_losses(self, x_start, cond, t, noise=None, *args, **kwargs):
        # 先清空暫存分支輸出，讓本次 forward/反傳時 apply_model 重新填入
        self._mt_last_onset = None
        self._mt_last_timbre = None
        # 先走父類，拿到 diffusion 的主損失（父類會呼叫本類的 apply_model）
        base_loss, loss_dict = super().p_losses(x_start, cond, t, noise=noise, *args, **kwargs)

        total_loss = base_loss
        batch = kwargs.get('batch')
        prefix = "train" if self.training else "val"

        # Onset 分支損失（以 logits 用 focal loss）
        if self.enable_onset_prediction and (self._mt_last_onset is not None) and batch is not None:
            # debug disabled
            gt_onset = batch.get('onset_pianoroll')
            if gt_onset is not None:
                if not isinstance(gt_onset, torch.Tensor):
                    gt_onset = torch.tensor(gt_onset)
                gt_onset = gt_onset.to(self.device, dtype=torch.float32)
                onset_logits = self._mt_last_onset
                # 尺寸對齊：B,S,T
                while onset_logits.dim() > 3:
                    onset_logits = onset_logits.mean(dim=2)
                T = min(onset_logits.shape[-1], gt_onset.shape[-1])
                onset_logits = onset_logits[..., :T]
                gt_onset = gt_onset[..., :T]
                onset_loss = binary_focal_loss_with_logits(onset_logits, gt_onset, alpha=0.25, gamma=2.0, reduction='mean')
                if self.training and self.onset_warmup_steps > 0:
                    current_step = getattr(self, 'global_step', 0)
                    warm = min(1.0, float(current_step) / float(self.onset_warmup_steps))
                else:
                    warm = 1.0
                total_loss = total_loss + self.onset_loss_weight * warm * onset_loss
                loss_dict.update({f"{prefix}/loss_onset": float(onset_loss.item())})
                # 讓進度條也能看到 onset loss（與主 loss 類似的 on_step/on_epoch 行為）
                self.log(f"{prefix}/onset_loss_step", onset_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log(f"{prefix}/onset_loss", onset_loss.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True)

        # Timbre 分支損失（以 MSE）
        # 放寬條件：只要開啟 timbre 分支且 batch 有 GT，就嘗試從多任務輸出或 UNet 分支抓預測
        if self.enable_timbre_prediction and batch is not None:
            gt_timbre = batch.get('timbre_features')
            if gt_timbre is not None:
                gt_timbre = gt_timbre.to(self.device, dtype=torch.float32)
                timbre_pred = self._mt_last_timbre
                if timbre_pred is not None:
                    timbre_loss = torch.nn.functional.mse_loss(timbre_pred, gt_timbre, reduction='mean')
                    total_loss = total_loss + self.timbre_loss_weight * timbre_loss
                    # 同時寫入兩個鍵：與 p_losses 命名、與 training_step 檢查相容
                    loss_dict.update({
                        f"{prefix}/loss_timbre": float(timbre_loss.item()),
                        "timbre_loss": float(timbre_loss.item()),
                    })
                    # 直接記錄，讓 train/val 都能看到
                    self.log(f"{prefix}/timbre_loss_step", timbre_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
                    self.log(f"{prefix}/timbre_loss", timbre_loss.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
                    # try:
                    #     print(f"[{prefix.upper()}] Timbre Loss: {float(timbre_loss.item()):.4f}")
                    # except Exception:
                    #     pass

        return total_loss, loss_dict


class MusicLDM_multitask_gradnorm(MusicLDM):
    """最小差異版本：僅在 MusicLDM 基礎上增加 onset/timbre 兩個多任務分支的控制與損失。
    其餘訓練/取樣/資料流與 MusicLDM 完全一致。
    """

    def __init__(self, *args, enable_onset_prediction=False, enable_timbre_prediction=False,
                 onset_loss_weight=1.0, timbre_loss_weight=1.0, onset_warmup_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        # 多任務旗標與權重
        self.enable_onset_prediction = enable_onset_prediction
        self.enable_timbre_prediction = enable_timbre_prediction
        self.onset_loss_weight = onset_loss_weight
        self.timbre_loss_weight = timbre_loss_weight
        self.onset_warmup_steps = onset_warmup_steps or 0
        # 暫存 UNet 分支輸出
        self._mt_last_onset = None
        self._mt_last_timbre = None
        # 任務名：包含 diffusion 主任務 + 依旗標加入 onset/timbre
        self._tasks = ["diff"]
        if self.enable_onset_prediction: self._tasks.append("onset")
        if self.enable_timbre_prediction: self._tasks.append("timbre")
        
        # 可學習的權重（每個任務一個），初始化為 1
        self._task_weights = torch.nn.Parameter(torch.ones(len(self._tasks), device=self.device))
        self._tw_optim = torch.optim.Adam([self._task_weights], lr=1e-3)

        # GradNorm 參數
        self._gradnorm_alpha = 1.0    # 建議先用 1.0；可測 0~3
        self._L0 = {t: None for t in self._tasks}  # 初始損失
        self._shared_param = self._pick_shared_parameter()  # 共享主幹參數（見下）
        # 安全選項與頻率
        self._gradnorm_update_every = 1
        self._gradnorm_safe_mode = True
        self._gradnorm_warned = False

        # 為避免與自定義 gradient checkpoint backward 衝突，直接關閉模型內所有 checkpoint 開關
        # （注意：會增加顯存佔用）
        self._disable_all_checkpoints_for_gradnorm()
    
    def _current_weights_vector(self):
        # 正規化（總和=任務數），避免權重爆炸/塌縮
        with torch.no_grad():
            w = torch.relu(self._task_weights) + 1e-6
            w = w * (len(self._tasks) / w.sum())
        return w

    def _gradnorm_update(self, task_to_loss: dict):
        # 可降低頻率，減少額外開銷
        step = int(getattr(self, 'global_step', 0))
        if (self._gradnorm_update_every is not None) and (self._gradnorm_update_every > 1):
            if step % int(self._gradnorm_update_every) != 0:
                return
        # 初始化 L0
        for task, loss in task_to_loss.items():
            if self._L0[task] is None:
                self._L0[task] = float(loss.detach().item())

        # 對齊順序
        tasks = self._tasks
        losses = [task_to_loss[t] for t in tasks]
        L0 = torch.tensor([self._L0[t] for t in tasks], device=self.device)

        # 相對訓練速率 r_i
        with torch.no_grad():
            Li = torch.tensor([float(l.detach().item()) for l in losses], device=self.device)
            ri = (Li / (L0 + 1e-8)).clamp(min=0.0) ** self._gradnorm_alpha

        # 每個任務的基礎梯度範數 gbar_i = ||∂L_i/∂θ_shared||
        # 嘗試以共享參數計算梯度範數；如遇 checkpoint 衝突，回退到各任務 head 參數或跳過本步
        gbar_values = []
        try:
            for l in losses:
                g = torch.autograd.grad(l, self._shared_param, retain_graph=True, create_graph=False, allow_unused=True)[0]
                if g is None:
                    raise RuntimeError("shared_param has no grad for a task")
                gbar_values.append(g.norm(p=2))
        except Exception as e:
            # 回退：改用每個任務對應的 head 參數作為代理，避免穿過 checkpoint 的主幹
            if self._gradnorm_safe_mode:
                if (not self._gradnorm_warned) and (getattr(self, 'global_rank', 0) == 0):
                    print(f"[GradNorm] Fallback to head params due to checkpoint conflict: {e}")
                    self._gradnorm_warned = True
                gbar_values = []
                for t, l in zip(tasks, losses):
                    proxy_p = self._pick_head_parameter(t)
                    if proxy_p is None:
                        # 無可用代理參數，放棄本步更新
                        return
                    g = torch.autograd.grad(l, proxy_p, retain_graph=True, create_graph=False, allow_unused=True)[0]
                    if g is None:
                        return
                    gbar_values.append(g.norm(p=2))
            else:
                # 安全模式關閉時，直接略過本步
                return
        gbar = torch.stack(gbar_values).detach()

        # 目前權重（正規化到總和=任務數）
        with torch.no_grad():
            w = torch.relu(self._task_weights) + 1e-6
            w = w * (len(tasks) / w.sum())

            G = w * gbar
            G_mean = G.mean()
            G_target = ri * G_mean

            # dL/dw_i ≈ sign(G_i - G*_i) * gbar_i   （忽略 G_mean 對 w 的依賴）
            grad_w = torch.sign(G - G_target) * gbar

            # 手動 SGD 更新 task weights
            self._task_weights.data -= 1e-3 * grad_w    # 可調：lr=1e-3
            self._task_weights.data.clamp_(min=1e-6)

            # 重新正規化
            w = torch.relu(self._task_weights.data) + 1e-6
            self._task_weights.data.copy_(w * (len(tasks) / w.sum()))

    def _pick_shared_parameter(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.dim() >= 2 and all(k not in name.lower() for k in ("onset", "timbre", "head", "out")):
                return p
        return next(self.model.parameters()) 

    def _disable_all_checkpoints_for_gradnorm(self):
        try:
            for m in self.modules():
                if hasattr(m, 'checkpoint'):
                    try:
                        m.checkpoint = False
                    except Exception:
                        pass
                if hasattr(m, 'use_checkpoint'):
                    try:
                        m.use_checkpoint = False
                    except Exception:
                        pass
        except Exception:
            pass

    def _pick_head_parameter(self, task: str):
        """為每個任務選擇一個盡量靠近輸出的代理參數，避免經過主幹 checkpoint。"""
        task = (task or '').lower()
        # 優先匹配專屬 head
        keywords = []
        if task == 'onset':
            keywords = ['onset']
        elif task == 'timbre':
            keywords = ['timbre']
        else:  # diff 主任務：選 UNet out 層
            keywords = ['out']
        for name, p in self.model.named_parameters():
            n = name.lower()
            if p.requires_grad and any(k in n for k in keywords):
                return p
        # 後備：回傳任一可訓練的 2D 權重
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.dim() >= 2:
                return p
        return None

    def apply_model(self, x_noisy, t, cond, return_ids=False, guides=None, film_vec=None):
        # 直接仿照父類處理 cond，但不截斷 UNet 的 tuple 輸出，方便多任務分支取用
        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            if self.model.conditioning_key == "concat":
                key = "c_concat"
            elif self.model.conditioning_key == "crossattn":
                key = "c_crossattn"
            else:
                key = "c_film"
            cond = {key: cond}

        # 導引圖：僅以 FiLM 方式添加，不把 guides 當 kw 傳入 UNet
        if hasattr(self, 'enable_guide_features') and self.enable_guide_features and (film_vec is not None):
            # 構建 / 使用 FiLM MLP
            if not hasattr(self, 'film_mlp'):
                Cg = film_vec.shape[1] if len(film_vec.shape) > 1 else 4
                extra_film_condition_dim = int(self.guide_features_config.get('film_dim', 32))
                self.film_mlp = torch.nn.Sequential(
                    torch.nn.Linear(Cg, extra_film_condition_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(extra_film_condition_dim, extra_film_condition_dim)
                ).to(self.device)
            film_condition = self.film_mlp(film_vec)
            # 在 cond 中補上 c_film，讓 UNet 能同時吃 concat 的 mix 與 FiLM
            if 'c_film' in cond and isinstance(cond['c_film'], list):
                cond['c_film'].append(film_condition)
            else:
                cond['c_film'] = [film_condition]

        x_recon = self.model(x_noisy, t, **cond)

        # 重置暫存
        self._mt_last_onset = None
        self._mt_last_timbre = None

        # 僅在 rank0 且前幾步印出一次偵錯資訊
        try:
            _rank0 = (not hasattr(self, "global_rank")) or (self.global_rank == 0)
            _gstep = int(getattr(self, "global_step", 0))
        except Exception:
            _rank0, _gstep = True, 0

        if isinstance(x_recon, tuple):
            main_out = x_recon[0]
            if len(x_recon) >= 3:
                self._mt_last_onset = x_recon[1]
                self._mt_last_timbre = x_recon[2]
            elif len(x_recon) == 2:
                if self.enable_onset_prediction and not self.enable_timbre_prediction:
                    self._mt_last_onset = x_recon[1]
                elif self.enable_timbre_prediction and not self.enable_onset_prediction:
                    self._mt_last_timbre = x_recon[1]
                else:
                    self._mt_last_onset = x_recon[1]

            # debug disabled

            # 與父類一致：預設回傳主輸出（除非顯式要求 return_ids）
            if not return_ids:
                return main_out
            return x_recon
        else:
            # debug disabled
            return x_recon

    def p_losses(self, x_start, cond, t, noise=None, *args, **kwargs):
        # 先清空暫存分支輸出，讓本次 forward/反傳時 apply_model 重新填入
        self._mt_last_onset = None
        self._mt_last_timbre = None
        # 先走父類，拿到 diffusion 的主損失（父類會呼叫本類的 apply_model）
        base_loss, loss_dict = super().p_losses(x_start, cond, t, noise=noise, *args, **kwargs)

        total_loss = base_loss
        batch = kwargs.get('batch')
        prefix = "train" if self.training else "val"

        task_to_loss = {"diff": base_loss}

        # Onset 分支損失（以 logits 用 focal loss）
        if self.enable_onset_prediction and (self._mt_last_onset is not None) and batch is not None:
            # debug disabled
            gt_onset = batch.get('onset_pianoroll')
            if gt_onset is not None:
                if not isinstance(gt_onset, torch.Tensor):
                    gt_onset = torch.tensor(gt_onset)
                gt_onset = gt_onset.to(self.device, dtype=torch.float32)
                onset_logits = self._mt_last_onset
                # 尺寸對齊：B,S,T
                while onset_logits.dim() > 3:
                    onset_logits = onset_logits.mean(dim=2)
                T = min(onset_logits.shape[-1], gt_onset.shape[-1])
                onset_logits = onset_logits[..., :T]
                gt_onset = gt_onset[..., :T]
                onset_loss = binary_focal_loss_with_logits(onset_logits, gt_onset, alpha=0.25, gamma=2.0, reduction='mean')
                if self.training and self.onset_warmup_steps > 0:
                    current_step = getattr(self, 'global_step', 0)
                    warm = min(1.0, float(current_step) / float(self.onset_warmup_steps))
                else:
                    warm = 1.0
                total_loss = total_loss + self.onset_loss_weight * warm * onset_loss
                loss_dict.update({f"{prefix}/loss_onset": float(onset_loss.item())})
                # 讓進度條也能看到 onset loss（與主 loss 類似的 on_step/on_epoch 行為）
                self.log(f"{prefix}/onset_loss_step", onset_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log(f"{prefix}/onset_loss", onset_loss.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
            
            task_to_loss["onset"] = onset_loss

        # Timbre 分支損失（以 MSE）
        if self.enable_timbre_prediction and (self._mt_last_timbre is not None) and batch is not None:
            gt_timbre = batch.get('timbre_features')
            if gt_timbre is not None:
                gt_timbre = gt_timbre.to(self.device, dtype=torch.float32)
                timbre_pred = self._mt_last_timbre
                timbre_loss = torch.nn.functional.mse_loss(timbre_pred, gt_timbre, reduction='mean')
                total_loss = total_loss + self.timbre_loss_weight * timbre_loss
                loss_dict.update({f"{prefix}/loss_timbre": float(timbre_loss.item())})
            
            task_to_loss["timbre"] = timbre_loss

        # 只在訓練狀態跑權重更新
        if self.training and len(task_to_loss) > 1:
            self._gradnorm_update(task_to_loss)

        # 依照學到的權重做加權總損
        w = self._current_weights_vector()  # 依 self._tasks 對齊成 tensor
        total_loss = 0.0
        for i, task in enumerate(self._tasks):
            if task in task_to_loss:
                total_loss = total_loss + w[i] * task_to_loss[task]
        # 記錄權重以便觀察
        for i, task in enumerate(self._tasks):
            self.log(f"train/gradnorm_w_{task}", float(w[i].item()), on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return total_loss, loss_dict


class MusicLDM_multitask_dwa(MusicLDM):
    """多任務（diffusion / onset / timbre）使用 DWA（Dynamic Weight Average）自動調權重。
    - 不使用 autograd.grad、不觸碰主幹圖，避開 gradient checkpoint 衝突。
    - 權重每個 epoch 依最近兩個 epoch 的任務損失比值更新。
    """

    def __init__(self, *args, enable_onset_prediction=False, enable_timbre_prediction=False,
                 onset_loss_weight=1.0, timbre_loss_weight=1.0, onset_warmup_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        from collections import defaultdict, deque

        self.enable_onset_prediction = enable_onset_prediction
        self.enable_timbre_prediction = enable_timbre_prediction
        self.onset_warmup_steps = onset_warmup_steps or 0

        # 任務清單
        self._tasks = ["diff"]
        if self.enable_onset_prediction:
            self._tasks.append("onset")
        if self.enable_timbre_prediction:
            self._tasks.append("timbre")

        # DWA 狀態
        self._dwa_T = 2.0
        self._hist = {t: deque(maxlen=2) for t in self._tasks}        # 每任務最近兩個 epoch loss
        self._epoch_sums = {t: 0.0 for t in self._tasks}
        self._epoch_counts = {t: 0 for t in self._tasks}
        self._weights = torch.ones(len(self._tasks), device=self.device)  # 當前權重，總和未必=任務數

        # 暫存 UNet 分支輸出
        self._mt_last_onset = None
        self._mt_last_timbre = None

    def _current_weights_vector(self):
        with torch.no_grad():
            w = torch.clamp(self._weights, min=1e-6)
            w = w * (len(self._tasks) / w.sum())
        return w

    def on_train_epoch_start(self) -> None:
        # 重置當前 epoch 累積
        for t in self._tasks:
            self._epoch_sums[t] = 0.0
            self._epoch_counts[t] = 0
        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        # 以 epoch 平均 loss 更新 DWA 權重
        means = {}
        for t in self._tasks:
            c = max(1, self._epoch_counts[t])
            means[t] = float(self._epoch_sums[t]) / float(c)
            self._hist[t].append(means[t])

        # 需要每個任務至少兩個歷史點
        if all(len(self._hist[t]) >= 2 for t in self._tasks):
            import math
            ratios = torch.tensor([
                self._hist[t][-1] / (self._hist[t][-2] + 1e-8) for t in self._tasks
            ], device=self.device)
            w = torch.exp(ratios / self._dwa_T)
            w = w * (len(self._tasks) / w.sum())
            self._weights = w

        # 紀錄當前權重
        w_log = self._current_weights_vector()
        for i, t in enumerate(self._tasks):
            try:
                self.log(f"train/dwa_w_{t}", float(w_log[i].item()), on_step=False, on_epoch=True, prog_bar=False, logger=True)
            except Exception:
                pass

        return super().on_train_epoch_end()

    def apply_model(self, x_noisy, t, cond, return_ids=False, guides=None, film_vec=None):
        # 與 gradnorm 版本一致：保留分支輸出
        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            if self.model.conditioning_key == "concat":
                key = "c_concat"
            elif self.model.conditioning_key == "crossattn":
                key = "c_crossattn"
            else:
                key = "c_film"
            cond = {key: cond}

        # 處理 FiLM 條件（來自 C0、C1 的全域平均）
        if hasattr(self, 'enable_guide_features') and self.enable_guide_features and film_vec is not None:
            # 創建一個小的 MLP 來處理 film_vec，輸出 gamma 和 beta
            if not hasattr(self, 'film_mlp'):
                Cg = film_vec.shape[1] if len(film_vec.shape) > 1 else 4
                # 從配置中獲取參數，如果沒有則使用默認值
                extra_film_condition_dim = self.guide_features_config.get('film_dim', 32)
                z_channels = getattr(self, 'z_channels', 8)  # 獲取 VAE 的 z_channels
                self.film_mlp = torch.nn.Sequential(
                    torch.nn.Linear(Cg, extra_film_condition_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(extra_film_condition_dim, 2 * z_channels)  # gamma + beta
                ).to(self.device)
                # 零初始化 FiLM 的最後一層，避免初始隨機放大
                with torch.no_grad():
                    self.film_mlp[-1].weight.zero_()
                    self.film_mlp[-1].bias.zero_()
            
            # 處理 film_vec，輸出 gamma 和 beta
            film_params = self.film_mlp(film_vec)  # [B, 2*z_channels]
            
            # 將 film_params 添加到條件中作為額外的 FiLM 條件
            # 注意：這裡不改變原本的 mix concat 邏輯，只是添加額外的 FiLM 條件
            cond['c_film'] = [film_params]

        x_recon = self.model(x_noisy, t, **cond)

        self._mt_last_onset = None
        self._mt_last_timbre = None

        if isinstance(x_recon, tuple):
            main_out = x_recon[0]
            if len(x_recon) >= 3:
                self._mt_last_onset = x_recon[1]
                self._mt_last_timbre = x_recon[2]
            elif len(x_recon) == 2:
                if self.enable_onset_prediction and not self.enable_timbre_prediction:
                    self._mt_last_onset = x_recon[1]
                elif self.enable_timbre_prediction and not self.enable_onset_prediction:
                    self._mt_last_timbre = x_recon[1]
                else:
                    self._mt_last_onset = x_recon[1]
            return main_out if not return_ids else x_recon
        else:
            return x_recon

    def _accumulate_epoch(self, task_to_loss: dict):
        # 累積當前 step 的各任務 loss
        for t, l in task_to_loss.items():
            if t in self._epoch_sums:
                self._epoch_sums[t] += float(l.detach().item())
                self._epoch_counts[t] += 1

    def p_losses(self, x_start, cond, t, noise=None, *args, **kwargs):
        # 每 step 更新任務 loss，依 DWA 權重組合總損
        self._mt_last_onset = None
        self._mt_last_timbre = None

        base_loss, loss_dict = super().p_losses(x_start, cond, t, noise=noise, *args, **kwargs)
        total_loss = base_loss
        batch = kwargs.get('batch')
        prefix = "train" if self.training else "val"

        task_to_loss = {"diff": base_loss}

        if self.enable_onset_prediction and (self._mt_last_onset is not None) and batch is not None:
            gt_onset = batch.get('onset_pianoroll')
            if gt_onset is not None:
                if not isinstance(gt_onset, torch.Tensor):
                    gt_onset = torch.tensor(gt_onset)
                gt_onset = gt_onset.to(self.device, dtype=torch.float32)
                onset_logits = self._mt_last_onset
                while onset_logits.dim() > 3:
                    onset_logits = onset_logits.mean(dim=2)
                Tlen = min(onset_logits.shape[-1], gt_onset.shape[-1])
                onset_logits = onset_logits[..., :Tlen]
                gt_onset = gt_onset[..., :Tlen]
                onset_loss = binary_focal_loss_with_logits(onset_logits, gt_onset, alpha=0.25, gamma=2.0, reduction='mean')
                if self.training and self.onset_warmup_steps > 0:
                    current_step = getattr(self, 'global_step', 0)
                    warm = min(1.0, float(current_step) / float(self.onset_warmup_steps))
                else:
                    warm = 1.0
                onset_loss = warm * onset_loss
                loss_dict.update({f"{prefix}/loss_onset": float(onset_loss.item())})
                try:
                    self.log(f"{prefix}/onset_loss_step", onset_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
                    self.log(f"{prefix}/onset_loss", onset_loss.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
                except Exception:
                    pass
                task_to_loss["onset"] = onset_loss

        if self.enable_timbre_prediction and (self._mt_last_timbre is not None) and batch is not None:
            gt_timbre = batch.get('timbre_features')
            if gt_timbre is not None:
                gt_timbre = gt_timbre.to(self.device, dtype=torch.float32)
                timbre_pred = self._mt_last_timbre
                timbre_loss = torch.nn.functional.mse_loss(timbre_pred, gt_timbre, reduction='mean')
                loss_dict.update({f"{prefix}/loss_timbre": float(timbre_loss.item())})
                task_to_loss["timbre"] = timbre_loss

        # 累積當前 step 的各任務 loss（供 epoch-end 更新權重）
        if self.training:
            self._accumulate_epoch(task_to_loss)

        # 依 DWA 權重組合總損
        w = self._current_weights_vector()
        total_loss = 0.0
        for i, tname in enumerate(self._tasks):
            if tname in task_to_loss:
                total_loss = total_loss + w[i] * task_to_loss[tname]

        return total_loss, loss_dict
    # """
    # 多任務訓練模型 - 基於 MusicLDM，保留多任務功能
    # 支持 enable_latent_diffusion, enable_onset_prediction, enable_timbre_prediction 控制
    # """

    # def __init__(
    #     self,
    #     first_stage_config,
    #     cond_stage_config=None,
    #     num_timesteps_cond=None,
    #     cond_stage_key="image",
    #     cond_stage_trainable=False,
    #     concat_mode=True,
    #     cond_stage_forward=None,
    #     conditioning_key=None,
    #     scale_factor=1.0,
    #     batchsize=None,
    #     evaluation_params={},
    #     scale_by_std=False,
    #     base_learning_rate=None,
    #     latent_mixup=0.,
    #     num_stems=None,
    #     seperate_stem_z=False,
    #     use_silence_weight=False,
    #     tau=3.0,
    #     # ====== 多任務訓練參數 ======
    #     enable_latent_diffusion=True,  # 是否啟用原本的 latent diffusion 訓練
    #     enable_onset_prediction=True,  # 是否啟用 onset-pianoroll 預測
    #     enable_timbre_prediction=False,  # 是否啟用 timbre 特徵預測
    #     latent_loss_weight=1.0,  # latent diffusion 損失權重
    #     onset_loss_weight=1.0,  # onset loss 權重
    #     timbre_loss_weight=1.0,  # timbre loss 權重
    #     onset_warmup_steps=0,
    #     # 數據路徑配置
    #     onset_data_path=None,
    #     timbre_data_path=None,
    #     track_mapping_path=None,
    #     *args,
    #     **kwargs,
    # ):
    #     print(f"[DEBUG] MusicLDM_multitask.__init__ conditioning_key: {conditioning_key}")
        
    #     # 设置num_timesteps_cond
    #     self.num_timesteps_cond = default(num_timesteps_cond, 1)
        
    #     # 過濾掉不應該傳遞給父類的參數
    #     parent_kwargs = kwargs.copy()
    #     # 移除多任務相關的參數，這些參數不應該傳遞給 DDPM 父類
    #     parent_kwargs.pop('onset_output_frames', None)
    #     parent_kwargs.pop('timbre_feature_dim', None)
    #     parent_kwargs.pop('use_onset_branch', None)
    #     parent_kwargs.pop('use_timbre_branch', None)
    #     parent_kwargs.pop('onset_branch_channels', None)
    #     parent_kwargs.pop('timbre_branch_channels', None)
    #     parent_kwargs.pop('num_stems', None)  # 這個參數也不應該傳遞給父類
    #     # 移除 MusicLDM_multitask 特有的參數
    #     parent_kwargs.pop('use_noise', None)
    #     parent_kwargs.pop('use_random_timesteps', None)
    #     parent_kwargs.pop('fixed_timestep', None)
    #     parent_kwargs.pop('latent_t_size', None)
    #     parent_kwargs.pop('latent_f_size', None)
    #     parent_kwargs.pop('z_channels', None)
    #     parent_kwargs.pop('loss_type', None)
    #     parent_kwargs.pop('monitor', None)
    #     parent_kwargs.pop('use_ema', None)
    #     parent_kwargs.pop('log_every_t', None)
    #     # 移除其他可能冲突的参数
    #     parent_kwargs.pop('enable_latent_diffusion', None)
    #     parent_kwargs.pop('enable_onset_prediction', None)
    #     parent_kwargs.pop('enable_timbre_prediction', None)
    #     parent_kwargs.pop('latent_loss_weight', None)
    #     parent_kwargs.pop('onset_loss_weight', None)
    #     parent_kwargs.pop('timbre_loss_weight', None)
    #     parent_kwargs.pop('onset_data_path', None)
    #     parent_kwargs.pop('timbre_data_path', None)
    #     parent_kwargs.pop('track_mapping_path', None)
    #     parent_kwargs.pop('onset_warmup_steps', None)
    #     # 移除 MusicLDM 特有的参数
    #     parent_kwargs.pop('base_learning_rate', None)
    #     parent_kwargs.pop('batchsize', None)
    #     parent_kwargs.pop('evaluation_params', None)
    #     parent_kwargs.pop('latent_mixup', None)
    #     parent_kwargs.pop('seperate_stem_z', None)
    #     parent_kwargs.pop('use_silence_weight', None)
    #     parent_kwargs.pop('tau', None)
    #     parent_kwargs.pop('scale_factor', None)
    #     parent_kwargs.pop('concat_mode', None)
    #     parent_kwargs.pop('cond_stage_trainable', None)
    #     parent_kwargs.pop('cond_stage_forward', None)
    #     # 处理 unet_config 参数 - 不应该过滤掉，因为 DDPM 需要它
    #     # unet_config = parent_kwargs.pop('unet_config', None)
        
    #     # 调用父类初始化
    #     super().__init__(conditioning_key=conditioning_key, *args, **parent_kwargs)
        
    #     self.num_stems = num_stems
    #     self.learning_rate = base_learning_rate
    #     self.scale_by_std = scale_by_std
    #     self.evaluation_params = evaluation_params
        
    #     assert self.num_timesteps_cond <= kwargs["timesteps"]
        
    #     # for backwards compatibility after implementation of DiffusionWrapper
    #     if conditioning_key is None:
    #         conditioning_key = "concat" if concat_mode else "crossattn"
    #     if cond_stage_config == "__is_unconditional__":
    #         conditioning_key = None
    #     ckpt_path = kwargs.pop("ckpt_path", None)
    #     ignore_keys = kwargs.pop("ignore_keys", [])

    #     self.concat_mode = concat_mode
    #     self.cond_stage_trainable = cond_stage_trainable
    #     self.cond_stage_key = cond_stage_key
    #     self.cond_stage_key_orig = cond_stage_key
    #     self.latent_mixup = latent_mixup

    #     print(f'Use the Latent MixUP of {self.latent_mixup}')

    #     try:
    #         self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
    #     except:
    #         self.num_downs = 0
    #     if not scale_by_std:
    #         self.scale_factor = scale_factor
    #     else:
    #         self.register_buffer("scale_factor", torch.tensor(scale_factor))
    #     self.instantiate_first_stage(first_stage_config)
    #     self.instantiate_cond_stage(cond_stage_config)

    #     # Patch VAE functions into cond_stage_model
    #     #####################
    #     if 'target' in cond_stage_config and cond_stage_config['target'] == 'latent_diffusion.modules.encoders.modules.Patch_Cond_Model':
    #         self.cond_stage_model.encode_first_stage = self.encode_first_stage
    #         self.cond_stage_model.get_first_stage_encoding = self.get_first_stage_encoding
    #         self.cond_stage_model.num_stems = self.num_stems
    #         self.cond_stage_model.device = self.first_stage_model.get_device
    #     #####################

    #     self.cond_stage_forward = cond_stage_forward
    #     self.clip_denoised = False
    #     self.bbox_tokenizer = None

    #     self.restarted_from_ckpt = False
    #     if ckpt_path is not None:
    #         self.init_from_ckpt(ckpt_path, ignore_keys)
    #         self.restarted_from_ckpt = True

    #     self.z_channels = first_stage_config["params"]["ddconfig"]["z_channels"]

    #     self.seperate_stem_z = seperate_stem_z
    #     self.use_silence_weight = use_silence_weight
    #     self.tau = tau
        
    #     # ====== 多任務訓練參數初始化 ======
    #     self.enable_latent_diffusion = enable_latent_diffusion
    #     self.enable_onset_prediction = enable_onset_prediction
    #     self.enable_timbre_prediction = enable_timbre_prediction
    #     self.latent_loss_weight = latent_loss_weight
    #     self.onset_loss_weight = onset_loss_weight
    #     self.timbre_loss_weight = timbre_loss_weight
    #     self.onset_warmup_steps = onset_warmup_steps if onset_warmup_steps is not None else 0
    #     self.onset_data_path = onset_data_path
    #     self.timbre_data_path = timbre_data_path
    #     self.track_mapping_path = track_mapping_path
        
    #     # 初始化 onset prediction 相關變量
    #     self._onset_pred_list = []
    #     self._onset_gt_list = []
        
    #     self.model.conditioning_key = "concat"
    #     self.concat_mode = True 
    #     # 保證 self.num_stems 有效：若未從參數取得，則從 UNet 實例推斷
    #     if getattr(self, 'num_stems', None) in (None, 0):
    #         dm = getattr(self.model, 'diffusion_model', None)
    #         inferred = getattr(dm, 'num_stems', None) if dm is not None else None
    #         if inferred is not None:
    #             self.num_stems = inferred
    #         else:
    #             # 最後保底：設為 5（與預設 drum stems 對齊）
    #             self.num_stems = 5
    #     # 处理 unet_config 参数 - 不需要额外处理，因为 DDPM 已经处理了
    #     # if unet_config is not None:
    #     #     # 将 unet_config 作为 cond_stage_config 使用
    #     #     self.cond_stage_config = unet_config
    #     #     # 重新初始化 cond_stage_model
    #     #     self.instantiate_cond_stage(unet_config)

    # def instantiate_first_stage(self, config):
    #     model = instantiate_from_config(config)
    #     self.first_stage_model = model.eval()
    #     self.first_stage_model.train = disabled_train
    #     for param in self.first_stage_model.parameters():
    #         param.requires_grad = False
        
    # def instantiate_cond_stage(self, config):
    #     if config == "__is_first_stage__":
    #         print("Using first stage as conditioning stage.")
    #         self.cond_stage_model = self.first_stage_model
    #     elif config == "__is_unconditional__":
    #         print(f"Training {self.__class__.__name__} as an unconditional model.")
    #         self.cond_stage_model = None
    #     else:
    #         model = instantiate_from_config(config)
    #         self.cond_stage_model = model.eval()
    #         self.cond_stage_model.train = disabled_train
    #         for param in self.cond_stage_model.parameters():
    #             param.requires_grad = False

    # def shared_step(self, batch, **kwargs):
    #     x, c = self.get_input(batch, self.first_stage_key)
    #     loss, loss_dict = self(x, c, batch=batch, **kwargs)
    #     return loss, loss_dict

    # def forward(self, x, c, *args, **kwargs):
    #     t = torch.randint(
    #         0, self.num_timesteps, (x.shape[0],), device=self.device
    #     ).long()
    #     if self.model.conditioning_key is not None:
    #         assert c is not None
    #         if self.cond_stage_trainable:
    #             c = self.get_learned_conditioning(c)

    #     loss = self.p_losses(x, c, t, *args, **kwargs)
    #     return loss

    # def apply_model(self, x_noisy, t, cond, return_ids=False):
    #     """
    #     重寫 apply_model 方法，添加維度處理
    #     """
    #     # 處理維度轉換：如果輸入是4D，轉換為5D以匹配多stem格式
    #     if len(x_noisy.shape) == 4:
    #         # [batch, z_channels, height, width] -> [batch, num_stems, z_channels, height, width]
    #         batch_size = x_noisy.shape[0]
    #         z_channels = x_noisy.shape[1]
    #         height = x_noisy.shape[2]
    #         width = x_noisy.shape[3]
    #         x_noisy = x_noisy.unsqueeze(1).expand(batch_size, self.num_stems, z_channels, height, width)
        
    #     if isinstance(cond, dict):
    #         # hybrid case, cond is expected to be a dict
    #         pass
    #     elif cond is None:
    #         # 如果 cond 為 None，直接傳遞空字典
    #         cond = {}
    #     else:
    #         if not isinstance(cond, list):
    #             cond = [cond]
    #         if self.model.conditioning_key == "concat":
    #             key = "c_concat"
    #             # 修正：如果x是5D，確保c_concat也是5D，且z_channels維度與x_noisy一致
    #             if len(x_noisy.shape) == 5:
    #                 # 將4D或3D的條件張量轉換為5D
    #                 for i in range(len(cond)):
    #                     if len(cond[i].shape) == 4:
    #                         # [batch, stems, t, f] -> [batch, stems, z_channels, t, f]
    #                         batch_size, stems, t, f = cond[i].shape
    #                         z_channels = x_noisy.shape[2]  # 使用x_noisy的z_channels維度
    #                         cond[i] = cond[i].unsqueeze(2).expand(batch_size, stems, z_channels, t, f)
    #                     elif len(cond[i].shape) == 3:
    #                         # [batch, t, f] -> [batch, stems, z_channels, t, f]
    #                         batch_size, t, f = cond[i].shape
    #                         stems = x_noisy.shape[1]  # 使用x_noisy的stems維度
    #                         z_channels = x_noisy.shape[2]  # 使用x_noisy的z_channels維度
    #                         cond[i] = cond[i].unsqueeze(1).unsqueeze(2).expand(batch_size, stems, z_channels, t, f)
    #         elif self.model.conditioning_key == "crossattn":
    #             key = "c_crossattn"
    #         else:
    #             key = "c_film"
    #         cond = {key: cond}
        
    #     x_recon = self.model(x_noisy, t, **cond)

    #     # ====== 處理多任務分支輸出 ======
    #     onset_output = None
    #     timbre_output = None
        
    #     if isinstance(x_recon, tuple):
    #         if len(x_recon) == 2:
    #             # (main_output, onset_output) 或 (main_output, timbre_output)
    #             main_output, task_output = x_recon
    #             dm = getattr(self, 'model', None)
    #             unet = getattr(dm, 'diffusion_model', None) if dm is not None else None
    #             use_onset = (hasattr(dm, 'use_onset_branch') and getattr(dm, 'use_onset_branch')) or \
    #                         (unet is not None and getattr(unet, 'use_onset_branch', False))
    #             use_timbre = (hasattr(dm, 'use_timbre_branch') and getattr(dm, 'use_timbre_branch')) or \
    #                          (unet is not None and getattr(unet, 'use_timbre_branch', False))
    #             if use_onset:
    #                 onset_output = task_output
    #             if use_timbre:
    #                 timbre_output = task_output
    #         elif len(x_recon) == 3:
    #             # (main_output, onset_output, timbre_output)
    #             main_output, onset_output, timbre_output = x_recon
            
    #         if not return_ids:
    #             outputs = [main_output]
    #             if onset_output is not None:
    #                 outputs.append(onset_output)
    #             if timbre_output is not None:
    #                 outputs.append(timbre_output)
    #             return tuple(outputs) if len(outputs) > 1 else outputs[0]
    #         else:
    #             return x_recon
    #     else:
    #         return x_recon
    # def p_losses(self, x_start, cond, t, noise=None, *args, **kwargs):
    #     noise = default(noise, lambda: torch.randn_like(x_start))
    #     x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    #     model_output = self.apply_model(x_noisy, t, cond)
        
    #     # ====== 處理多任務分支輸出 ======
    #     onset_output = None
    #     timbre_output = None
        
    #     if isinstance(model_output, tuple):
    #         if len(model_output) == 2:
    #             # (main_output, onset_output) 或 (main_output, timbre_output)
    #             main_output, task_output = model_output
    #             dm = getattr(self, 'model', None)
    #             unet = getattr(dm, 'diffusion_model', None) if dm is not None else None
    #             use_onset = (hasattr(dm, 'use_onset_branch') and getattr(dm, 'use_onset_branch', False)) or \
    #                         (unet is not None and getattr(unet, 'use_onset_branch', False))
    #             use_timbre = (hasattr(dm, 'use_timbre_branch') and getattr(dm, 'use_timbre_branch', False)) or \
    #                          (unet is not None and getattr(unet, 'use_timbre_branch', False))
    #             if use_onset:
    #                 onset_output = task_output
    #             if use_timbre:
    #                 timbre_output = task_output
    #         elif len(model_output) == 3:
    #             # (main_output, onset_output, timbre_output)
    #             main_output, onset_output, timbre_output = model_output
    #     else:
    #         main_output = model_output
        
    #     loss_dict = {}
    #     prefix = "train" if self.training else "val"

    #     if self.parameterization == "x0":
    #         target = x_start
    #     elif self.parameterization == "eps":
    #         target = noise
    #     else:
    #         raise NotImplementedError()

    #     if self.use_silence_weight:
    #         batch = kwargs.pop('batch', None)  # Extract 'batch' from 'kwargs'
            
    #         z_mask = torch.nn.functional.interpolate(batch['fbank_stems'], size=(256, 16), mode='nearest')

    #         z_mask = torch.exp( - z_mask / self.tau)

    #         loss_simple = self.get_loss(main_output, target, mean=False).mean([2])
    #         loss_simple = loss_simple * z_mask
    #         loss_simple = loss_simple.mean([1, 2, 3])
    #     else:
    #         loss_simple = self.get_loss(main_output, target, mean=False).mean([1, 2, 3, 4])

    #     # ====== 條件性 Latent Diffusion 損失計算 ======
    #     if self.enable_latent_diffusion:
    #         loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

    #         logvar_t = self.logvar[t].to(self.device)
    #         latent_loss = loss_simple / torch.exp(logvar_t) + logvar_t
    #         if self.learn_logvar:
    #             loss_dict.update({f"{prefix}/loss_gamma": latent_loss.mean()})
    #             loss_dict.update({"logvar": self.logvar.data.mean()})

    #         latent_loss = self.l_simple_weight * latent_loss.mean()

    #         loss_vlb = self.get_loss(main_output, target, mean=False).mean(dim=(1, 2, 3, 4))
    #         loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
    #         loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
    #         latent_loss += self.original_elbo_weight * loss_vlb
            
    #         # 將 latent loss 加入總損失
    #         loss = self.latent_loss_weight * latent_loss
    #     else:
    #         # 如果不啟用 latent diffusion，初始化 loss 為 0
    #         loss = torch.tensor(0.0, device=self.device, requires_grad=True)

    #     #*******************************************************************
    #     ## loss term that pushes stems far from each other in z space
    #     if self.enable_latent_diffusion and self.seperate_stem_z:
    #         x_t_1 = self.denoise_one_step(x_noisy, t, main_output)
    #         seperation_loss = self.channel_separation_loss(x_t_1)
    #         loss_dict.update({f"{prefix}/loss_separation": seperation_loss})
    #         loss += 0.00001 * seperation_loss
    #     #*******************************************************************

    #     #*******************************************************************
    #     ## 多任務損失計算
    #     # Onset prediction loss (使用 UNet 內部的 onset 分支)
    #     if self.enable_onset_prediction and onset_output is not None and 'batch' in kwargs:
    #         batch = kwargs['batch']
    #         gt_onset = None
    #         if 'onset_pianoroll' in batch:
    #             # 直接使用 dataloader 提供的 GT
    #             gt_onset = batch['onset_pianoroll']
    #             if not isinstance(gt_onset, torch.Tensor):
    #                 gt_onset = torch.tensor(gt_onset)
    #             gt_onset = gt_onset.to(self.device, dtype=torch.float32)
    #         else:
    #             # 回退：嘗試從 JSON 讀取（僅限於不影響其他類的情況，邏輯寫在此類內部）
    #             try:
    #                 import os, json
    #                 import pandas as pd
    #                 from pathlib import Path
    #                 mapping_csv = self.track_mapping_path
    #                 onset_dir = self.onset_data_path
    #                 if mapping_csv is not None and onset_dir is not None and 'fname' in batch:
    #                     fnames = batch['fname']
    #                     if isinstance(fnames, (list, tuple)):
    #                         fnames = list(fnames)
    #                     elif hasattr(fnames, 'tolist'):
    #                         fnames = [str(x) for x in fnames.tolist()]
    #                     else:
    #                         fnames = [fnames]
    #                     tids = []
    #                     for f in fnames:
    #                         tids.append(Path(str(f)).name.split('_from_')[0])
    #                     df = pd.read_csv(mapping_csv)
    #                     # dataset 的驗證資料夾可能標為 'validation' 或 'test'
    #                     split = 'train' if self.training else 'validation'
    #                     valid_splits = ['train', 'validation', 'valid', 'test']
    #                     batch_json = []
    #                     for tid in tids:
    #                         # 嘗試多種 split 名稱，避免 CSV/資料夾命名差異導致找不到
    #                         if 'split' in df.columns:
    #                             row = df[(df['track_id'] == tid) & (df['split'].isin(valid_splits))]
    #                         else:
    #                             row = df[df['track_id'] == tid]
    #                         if row.empty:
    #                             continue
    #                         take_name = row.iloc[0]['take']
    #                         json_path = os.path.join(onset_dir, f"{take_name}_onsets.json")
    #                         if not os.path.exists(json_path):
    #                             continue
    #                         with open(json_path) as fp:
    #                             batch_json.append(json.load(fp))
    #                     if len(batch_json) > 0:
    #                         gt_np = batch_convert_onset_json(batch_json)
    #                         gt_onset = torch.as_tensor(gt_np, dtype=torch.float32, device=self.device)
    #             except Exception:
    #                 gt_onset = None

    #         if gt_onset is not None:
    #             # 尺寸對齊：B,S,T
    #             if onset_output.dim() == 3 and gt_onset.dim() == 3:
    #                 if onset_output.shape[-1] != gt_onset.shape[-1]:
    #                     if onset_output.shape[-1] > gt_onset.shape[-1]:
    #                         onset_output = onset_output[:, :, :gt_onset.shape[-1]]
    #                     else:
    #                         padding = torch.zeros(
    #                             onset_output.shape[0], onset_output.shape[1],
    #                             gt_onset.shape[-1] - onset_output.shape[-1], 
    #                             device=self.device,
    #                         )
    #                         onset_output = torch.cat([onset_output, padding], dim=-1)
    #             # 使用 logits 版本的 focal loss
    #             onset_loss = binary_focal_loss_with_logits(onset_output, gt_onset, alpha=0.25, gamma=2.0, reduction='mean')
    #             loss_dict.update({f"{prefix}/loss_onset": onset_loss.item()})
    #             loss_dict['onset_loss'] = onset_loss.item()
    #             # warm-up：前 onset_warmup_steps 逐步拉升權重
    #             if self.training and self.onset_warmup_steps and self.onset_warmup_steps > 0:
    #                 current_step = getattr(self, 'global_step', 0)
    #                 warm_factor = min(1.0, float(current_step) / float(self.onset_warmup_steps))
    #                 loss += (self.onset_loss_weight * warm_factor) * onset_loss
    #             else:
    #                 loss += self.onset_loss_weight * onset_loss
        
    #     # Timbre prediction loss
    #     if self.enable_timbre_prediction and timbre_output is not None and 'batch' in kwargs:
    #         batch = kwargs['batch']
    #         if 'timbre_features' in batch:
    #             # 獲取 ground truth timbre features
    #             gt_timbre = batch['timbre_features'].to(self.device, dtype=torch.float32)
    #             # 計算 MSE loss
    #             timbre_loss = torch.nn.functional.mse_loss(timbre_output, gt_timbre, reduction='mean')
    #             loss_dict.update({f"{prefix}/loss_timbre": timbre_loss.item()})
    #             loss_dict['timbre_loss'] = timbre_loss.item()
    #             loss += self.timbre_loss_weight * timbre_loss
    #     #*******************************************************************

    #     loss_dict.update({f"{prefix}/loss": loss})
        
    #     return loss, loss_dict

    # def channel_separation_loss(self, z):
    #     bs, num_channels, _, _, _ = z.shape
    #     loss = 0.0
    #     # Iterate over pairs of channels
    #     for i in range(num_channels):
    #         for j in range(i + 1, num_channels):
    #             # Compute the squared Euclidean distance between channels i and j
    #             diff = z[:, i] - z[:, j]
    #             distance_squared = (diff ** 2).sum(dim=[1, 2, 3])  # Sum over all dimensions except the batch dimension
    #             loss += distance_squared.mean()  # Average over the batch dimension

    #     # The negative of the sum of distances (because we want to maximize the distance)
    #     return -loss

    # def training_step(self, batch, batch_idx):
    #     assert self.training, "training step must be in training stage"
    #     self.warmup_step()

    #     if (
    #         self.state is None
    #         and len(self.trainer.optimizers[0].state_dict()["state"].keys()) > 0
    #     ):
    #         self.state = (
    #             self.trainer.optimizers[0].state_dict()["state"][0]["exp_avg"].clone()
    #         )
    #     elif self.state is not None and batch_idx % 1000 == 0:
    #         assert (
    #             torch.sum(
    #                 torch.abs(
    #                     self.state
    #                     - self.trainer.optimizers[0].state_dict()["state"][0]["exp_avg"]
    #                 )
    #             )
    #             > 1e-7
    #         ), "Optimizer is not working"

    #     loss, loss_dict = self.shared_step(batch)

    #     self.log_dict(
    #         {k: float(v) for k, v in loss_dict.items()},
    #         prog_bar=True,
    #         logger=True,
    #         on_step=True,
    #         on_epoch=True,
    #     )

    #     self.log(
    #         "global_step",
    #         float(self.global_step),
    #         prog_bar=True,
    #         logger=True,
    #         on_step=True,
    #         on_epoch=False,
    #     )

    #     lr = self.trainer.optimizers[0].param_groups[0]["lr"]
    #     self.log(
    #         "lr_abs",
    #         float(lr),
    #         prog_bar=True,
    #         logger=True,
    #         on_step=True,
    #         on_epoch=False,
    #     )
        
    #     # 額外補上 onset log
    #     if 'onset_loss' in loss_dict:
    #         self.log('train/onset_loss', loss_dict['onset_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     if 'train/loss_onset' in loss_dict:
    #         self.log('train/loss_onset_step', loss_dict['train/loss_onset'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     if 'train/loss' in loss_dict:
    #         self.log('train/loss_step', loss_dict['train/loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     if 'train/loss_simple' in loss_dict:
    #         self.log('train/loss_simple_step', loss_dict['train/loss_simple'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     if 'train/loss_vlb' in loss_dict:
    #         self.log('train/loss_vlb_step', loss_dict['train/loss_vlb'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     if 'timbre_loss' in loss_dict:
    #         self.log('train/timbre_loss', loss_dict['timbre_loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     if 'train/loss_timbre' in loss_dict:
    #         self.log('train/loss_timbre_step', loss_dict['train/loss_timbre'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     assert not self.training, "Validation/Test must not be in training stage"

    #     # 收集 onset prediction 所需的資料
    #     onset_data = None
    #     if self.enable_onset_prediction:
    #         # 使用真實的 JSON 文件生成 GT onset
    #         batch_json = load_real_onset_json(batch=batch, split='validation')
    #         if batch_json is not None:
    #             gt_onset = torch.as_tensor(batch_convert_onset_json(batch_json), dtype=torch.float32, device=self.device)
    #         else:
    #             # 如果無法載入 JSON，使用 batch 中的數據作為備用
    #             if 'onset_pianoroll' in batch:
    #                 gt_onset = batch['onset_pianoroll']
    #                 if not isinstance(gt_onset, torch.Tensor):
    #                     gt_onset = torch.tensor(gt_onset, device=self.device, dtype=torch.float32)
    #                 else:
    #                     gt_onset = gt_onset.clone().detach().to(device=self.device, dtype=torch.float32)
    #             else:
    #                 gt_onset = None

    #     # 只在啟用 latent diffusion 且為 rank 0 時進行取樣／可視化輸出
    #     samples = None
    #     name = self.get_validation_folder_name()
    #     if self.enable_latent_diffusion and self.global_rank == 0:
    #         stems_to_inpaint = self.model._trainer.datamodule.config.get('path', {}).get('stems_to_inpaint', None)
    #         stems = self.model._trainer.datamodule.config.get('path', {}).get('stems', [])

    #         if stems_to_inpaint is not None:
    #             stemidx_to_inpaint = [i for i, s in enumerate(stems) if s in stems_to_inpaint]
    #             self.inpainting(
    #                 [batch],
    #                 ddim_steps=self.evaluation_params["ddim_sampling_steps"],
    #                 ddim_eta=1.0,
    #                 x_T=None,
    #                 n_gen=self.evaluation_params["n_candidates_per_samples"],
    #                 unconditional_guidance_scale=self.evaluation_params["unconditional_guidance_scale"],
    #                 unconditional_conditioning=None,
    #                 name=name,
    #                 use_plms=False,
    #                 stemidx_to_inpaint=stemidx_to_inpaint,
    #             )
    #         else:
    #             # 若未設定 inpainting，就走與原本 MSG-LD 一樣的 generate_sample 流程
    #             samples = self.generate_sample(
    #                 [batch],
    #                 name=name,
    #                 unconditional_guidance_scale=self.evaluation_params["unconditional_guidance_scale"],
    #                 ddim_steps=self.evaluation_params["ddim_sampling_steps"],
    #                 n_gen=self.evaluation_params["n_candidates_per_samples"],
    #                 return_samples=True,
    #             )
    #     elif self.global_rank == 0:
    #         # 未啟用 latent diffusion 但仍希望得到可視化與 onset 評估時，也執行取樣
    #         samples = self.generate_sample(
    #             [batch],
    #             name=name,
    #             unconditional_guidance_scale=self.evaluation_params["unconditional_guidance_scale"],
    #             ddim_steps=self.evaluation_params["ddim_sampling_steps"],
    #             n_gen=self.evaluation_params["n_candidates_per_samples"],
    #             return_samples=True,
    #         )

    #     # 基於 samples 計算 onset_data，供 callback 與可視化
    #     if samples is not None and self.enable_onset_prediction and gt_onset is not None:
    #         try:
    #             dm = getattr(self, 'model', None)
    #             diffusion_model = getattr(dm, 'diffusion_model', None) if dm is not None else None
    #             pred_onset = None
    #             if diffusion_model is not None and getattr(diffusion_model, 'use_onset_branch', False):
    #                 # 以 zero-noise forward 取得分支輸出
    #                 B, S, C, H, W = samples.shape
    #                 t = torch.zeros(B, device=self.device, dtype=torch.long)
    #                 out_tuple = diffusion_model(samples, timesteps=t, context=None, y=None)
    #                 if isinstance(out_tuple, tuple) and len(out_tuple) >= 2:
    #                     pred_onset_logits = out_tuple[1]
    #                     pred_onset = torch.sigmoid(pred_onset_logits)

    #             if pred_onset is not None:
    #                 while pred_onset.dim() > 3:
    #                     pred_onset = pred_onset.mean(dim=2)
    #                 if pred_onset.shape[-1] != gt_onset.shape[-1]:
    #                     if pred_onset.shape[-1] > gt_onset.shape[-1]:
    #                         pred_onset = pred_onset[:, :, :gt_onset.shape[-1]]
    #                     else:
    #                         pad = torch.zeros(
    #                             pred_onset.shape[0], pred_onset.shape[1],
    #                             gt_onset.shape[-1] - pred_onset.shape[-1],
    #                             device=self.device,
    #                         )
    #                         pred_onset = torch.cat([pred_onset, pad], dim=-1)

    #                 onset_data = {
    #                     'pred_onset': pred_onset,
    #                     'gt_onset': gt_onset,
    #                     'batch_idx': batch_idx,
    #                 }
    #         except Exception:
    #             onset_data = None

    #     # 計算 loss
    #     _, loss_dict_no_ema = self.shared_step(batch)
    #     with self.ema_scope():
    #         _, loss_dict_ema = self.shared_step(batch)
    #         loss_dict_ema = {
    #             key + "_ema": loss_dict_ema[key] for key in loss_dict_ema
    #         }

    #     # 將 validation 的重點指標寫入 logger（供 checkpoint 監控與可視化）
    #     try:
    #         # val/loss
    #         if 'val/loss' in loss_dict_no_ema:
    #             v = loss_dict_no_ema['val/loss']
    #             v = float(v.detach().item()) if hasattr(v, 'detach') else float(v)
    #             self.log('val/loss', v, prog_bar=True, logger=True, on_step=False, on_epoch=True)
    #         # val/loss_simple, val/loss_vlb
    #         for k in ['val/loss_simple', 'val/loss_vlb']:
    #             if k in loss_dict_no_ema:
    #                 v = loss_dict_no_ema[k]
    #                 v = float(v.detach().item()) if hasattr(v, 'detach') else float(v)
    #                 self.log(k, v, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    #         # val/onset_loss（來自 p_losses 以 'onset_loss' 暫存）
    #         if 'onset_loss' in loss_dict_no_ema:
    #             v = loss_dict_no_ema['onset_loss']
    #             v = float(v)
    #             self.log('val/onset_loss', v, prog_bar=True, logger=True, on_step=False, on_epoch=True)
    #     except Exception:
    #         pass

    #     # 返回 onset 資料供 epoch_end 使用
    #     return {
    #         'loss_dict_no_ema': loss_dict_no_ema,
    #         'loss_dict_ema': loss_dict_ema,
    #         'onset_data': onset_data
    #     }

    # def get_validation_folder_name(self):
    #     return "val_%s" % (self.global_step)

    # @torch.no_grad()
    # def on_validation_epoch_end(self) -> None:
    #     # 這裡可以添加 onset F1 評估等邏輯
    #     pass

    # def configure_optimizers(self):
    #     # 只訓練需要的參數
    #     params_to_update = []
    #     for name, param in self.named_parameters():
    #         if param.requires_grad:
    #             params_to_update.append(param)
        
    #     optimizer = torch.optim.AdamW(params_to_update, lr=self.learning_rate)
    #     return optimizer

    # def on_train_batch_end(self, *args, **kwargs):
    #     # 更新 EMA
    #     if self.use_ema:
    #         self.model_ema(self.model)

    # def on_validation_epoch_start(self):
    #     # 使用 EMA 模型進行驗證
    #     if self.use_ema:
    #         self.model_ema.store(self.model.parameters())
    #         self.model_ema.copy_to(self.model)

    # def on_validation_epoch_end(self):
    #     # 恢復原始模型
    #     if self.use_ema:
    #         self.model_ema.restore(self.model.parameters())
    
    # # 繼承 MusicLDM 的其他方法
    # @torch.no_grad()
    # def get_input(
    #     self,
    #     batch,
    #     k,
    #     return_first_stage_encode=True,
    #     return_first_stage_outputs=False,
    #     force_c_encode=False,
    #     cond_key=None,
    #     return_original_cond=False,
    #     bs=None,
    #     **kwargs,
    # ):
    #     """
    #     與原始 MusicLDM 對齊：
    #     - 若 batch 含 `fbank_stems`（形狀 [B, S, T, F]），走 VAE encoder -> latent z，並 reshape 成 [B, S, C, 256, 16]；
    #     - 否則回退父類邏輯，但最終仍返回 (x, None)。
    #     """
    #     if isinstance(batch, dict) and "fbank_stems" in batch:
    #         fbank_stems = batch["fbank_stems"].to(device=self.device, dtype=torch.float32)
    #         b, s, _, _ = fbank_stems.shape
    #         # [B, S, T, F] -> [B*S, 1, T, F]
    #         x_in = self.adapt_fbank_for_VAE_encoder(fbank_stems)
    #         with torch.no_grad():
    #             enc = self.encode_first_stage(x_in)
    #             z_flat = self.get_first_stage_encoding(enc).detach()
    #         # [B*S, C, 256, 16] -> [B, S, C, 256, 16]
    #         bs, c, h, w = z_flat.shape
    #         assert bs == b * s, f"latent batch {bs} != B*S {b*s}"
    #         z = z_flat.view(b, s, c, h, w)

    #         # 產生條件張量，形狀 [B, S, 8, 256, 16]
    #         # Patch_Cond_Model 期望輸入為 [B, 1, T, F]，然後在內部展開到 num_stems
    #         if getattr(self, "cond_stage_model", None) is not None:
    #             # 動態對齊 cond 模型的 num_stems，並確保已注入 VAE 編碼接口
    #             try:
    #                 self.cond_stage_model.num_stems = s
    #                 if getattr(self.cond_stage_model, 'encode_first_stage', None) is None:
    #                     self.cond_stage_model.encode_first_stage = self.encode_first_stage
    #                 if getattr(self.cond_stage_model, 'get_first_stage_encoding', None) is None:
    #                     self.cond_stage_model.get_first_stage_encoding = self.get_first_stage_encoding
    #             except Exception:
    #                 pass
    #             fbank_mix = fbank_stems.mean(dim=1, keepdim=True)  # [B, 1, T, F]
    #             c_embed = self.cond_stage_model(fbank_mix)
    #         else:
    #             c_embed = None

    #         return z, c_embed

    #     # 其他情況：回退父類；若父類只返回張量，包成 (x, None)
    #     key = k or getattr(self, "first_stage_key", "fbank")
    #     # 回退父類簡化版接口
    #     out = DDPM.get_input(self, batch, key)
    #     if isinstance(out, tuple) and len(out) == 2:
    #         return out
    #     return out, None

    # # ========= helpers: local實作，避免使用 super() 指向 DDPM 而不存在 =========
    # def adapt_fbank_for_VAE_encoder(self, tensor):
    #     # tensor: [B, num_stems, T, F] -> [B*num_stems, 1, T, F]
    #     b, s, t, f = tensor.shape
    #     return tensor.view(b * s, 1, t, f)

    # def adapt_latent_for_LDM(self, tensor):
    #     # tensor: [B*num_stems, C, H, W] -> [B, num_stems, C, H, W]
    #     bs, c, h, w = tensor.shape
    #     assert self.num_stems is not None and self.num_stems > 0, "num_stems 未設定"
    #     assert bs % self.num_stems == 0, f"批量大小不整除：{bs} vs num_stems={self.num_stems}"
    #     b = bs // self.num_stems
    #     return tensor.view(b, self.num_stems, c, h, w)

    # def adapt_latent_for_VAE_decoder(self, tensor):
    #     # tensor: [B, S, C, H, W] -> [B*S, C, H, W]
    #     b, s, c, h, w = tensor.shape
    #     return tensor.view(b * s, c, h, w)

    # def encode_first_stage(self, x):
    #     if hasattr(self.first_stage_model, 'encode'):
    #         return self.first_stage_model.encode(x)
    #     return self.first_stage_model(x)

    # def get_first_stage_encoding(self, encoder_posterior):
    #     if hasattr(encoder_posterior, 'sample'):
    #         return encoder_posterior.sample()
    #     return encoder_posterior

    # def get_learned_conditioning(self, c):
    #     # 對齊 MusicLDM 的實作
    #     return MusicLDM.get_learned_conditioning(self, c)

    # def meshgrid(self, h, w):
    #     return MusicLDM.meshgrid(self, h, w)

    # def delta_border(self, h, w):
    #     return MusicLDM.delta_border(self, h, w)

    # def get_weighting(self, h, w, Ly, Lx, device):
    #     return MusicLDM.get_weighting(self, h, w, Ly, Lx, device)

    # def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):
    #     return MusicLDM.get_fold_unfold(self, x, kernel_size, stride, uf, df)

    # # 以下方法轉發到 MusicLDM 的對應實作，避免指到 DDPM 而缺少條件版邏輯
    # def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
    #     return MusicLDM.decode_first_stage(self, z, predict_cids, force_not_quantize)

    # def mel_spectrogram_to_waveform(self, mel, savepath=".", bs=None, name="outwav", save=False):
    #     return MusicLDM.mel_spectrogram_to_waveform(self, mel, savepath=savepath, bs=bs, name=name, save=save)

    # def save_waveform(self, waveform, savepath, name="outwav"):
    #     return MusicLDM.save_waveform(self, waveform, savepath=savepath, name=name)

    # def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False, return_x0=False, score_corrector=None, corrector_kwargs=None):
    #     return MusicLDM.p_mean_variance(self, x, c, t, clip_denoised, return_codebook_ids, quantize_denoised, return_x0, score_corrector, corrector_kwargs)

    # def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False, return_codebook_ids=False, quantize_denoised=False, return_x0=False, temperature=1.0, noise_dropout=0.0, score_corrector=None, corrector_kwargs=None):
    #     return MusicLDM.p_sample(self, x, c, t, clip_denoised, repeat_noise, return_codebook_ids, quantize_denoised, return_x0, temperature, noise_dropout, score_corrector, corrector_kwargs)

    # def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False, img_callback=None, mask=None, x0=None, temperature=1.0, noise_dropout=0.0, score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None, log_every_t=None):
    #     return MusicLDM.progressive_denoising(self, cond, shape, verbose, callback, quantize_denoised, img_callback, mask, x0, temperature, noise_dropout, score_corrector, corrector_kwargs, batch_size, x_T, start_T, log_every_t)

    # def p_sample_loop(self, cond, shape, return_intermediates=False, x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False, mask=None, x0=None, img_callback=None, start_T=None, log_every_t=None):
    #     return MusicLDM.p_sample_loop(self, cond, shape, return_intermediates, x_T, verbose, callback, timesteps, quantize_denoised, mask, x0, img_callback, start_T, log_every_t)

    # def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None, verbose=True, timesteps=None, quantize_denoised=False, mask=None, x0=None, shape=None, **kwargs):
    #     return MusicLDM.sample(self, cond, batch_size, return_intermediates, x_T, verbose, timesteps, quantize_denoised, mask, x0, shape, **kwargs)

    # def sample_log(self, cond, batch_size, ddim, ddim_steps, unconditional_guidance_scale=1.0, unconditional_conditioning=None, use_plms=False, mask=None, **kwargs):
    #     return MusicLDM.sample_log(self, cond, batch_size, ddim, ddim_steps, unconditional_guidance_scale, unconditional_conditioning, use_plms, mask, **kwargs)

    # def generate_long_sample(self, batchs, ddim_steps=200, ddim_eta=1.0, x_T=None, n_gen=1, generate_duration=60, unconditional_guidance_scale=1.0, unconditional_conditioning=None, name="waveform", use_plms=False, **kwargs):
    #     return MusicLDM.generate_long_sample(self, batchs, ddim_steps, ddim_eta, x_T, n_gen, generate_duration, unconditional_guidance_scale, unconditional_conditioning, name, use_plms, **kwargs)
    # def generate_sample(self, batchs, ddim_steps=200, ddim_eta=1.0, x_T=None, n_gen=1, unconditional_guidance_scale=1.0, unconditional_conditioning=None, name="waveform", use_plms=False, return_samples=False, **kwargs):
    #     # 直接複製 MusicLDM.generate_sample 的流程，但移除 super() 用法，改用 DDPM.get_input 取 batch 欄位
    #     import os
    #     import numpy as np
    #     from pathlib import Path
    #     assert x_T is None
    #     try:
    #         batchs = iter(batchs)
    #     except TypeError:
    #         raise ValueError("The first input argument should be an iterable object")

    #     if use_plms:
    #         assert ddim_steps is not None

    #     use_ddim = ddim_steps is not None
    #     waveform_save_path = os.path.join(self.get_log_dir(), name)
    #     os.makedirs(waveform_save_path, exist_ok=True)
    #     wavefor_target_save_path = os.path.join(self.get_log_dir(), "target_%s" % (self.global_step))
    #     os.makedirs(wavefor_target_save_path, exist_ok=True)

    #     all_samples = []

    #     with self.ema_scope("Plotting"):
    #         for batch in batchs:
    #             z, c = self.get_input(
    #                 batch,
    #                 self.first_stage_key,
    #                 return_first_stage_outputs=False,
    #                 force_c_encode=True,
    #                 return_original_cond=False,
    #                 bs=None,
    #             )

    #             if self.cond_stage_model is not None:
    #                 batch_size = z.shape[0] * n_gen
    #                 if self.cond_stage_model.embed_mode == "text":
    #                     text = DDPM.get_input(self, batch, "text")
    #                     if c is not None:
    #                         c = torch.cat([c] * n_gen, dim=0)
    #                     text = text * n_gen
    #                 elif self.cond_stage_model.embed_mode == "audio":
    #                     text = DDPM.get_input(self, batch, "waveform")
    #                     if c is not None:
    #                         c = torch.cat([c] * n_gen, dim=0)
    #                     text = torch.cat([text] * n_gen, dim=0)

    #                 if unconditional_guidance_scale != 1.0:
    #                     unconditional_conditioning = (
    #                         self.cond_stage_model.get_unconditional_condition(batch_size)
    #                     )
    #             else:
    #                 batch_size = z.shape[0]
    #                 text = None

    #             # 檔名安全取得
    #             if isinstance(batch, dict) and "fname" in batch:
    #                 raw_fnames = batch["fname"]
    #                 if isinstance(raw_fnames, (list, tuple)):
    #                     fnames = list(raw_fnames)
    #                 elif hasattr(raw_fnames, 'tolist'):
    #                     fnames = [str(x) for x in raw_fnames.tolist()]
    #                 else:
    #                     fnames = [str(raw_fnames)]
    #             else:
    #                 fnames = [f"sample_{self.global_step}"] * z.shape[0]

    #             samples, _ = self.sample_log(
    #                 cond=c,
    #                 batch_size=batch_size,
    #                 x_T=x_T,
    #                 ddim=use_ddim,
    #                 ddim_steps=ddim_steps,
    #                 eta=ddim_eta,
    #                 unconditional_guidance_scale=unconditional_guidance_scale,
    #                 unconditional_conditioning=unconditional_conditioning,
    #                 use_plms=use_plms,
    #             )

    #             all_samples.append(samples.clone())

    #             samples = self.adapt_latent_for_VAE_decoder(samples)
    #             mel = self.decode_first_stage(samples)

    #             waveform = self.mel_spectrogram_to_waveform(
    #                 mel, savepath=waveform_save_path, bs=None, name=fnames, save=False
    #             )

    #             if waveform.dtype == 'float16':
    #                 waveform = waveform.astype('float32')
    #             waveform = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
    #             waveform = np.clip(waveform, -1, 1)

    #             waveform_reshaped = waveform.reshape(batch_size, self.num_stems, waveform.shape[-1])
    #             mix = waveform_reshaped.sum(axis=1)
    #             mix = np.nan_to_num(mix, nan=0.0, posinf=1.0, neginf=-1.0)
    #             mix = np.clip(mix, -1, 1)

    #             if self.model.conditioning_key is not None:
    #                 if self.cond_stage_model.embed_mode == "text":
    #                     similarity = self.cond_stage_model.cos_similarity(
    #                         torch.FloatTensor(mix).squeeze(1), text
    #                     )
    #                     best_index = []
    #                     for i in range(z.shape[0]):
    #                         candidates = similarity[i :: z.shape[0]]
    #                         max_index = torch.argmax(candidates).item()
    #                         best_index.append(i + max_index * z.shape[0])
    #                 else:
    #                     best_index = torch.arange(z.shape[0])
    #             else:
    #                 best_index = torch.arange(z.shape[0])

    #             mix = mix[best_index]

    #             selected_wavs = []
    #             selected_mels = []
    #             for start_index in best_index:
    #                 actual_start_index = start_index * self.num_stems
    #                 selected_slice = waveform[actual_start_index:actual_start_index + self.num_stems]
    #                 selected_wavs.append(selected_slice)
    #                 selected_slice = mel[actual_start_index:actual_start_index + self.num_stems].cpu().detach().numpy()
    #                 selected_mels.append(selected_slice)

    #             waveform = np.concatenate(selected_wavs, axis=0)[:,0,:]
    #             waveform = waveform.reshape(z.shape[0], self.num_stems, waveform.shape[-1])
    #             mel = np.concatenate(selected_mels, axis=0)[:,0,:]
    #             mel = mel.reshape(z.shape[0], self.num_stems, mel.shape[-2], mel.shape[-1])

    #             generated_mix_dir = os.path.join(waveform_save_path, "mix")
    #             os.makedirs(generated_mix_dir, exist_ok=True)
    #             if mix.ndim == 1:
    #                 mix = mix[np.newaxis, :]
    #             self.save_waveform(mix[:, np.newaxis, :], generated_mix_dir, name=fnames)

    #             # 目標音訊/頻譜：只有在 batch 提供 GT 時才寫入 target_*
    #             target_mix = None
    #             target_waveforms = None
    #             if isinstance(batch, dict) and 'waveform' in batch:
    #                 target_mix = batch['waveform']
    #                 target_mix_dir = os.path.join(wavefor_target_save_path, "mix")
    #                 os.makedirs(target_mix_dir, exist_ok=True)
    #                 self.save_waveform(target_mix.unsqueeze(1).cpu().detach(), target_mix_dir, name=fnames)
    #             if isinstance(batch, dict) and 'waveform_stems' in batch:
    #                 target_waveforms = batch['waveform_stems']

    #             for i in range(self.num_stems):
    #                 generated_stem_dir = os.path.join(waveform_save_path, "stem_"+str(i))
    #                 os.makedirs(generated_stem_dir, exist_ok=True)
    #                 self.save_waveform(waveform[:,i,:][:, np.newaxis, :], generated_stem_dir, name=fnames)
    #                 generated_stem_mel_dir = os.path.join(waveform_save_path, "stem_mel_"+str(i))
    #                 os.makedirs(generated_stem_mel_dir, exist_ok=True)
    #                 for j in range(mel.shape[0]):
    #                     file_path =  os.path.join(generated_stem_mel_dir,fnames[j]+".npy")
    #                     np.save(file_path, mel[j,i,:])

    #                 if target_waveforms is not None:
    #                     target_stem_dir = os.path.join(wavefor_target_save_path, "stem_"+str(i))
    #                     os.makedirs(target_stem_dir, exist_ok=True)
    #                     self.save_waveform(target_waveforms[:,i,:].unsqueeze(1).cpu().detach(), target_stem_dir, name=fnames)
    #                 # target mel 只有在 batch 提供 fbank_stems 時才寫入
    #                 if isinstance(batch, dict) and 'fbank_stems' in batch:
    #                     target_stem_mel_dir = os.path.join(wavefor_target_save_path, "stem_mel_"+str(i))
    #                     os.makedirs(target_stem_mel_dir, exist_ok=True)
    #                     for j in range(mel.shape[0]):
    #                         file_path =  os.path.join(target_stem_mel_dir,fnames[j]+".npy")
    #                         np.save(file_path, batch['fbank_stems'].cpu().numpy()[j,i,:])

    #             if self.logger is not None:
    #                 # None 防護：若缺 target，使用生成結果作為參考，避免 NoneType
    #                 safe_target_mix = target_mix
    #                 if safe_target_mix is None:
    #                     # mix: np.ndarray [B, T]
    #                     safe_target_mix = torch.from_numpy(mix).to(self.device)
    #                 safe_target_waveforms = target_waveforms
    #                 if safe_target_waveforms is None:
    #                     # waveform: np.ndarray [B, S, T]
    #                     safe_target_waveforms = torch.from_numpy(waveform).to(self.device)
    #                 log_data_batch = mel, waveform, safe_target_waveforms, mix, safe_target_mix, fnames, batch
    #                 self.log_images_audios(log_data_batch)

    #     if return_samples and all_samples:
    #         return torch.cat(all_samples, dim=0)
    #     return waveform_save_path

    # def audio_continuation(self, batchs, ddim_steps=200, ddim_eta=1.0, x_T=None, n_gen=1, unconditional_guidance_scale=1.0, unconditional_conditioning=None, name="waveform", use_plms=False, **kwargs):
    #     return MusicLDM.audio_continuation(self, batchs, ddim_steps, ddim_eta, x_T, n_gen, unconditional_guidance_scale, unconditional_conditioning, name, use_plms, **kwargs)

    # def inpainting(self, batchs, ddim_steps=200, ddim_eta=1.0, x_T=None, n_gen=1, unconditional_guidance_scale=1.0, unconditional_conditioning=None, name="waveform", use_plms=False, **kwargs):
    #     return MusicLDM.inpainting(self, batchs, ddim_steps, ddim_eta, x_T, n_gen, unconditional_guidance_scale, unconditional_conditioning, name, use_plms, **kwargs)

    # def super_resolution(self, batchs, ddim_steps=200, ddim_eta=1.0, x_T=None, n_gen=1, unconditional_guidance_scale=1.0, unconditional_conditioning=None, name="waveform", use_plms=False, **kwargs):
    #     return MusicLDM.super_resolution(self, batchs, ddim_steps, ddim_eta, x_T, n_gen, unconditional_guidance_scale, unconditional_conditioning, name, use_plms, **kwargs)

    # # ====== utils from MusicLDM ======
    # def tensor2numpy(self, tensor):
    #     return MusicLDM.tensor2numpy(self, tensor)

    # def log_images_audios(self, log_data_batch):
    #     return MusicLDM.log_images_audios(self, log_data_batch)

# class MusicLDM_inference(MusicLDM):
#     """
#     與 MusicLDM 幾乎一樣，但不計算 loss，適合無 target 的 dataset。
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # 設置推理模式
#         self.eval()
#         # 可選：直接使用CPU進行mel-spectrogram計算，避免cuFFT錯誤
#         self.use_cpu_mel_computation = kwargs.get('use_cpu_mel_computation', False)

#     def training_step(self, batch, batch_idx):
#         # 不計算 loss，直接 return None
#         return None
    
#     def validation_step(self, batch, batch_idx):
#         """
#         執行推論而不計算 loss，保持 MusicLDM 的所有邏輯
#         """
#         import os
#         import torch
#         import torchaudio
#         import numpy as np
#         from tqdm import tqdm
        
#         # 取得推論配置
#         infer_cfg = getattr(self, 'inference_config', {})
#         output_dir = infer_cfg.get('output_dir', './inference_results')
#         segment_length = infer_cfg.get('segment_length', 10.0)
#         sample_rate = infer_cfg.get('sample_rate', 16000)
#         ddim_steps = infer_cfg.get('ddim_steps', 200)
#         ddim_eta = infer_cfg.get('ddim_eta', 1.0)
#         unconditional_guidance_scale = infer_cfg.get('unconditional_guidance_scale', 1.0)
#         n_gen = infer_cfg.get('n_gen', 1)
#         bs = infer_cfg.get('batch_size', None)
        
#         # 創建輸出目錄
#         os.makedirs(output_dir, exist_ok=True)
        
#         # 取得檔案名稱
#         if isinstance(batch, dict) and 'fname' in batch:
#             fname = batch['fname']
#             if isinstance(fname, list):
#                 fname = fname[0]
#         else:
#             fname = f'batch_{batch_idx}'
            
#         print(f"Processing batch {batch_idx}: {fname}")
        
#         # 檢查是否有整首歌的 waveform
#         if 'waveform' in batch:
#             full_waveform = batch['waveform']
#             if len(full_waveform.shape) == 1:
#                 full_waveform = full_waveform.unsqueeze(0)
            
#             total_samples = full_waveform.shape[1]
#             segment_samples = int(segment_length * sample_rate)
            
#             # 只處理完整的段
#             num_complete_segments = total_samples // segment_samples
#             actual_processed_samples = num_complete_segments * segment_samples
            
#             print(f"  Full song length: {total_samples/16000:.2f}s")
#             print(f"  Will process {num_complete_segments} complete segments ({actual_processed_samples/16000:.2f}s)")
#             print(f"  Discarding last {total_samples - actual_processed_samples} samples ({(total_samples - actual_processed_samples)/16000:.2f}s)")
            
#             # 處理整首歌，只處理完整的段
#             all_stems = []
            
#             for seg_idx in range(num_complete_segments):
#                 start_sample = seg_idx * segment_samples
#                 end_sample = start_sample + segment_samples
#                 segment = full_waveform[:, start_sample:end_sample]
                
#                 print(f"    Processing segment {seg_idx+1}/{num_complete_segments}: {start_sample/16000:.2f}s - {end_sample/16000:.2f}s")
                
#                 # 檢查音頻數據的有效性
#                 if torch.isnan(segment).any() or torch.isinf(segment).any():
#                     print(f"    Warning: Segment {seg_idx} contains NaN or Inf values, skipping...")
#                     continue
                
#                 if segment.shape[1] == 0:
#                     print(f"    Warning: Segment {seg_idx} is empty, skipping...")
#                     continue
                
#                 # 確保音頻數據是 float32 類型
#                 if segment.dtype != torch.float32:
#                     segment = segment.float()
                
#                 # 將 segment 轉換為 mel-spectrogram
#                 import torchaudio.transforms as T
                
#                 # 檢查是否應該直接使用CPU計算（避免cuFFT錯誤）
#                 use_cpu_for_mel = False
#                 if hasattr(self, 'use_cpu_mel_computation'):
#                     use_cpu_for_mel = self.use_cpu_mel_computation
                
#                 if use_cpu_for_mel:
#                     # 直接使用CPU計算，避免cuFFT錯誤
#                     segment_cpu = segment.cpu()
#                     mel_transform = T.MelSpectrogram(
#                         sample_rate=sample_rate,
#                         n_fft=1024,
#                         hop_length=160,
#                         win_length=1024,
#                         n_mels=64,
#                         f_min=0,
#                         f_max=8000
#                     )
#                     mel = mel_transform(segment_cpu)
#                     mel = torch.log(mel + 1e-9)
#                     mel = mel.unsqueeze(1).transpose(-1, -2)
#                     mel = mel.to(self.device)
#                 else:
#                     # 嘗試GPU計算，失敗時回退到CPU
#                     mel_transform = T.MelSpectrogram(
#                         sample_rate=sample_rate,
#                         n_fft=1024,
#                         hop_length=160,
#                         win_length=1024,
#                         n_mels=64,
#                         f_min=0,
#                         f_max=8000
#                     ).to(segment.device)
                    
#                     try:
#                         mel = mel_transform(segment)
#                         mel = torch.log(mel + 1e-9)
#                         mel = mel.unsqueeze(1).transpose(-1, -2)
#                     except RuntimeError as e:
#                         if "cuFFT" in str(e):
#                             print(f"    Warning: cuFFT error in segment {seg_idx}, trying CPU fallback...")
#                             # 嘗試在 CPU 上計算
#                             try:
#                                 segment_cpu = segment.cpu()
#                                 mel_transform_cpu = T.MelSpectrogram(
#                                     sample_rate=sample_rate,
#                                     n_fft=1024,
#                                     hop_length=160,
#                                     win_length=1024,
#                                     n_mels=64,
#                                     f_min=0,
#                                     f_max=8000
#                                 )
#                                 mel = mel_transform_cpu(segment_cpu)
#                                 mel = torch.log(mel + 1e-9)
#                                 mel = mel.unsqueeze(1).transpose(-1, -2)
#                                 mel = mel.to(self.device)
#                             except Exception as cpu_e:
#                                 print(f"    Error: CPU fallback also failed for segment {seg_idx}: {cpu_e}")
#                                 continue
#                         else:
#                             print(f"    Error: Mel-spectrogram computation failed for segment {seg_idx}: {e}")
#                             continue
                
#                 # 確保 mel 的時間維度是 1024
#                 if mel.shape[-2] != 1024:
#                     if mel.shape[-2] > 1024:
#                         mel = mel[:, :, :1024, :]
#                     else:
#                         padding = 1024 - mel.shape[-2]
#                         mel = torch.nn.functional.pad(mel, (0, 0, 0, padding))
                
#                 mel = mel.to(self.device)
#                 print(f"    Mel shape: {mel.shape}")
                
#                 # 執行推論 - 使用原有的 get_input 邏輯
#                 try:
#                     # 創建一個包含必要數據的 batch，沿用原有的 key
#                     inference_batch = {
#                         'fbank_stems': mel.unsqueeze(2).expand(1, 1, self.num_stems, 1024, 64),  # [1, 1, 5, 1024, 64]
#                         'fname': [fname]
#                     }
                    
#                     # 如果原來的 cond_stage_key 不是 fbank_stems，需要添加對應的條件數據
#                     if self.cond_stage_key != self.first_stage_key:
#                         # 根據 cond_stage_key 添加對應的條件數據
#                         if self.cond_stage_key == 'fbank':
#                             inference_batch['fbank'] = mel  # 使用單一 mel 作為條件
#                         elif self.cond_stage_key == 'waveform':
#                             inference_batch['waveform'] = segment  # 使用原始 waveform 作為條件
#                         else:
#                             # 如果不知道具體的條件類型，使用 mel 作為 fallback
#                             inference_batch[self.cond_stage_key] = mel
                    
#                     # 使用原有的 get_input 方法，這樣會自動處理所有維度轉換和條件處理
#                     z, c = self.get_input(inference_batch, self.first_stage_key)
                    
#                     # 處理無條件引導 - 與 MusicLDM 保持一致
#                     unconditional_conditioning = None
#                     if unconditional_guidance_scale != 1.0:
#                         # 生成與c相同形狀的無條件條件張量
#                         unconditional_conditioning = self.cond_stage_model.get_unconditional_condition(1)
#                         print(f"    Generated unconditional_conditioning shape: {unconditional_conditioning.shape}")
                    
#                     # 使用 sample_log 方法進行推論
#                     samples, _ = self.sample_log(
#                         cond=c,
#                         batch_size=1,
#                         ddim=True,
#                         ddim_steps=ddim_steps,
#                         eta=ddim_eta,
#                         unconditional_guidance_scale=unconditional_guidance_scale,
#                         unconditional_conditioning=unconditional_conditioning,
#                         use_plms=False
#                     )
                    
#                     # 處理生成的樣本
#                     if len(samples.shape) == 4 and samples.shape[1] == self.num_stems:
#                         # 正確的形狀：[batch, stems, latent_t, latent_f]
#                         stem_latents = samples
                        
#                         # 將生成的 latent 轉換為 waveform
#                         waveforms = self.mel_spectrogram_to_waveform_multi_stem(
#                             stem_latents, 
#                             savepath=None,
#                             name=f"{fname}_segment_{seg_idx}", 
#                             save=False
#                         )
                        
#                         if waveforms is not None:
#                             waveforms = waveforms.squeeze(0)
#                             all_stems.append(waveforms.cpu().numpy())
#                         else:
#                             print(f"    Warning: No waveforms generated for segment {seg_idx}")
#                     elif len(samples.shape) == 5 and samples.shape[1] == 5:
#                         # 處理 [1, 5, 8, 256, 16] 形狀的樣本
                        
#                         samples_reshaped = self.adapt_latent_for_VAE_decoder(samples)
                        
#                         # 解碼每個 stem
#                         stem_waveforms = []
#                         for stem_idx in range(self.num_stems):
#                             stem_latent = samples_reshaped[stem_idx:stem_idx+1]  # [1, 8, 256, 16]
#                             stem_mel = self.decode_first_stage(stem_latent)
#                             stem_waveform = self.mel_spectrogram_to_waveform(
#                                 stem_mel, savepath=None, name=f"stem_{stem_idx}", save=False
#                             )
#                             if stem_waveform is not None:
#                                 stem_waveforms.append(stem_waveform.squeeze(0))
#                             else:
#                                 stem_waveforms.append(torch.zeros(segment_samples))
                        
#                         if stem_waveforms:
#                             # 確保所有元素都是 torch tensor
#                             stem_waveforms = [w if isinstance(w, torch.Tensor) else torch.from_numpy(w) for w in stem_waveforms]
#                             all_stems.append(torch.stack(stem_waveforms).cpu().numpy())
#                     else:
#                         print(f"    Warning: Unexpected samples shape: {samples.shape}")
                        
#                 except Exception as e:
#                     print(f"    Error processing segment {seg_idx}: {str(e)}")
#                     import traceback
#                     traceback.print_exc()
#                     continue
            
#             # 保存結果
#             if all_stems:
#                 all_stems = np.array(all_stems)  # [segments, stems, samples]
                
#                 # 使用 Lightning 的标准日志目录结构
#                 validation_folder = self.get_validation_folder_name()
#                 lightning_output_dir = os.path.join(self.get_log_dir(), validation_folder)
#                 os.makedirs(lightning_output_dir, exist_ok=True)
                
#                 # 保存為 numpy 文件
#                 output_path = os.path.join(lightning_output_dir, f"{fname}_stems.npy")
#                 np.save(output_path, all_stems)
#                 print(f"  Saved stems to: {output_path}")
                
#                 # 保存為音頻文件，按照標準結構：mix 和 stem_0~4
#                 # 1. 保存 mix（所有 stems 的混合）
#                 mix_waveform = all_stems.sum(axis=1)  # [segments, samples] - 混合所有 stems
#                 mix_path = os.path.join(lightning_output_dir, "mix", f"{fname}")
#                 os.makedirs(os.path.dirname(mix_path), exist_ok=True)
#                 torchaudio.save(mix_path, torch.tensor(mix_waveform.flatten()).unsqueeze(0), sample_rate)
#                 print(f"  Saved mix to: {mix_path}")
                
#                 # 2. 保存各個 stems
#                 for stem_idx in range(all_stems.shape[1]):
#                     stem_waveform = all_stems[:, stem_idx, :].flatten()  # 連接所有段
#                     stem_path = os.path.join(lightning_output_dir, f"stem_{stem_idx}", f"{fname}")
#                     os.makedirs(os.path.dirname(stem_path), exist_ok=True)
#                     torchaudio.save(stem_path, torch.tensor(stem_waveform).unsqueeze(0), sample_rate)
#                     print(f"  Saved stem {stem_idx} to: {stem_path}")
#             else:
#                 print(f"  No stems generated for {fname}")
        
#         return None  # 不計算 loss

#     def shared_step(self, batch, **kwargs):
#         # 不計算 loss，直接 return None
#         return None

#     def configure_optimizers(self):
#         # 不需要 optimizer
#         return None

#     def mel_spectrogram_to_waveform_multi_stem(self, mel, savepath=".", bs=None, name="outwav", save=True):
#         """
#         專門處理多 stem 的 mel_spectrogram 到 waveform 轉換
#         mel: [batch, stems, z_channels, t-steps, fbins] 或 [stems, z_channels, t-steps, fbins]
#         注意：這裡的 mel 實際上是 latent，需要先解碼為 mel-spectrogram
#         """
#         import torch
#         import numpy as np
        
#         if len(mel.shape) == 4:
#             # [stems, z_channels, t-steps, fbins]
#             stems = mel.shape[0]
            
#             all_waveforms = []
#             for i in range(stems):
#                 stem_latent = mel[i:i+1]  # [1, z_channels, t-steps, fbins] - 實際上是 latent
                
#                 # 先將 latent 解碼為 mel-spectrogram
#                 stem_mel = self.decode_first_stage(stem_latent)  # 解碼為 mel-spectrogram
                
#                 # 使用原有的 mel_spectrogram_to_waveform 方法
#                 stem_waveform = self.mel_spectrogram_to_waveform(
#                     mel=stem_mel,
#                     savepath=savepath,
#                     bs=bs,
#                     name=f"{name}_stem_{i}",
#                     save=save
#                 )
                
#                 if stem_waveform is not None:
#                     # 確保是 torch tensor
#                     if isinstance(stem_waveform, np.ndarray):
#                         stem_waveform = torch.from_numpy(stem_waveform)
#                     all_waveforms.append(stem_waveform)
            
#             # 合併所有 stems
#             if all_waveforms:
#                 combined_waveform = torch.stack(all_waveforms, dim=0)  # [stems, samples]
#                 return combined_waveform
#             else:
#                 return None
                
#         elif len(mel.shape) == 5:
#             # [batch, stems, z_channels, t-steps, fbins]
#             batch_size = mel.shape[0]
#             stems = mel.shape[1]
            
#             all_batch_waveforms = []
#             for b in range(batch_size):
#                 batch_mel = mel[b]  # [stems, z_channels, t-steps, fbins]
#                 batch_waveform = self.mel_spectrogram_to_waveform_multi_stem(
#                     mel=batch_mel,
#                     savepath=savepath,
#                     bs=bs,
#                     name=f"{name}_batch_{b}",
#                     save=save
#                 )
#                 if batch_waveform is not None:
#                     all_batch_waveforms.append(batch_waveform)
            
#             if all_batch_waveforms:
#                 return torch.stack(all_batch_waveforms, dim=0)  # [batch, stems, samples]
#             else:
#                 return None
#         else:
#             raise ValueError(f"Unexpected mel shape: {mel.shape}")


# class OnsetPredictor(nn.Module):
#     """
#     重新設計的 Onset 預測網絡
#     真正保持 stems 間關聯性，不將 batch 和 stems 相乘
#     輸入: latent z (batch_size, 5, 8, 256, 16)
#     輸出: onset-pianoroll (batch_size, 5, 1024)
#     """
#     def __init__(self, latent_channels=8, latent_height=256, latent_width=16, 
#                  num_stems=5, output_length=1024, hidden_dim=256, n_heads=4):
#         super().__init__()
#         self.latent_channels = latent_channels
#         self.latent_height = latent_height
#         self.latent_width = latent_width
#         self.num_stems = num_stems
#         self.output_length = output_length

#         # 1. 時頻特徵提取 (保持 batch 和 stems 分離)
#         self.time_freq_conv = nn.Sequential(
#             nn.Conv2d(latent_channels, hidden_dim, kernel_size=(3, 3), padding=(1, 1)),
#             nn.ReLU(),
#             nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=(1, 1)),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((latent_height, 1))  # 保留時間維度，壓縮頻率維度
#         )
        
#         # 2. Stems 間關聯性建模 (使用 3D attention)
#         self.stem_attention = nn.MultiheadAttention(
#             embed_dim=hidden_dim, 
#             num_heads=n_heads, 
#             batch_first=True
#         )

#         # 3. 時間維度建模 (保持 batch 和 stems 分離)
#         self.temporal_conv = nn.Sequential(
#             nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
        
#         # 4. 輸出層
#         self.output_proj = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, 1)
#         )

#         # 5. 時間維度映射
#         self.time_mapping = nn.Linear(latent_height, output_length)
#         self.sigmoid = nn.Sigmoid()
        
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
#                 init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
#                     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#                     init.uniform_(m.bias, -bound, bound)

#     def forward(self, z):
#         # z: [B, num_stems, latent_channels, latent_height, latent_width]
#         B, num_stems, C, H, W = z.shape
        
#         # Reshape for processing
#         z_flat = z.view(B * num_stems, C, H, W)
        
#         # 1. 時頻特徵提取
#         x = self.time_freq_conv(z_flat)  # [B*num_stems, hidden_dim, latent_height, 1]
#         x = x.squeeze(-1)  # [B*num_stems, hidden_dim, latent_height]
        
#         # 2. 重塑為 [B*num_stems, latent_height, hidden_dim] 用於attention
#         x = x.transpose(1, 2)  # [B*num_stems, latent_height, hidden_dim]
        
#         # 3. Stems 間關聯性建模 (對每個時間步應用attention)
#         # 重塑為 [B, num_stems, latent_height, hidden_dim]
#         x = x.view(B, num_stems, H, -1)
#         # 對每個時間步應用attention
#         attended_features = []
#         for t in range(H):
#             time_slice = x[:, :, t, :]  # [B, num_stems, hidden_dim]
#             attended, _ = self.stem_attention(time_slice, time_slice, time_slice)
#             attended_features.append(attended)
        
#         # 重新組合
#         x = torch.stack(attended_features, dim=2)  # [B, num_stems, latent_height, hidden_dim]
        
#         # 4. 時間維度建模
#         # 重塑為 [B*num_stems, hidden_dim, latent_height]
#         x = x.view(B * num_stems, -1, H)
#         x = self.temporal_conv(x)  # [B*num_stems, hidden_dim, latent_height]
        
#         # 5. 輸出層
#         # 對每個時間步進行預測
#         outputs = []
#         for t in range(H):
#             time_feature = x[:, :, t]  # [B*num_stems, hidden_dim]
#             output = self.output_proj(time_feature)  # [B*num_stems, 1]
#             outputs.append(output)
        
#         # 組合所有時間步的輸出
#         x = torch.stack(outputs, dim=2)  # [B*num_stems, 1, latent_height]
#         x = x.squeeze(1)  # [B*num_stems, latent_height]
        
#         # 6. 時間維度映射到目標長度
#         x = self.time_mapping(x)  # [B*num_stems, output_length]
        
#         # 7. 應用sigmoid並重塑回原始形狀
#         x = self.sigmoid(x)  # [B*num_stems, output_length]
#         x = x.view(B, num_stems, self.output_length)  # [B, num_stems, output_length]
        
#         return x


def plot_onset_pianoroll_comparison(pred_onset, gt_onset, save_path, epoch, step, sample_idx, stem_names=None):
    """
    繪製預測和ground truth的onset pianoroll比較圖
    
    Args:
        pred_onset: 預測的onset pianoroll [num_stems, time_steps]
        gt_onset: ground truth onset pianoroll [num_stems, time_steps]
        save_path: 保存路徑
        epoch: 當前epoch
        step: 當前step
        sample_idx: 樣本索引
        stem_names: 樂器名稱列表
    """
    if stem_names is None:
        stem_names = ["Kick", "Snare", "Toms", "Hi-Hats", "Cymbals"]
    
    # 創建子圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor('white')
    
    # 設置標題
    fig.suptitle(f'Onset Pianoroll Comparison - Epoch {epoch}, Step {step}, Sample {sample_idx}', 
                 fontsize=16, color='black', fontweight='bold')
    
    # 繪製預測結果 - 背景白色，onset黑色
    im1 = ax1.imshow(pred_onset, cmap='gray_r', aspect='auto', origin='lower', 
                     vmin=0, vmax=1, interpolation='nearest')
    ax1.set_title('Prediction', fontsize=14, color='black', fontweight='bold')
    ax1.set_xlabel('Time Steps', fontsize=12, color='black')
    ax1.set_ylabel('Instruments', fontsize=12, color='black')
    ax1.set_yticks(range(len(stem_names)))
    ax1.set_yticklabels(stem_names, fontsize=10, color='black')
    ax1.tick_params(axis='x', colors='black')
    ax1.tick_params(axis='y', colors='black')
    ax1.grid(False)
    
    # 繪製ground truth - 背景白色，onset黑色
    im2 = ax2.imshow(gt_onset, cmap='gray_r', aspect='auto', origin='lower', 
                     vmin=0, vmax=1, interpolation='nearest')
    ax2.set_title('Ground Truth', fontsize=14, color='black', fontweight='bold')
    ax2.set_xlabel('Time Steps', fontsize=12, color='black')
    ax2.set_ylabel('Instruments', fontsize=12, color='black')
    ax2.set_yticks(range(len(stem_names)))
    ax2.set_yticklabels(stem_names, fontsize=10, color='black')
    ax2.tick_params(axis='x', colors='black')
    ax2.tick_params(axis='y', colors='black')
    ax2.grid(False)
    
    # 設置背景色
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    # 調整佈局
    plt.tight_layout()
    
    # 保存圖像
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved onset pianoroll comparison to: {save_path}")


def plot_onset_pianoroll_combined(pred_onset, gt_onset, save_path, epoch, step, sample_idx, stem_names=None):
    """
    繪製預測和ground truth的onset pianoroll組合圖（上下排列）
    
    Args:
        pred_onset: 預測的onset pianoroll [num_stems, time_steps]
        gt_onset: ground truth onset pianoroll [num_stems, time_steps]
        save_path: 保存路徑
        epoch: 當前epoch
        step: 當前step
        sample_idx: 樣本索引
        stem_names: 樂器名稱列表
    """
    if stem_names is None:
        stem_names = ["Kick", "Snare", "Toms", "Hi-Hats", "Cymbals"]
    
    # 創建子圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    fig.patch.set_facecolor('white')
    
    # 設置標題
    fig.suptitle(f'Onset Pianoroll Combined - Epoch {epoch}, Step {step}, Sample {sample_idx}', 
                 fontsize=16, color='black', fontweight='bold')
    
    # 繪製預測結果（上方）- 背景白色，onset黑色
    im1 = ax1.imshow(pred_onset, cmap='gray_r', aspect='auto', origin='lower', 
                     vmin=0, vmax=1, interpolation='nearest')
    ax1.set_title('Prediction', fontsize=14, color='black', fontweight='bold')
    ax1.set_ylabel('Instruments', fontsize=12, color='black')
    ax1.set_yticks(range(len(stem_names)))
    ax1.set_yticklabels(stem_names, fontsize=10, color='black')
    ax1.tick_params(axis='x', colors='black')
    ax1.tick_params(axis='y', colors='black')
    ax1.grid(False)
    
    # 繪製ground truth（下方）- 背景白色，onset黑色
    im2 = ax2.imshow(gt_onset, cmap='gray_r', aspect='auto', origin='lower', 
                     vmin=0, vmax=1, interpolation='nearest')
    ax2.set_title('Ground Truth', fontsize=14, color='black', fontweight='bold')
    ax2.set_xlabel('Time Steps', fontsize=12, color='black')
    ax2.set_ylabel('Instruments', fontsize=12, color='black')
    ax2.set_yticks(range(len(stem_names)))
    ax2.set_yticklabels(stem_names, fontsize=10, color='black')
    ax2.tick_params(axis='x', colors='black')
    ax2.tick_params(axis='y', colors='black')
    ax2.grid(False)
    
    # 設置背景色
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    # 調整佈局
    plt.tight_layout()
    
    # 保存圖像
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


if __name__ == "__main__":
    import yaml

    model_config = "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/stable-diffusion/models/ldm/text2img256/config.yaml"
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)

    latent_diffusion = LatentDiffusion(**model_config["model"]["params"])

    import ipdb

    ipdb.set_trace()