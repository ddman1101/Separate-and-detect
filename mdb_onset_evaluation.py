import os
import argparse
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import glob
from madmom.features.onsets import (
    CNNOnsetProcessor,
    OnsetPeakPickingProcessor,
    SpectralOnsetProcessor
)
from madmom.audio.signal import Signal
import scipy.signal
from collections import defaultdict
import json

# MDB標籤到5-stem的映射 (根據MDB Drums官方分類)
MDB_TO_STEM_MAPPING = {
    # Kick Drum
    'KD': 'kick',
    
    # Snare Drum (包括各種技巧)
    'SD': 'snare',     # snare drum
    'SDB': 'snare',    # snare drum: brush
    'SDD': 'snare',    # snare drum: drag
    'SDF': 'snare',    # snare drum: flam
    'SDG': 'snare',    # snare drum: ghost note
    'SDNS': 'snare',   # snare drum: no snare
    
    # Hi-Hat
    'CHH': 'hi_hats',  # closed hi-hat
    'OHH': 'hi_hats',  # open hi-hat
    'PHH': 'hi_hats',  # pedal hi-hat
    
    # Toms
    'HIT': 'toms',     # high tom
    'MHT': 'toms',     # high-mid tom
    'HFT': 'toms',     # high floor tom
    'LFT': 'toms',     # low floor tom
    
    # Cymbals
    'RDC': 'cymbals',  # ride cymbal
    'RDB': 'cymbals',  # ride cymbal bell
    'CRC': 'cymbals',  # crash cymbal
    'CHC': 'cymbals',  # china cymbal
    'SPC': 'cymbals',  # splash cymbal
    
    # Other Percussion
    'SST': 'snare',    # side stick (歸類為snare)
    'TMB': 'cymbals',  # tambourine (歸類為cymbals)
    
    # MDB實際使用的標籤格式
    'TT': 'toms',      # Toms (在MDB中實際使用)
    'OT': 'toms',      # Other (在MDB中實際使用，歸類為toms)
    
    # 保留一些可能的舊格式
    'HH': 'hi_hats',   # 如果HH代表hi-hat
    'CY': 'cymbals',   # 如果CY代表cymbals
}

# 5-stem的MIDI音符映射
STEM_MIDI_NOTES = {
    'kick': 36,      # C2
    'snare': 38,     # D2
    'toms': 45,      # A2 (代表所有toms)
    'hi_hats': 42,   # F#2
    'cymbals': 49,   # C#3 (代表所有cymbals)
}

def butter_lowpass(cutoff, sr, order=5):
    """創建低通濾波器係數"""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype='lowpass', analog=False)
    return b, a

def butter_highpass(cutoff, sr, order=5):
    """創建高通濾波器係數"""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype='highpass', analog=False)
    return b, a

def apply_filter(audio, cutoff, sr, filter_type='highpass'):
    """應用濾波器到音頻信號"""
    if filter_type == 'highpass':
        b, a = butter_highpass(cutoff, sr)
    else:
        b, a = scipy.signal.butter(5, cutoff / (0.5 * sr), btype='lowpass', analog=False)
    filtered = scipy.signal.filtfilt(b, a, audio)
    return filtered

def detect_kick_onsets(audio_path, threshold=0.8, min_separation=0.08, cutoff=3417.3793722594723):
    """檢測Kick onsets - 使用原本Drum_Transcription的方法"""
    if not os.path.isfile(audio_path):
        return []
    
    # 讀取音檔
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # 如果有指定 cutoff frequency，套用 low-pass filter
    if cutoff is not None and cutoff < sr/2:
        b, a = butter_lowpass(cutoff, sr)
        audio = scipy.signal.filtfilt(b, a, audio)
    
    # 加入 1 秒空白
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
    
    # 基本邊界過濾：移除接近音頻開頭和結尾的onsets
    duration = len(audio) / sr
    boundary_threshold = 0.2  # 0.2秒邊界閾值
    
    filtered = []
    for onset in detected_onsets_cnn:
        # 過濾邊界onsets
        if onset < boundary_threshold or onset > (duration - boundary_threshold):
            continue
        filtered.append(onset)
    
    return filtered

def detect_snare_onsets(audio_path, threshold=0.5, min_separation=0.12, cutoff=2832):
    """檢測Snare onsets - 改用CNNOnsetProcessor避免onsets擠在一起"""
    if not os.path.isfile(audio_path):
        return []
    
    # 讀取音檔時指定 sample rate
    sr = 16000
    audio, sr = librosa.load(audio_path, sr=sr)
    
    # 優化cutoff frequency: 可調參數，保留更多snare的中高頻特性
    nyquist = sr / 2
    cutoff_freq = cutoff  # 可調參數，保留snare的crack和body
    normal_cutoff = cutoff_freq / nyquist
    b, a = scipy.signal.butter(5, normal_cutoff, btype='low')
    audio = scipy.signal.filtfilt(b, a, audio)
    
    # 加入 1 秒空白
    offset_sec = 1.0
    offset_samples = int(offset_sec * sr)
    padded_audio = np.concatenate([np.zeros(offset_samples), audio])
    padded_signal = Signal(padded_audio, sample_rate=sr)
    
    # 改用CNNOnsetProcessor，避免SpectralOnsetProcessor造成的onsets擠在一起
    proc_CNN = CNNOnsetProcessor()  
    activation_cnn = proc_CNN(padded_signal)
    
    # 使用更保守的peak picking參數
    peak_picking = OnsetPeakPickingProcessor(threshold=threshold,
                                           pre_avg=0.03, post_avg=0.03, 
                                           pre_max=0.03, post_max=0.03)
    detected_onsets_cnn_padded = peak_picking(activation_cnn)
    detected_onsets_cnn = detected_onsets_cnn_padded - offset_sec
    detected_onsets_cnn[detected_onsets_cnn < 0] = 0
    
    # 基本邊界過濾：移除接近音頻開頭和結尾的onsets
    duration = len(audio) / sr
    boundary_threshold = 0.2  # 0.2秒邊界閾值
    
    filtered = []
    for onset in detected_onsets_cnn:
        # 過濾邊界onsets
        if onset < boundary_threshold or onset > (duration - boundary_threshold):
            continue
        filtered.append(onset)
    
    return filtered

def detect_toms_onsets(audio_path, threshold=0.4, min_separation=0.2, cutoff_low=299, cutoff_high=2244):
    """
    檢測Toms onsets - 再不敏感一點。
    可選可調band-pass：cutoff_low（高通）與 cutoff_high（低通）。
    預設維持原本 299/2244。
    """
    if not os.path.isfile(audio_path):
        return []
    
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # 調整band-pass filter（可調）：參考鼓混音技巧
    if cutoff_low is not None and cutoff_low > 0:
        audio = apply_filter(audio, cutoff_low, sr, 'highpass')
    if cutoff_high is not None and cutoff_high > 0 and cutoff_high < sr / 2:
        audio = apply_filter(audio, cutoff_high, sr, 'lowpass')
    
    # 加入 1 秒空白 (使用原本Drum_Transcription的方法)
    offset_sec = 1.0
    offset_samples = int(offset_sec * sr)
    padded_audio = np.concatenate([np.zeros(offset_samples), audio])
    padded_signal = Signal(padded_audio, sample_rate=sr)
    
    # 使用CNNOnsetProcessor
    proc_CNN = CNNOnsetProcessor()  
    activation_cnn = proc_CNN(padded_signal)
    
    # 使用更保守的peak picking參數，再不敏感一點
    peak_picking = OnsetPeakPickingProcessor(threshold=threshold,
                                           pre_avg=0.03, post_avg=0.03, 
                                           pre_max=0.03, post_max=0.03)
    detected_onsets_cnn_padded = peak_picking(activation_cnn)
    detected_onsets_cnn = detected_onsets_cnn_padded - offset_sec
    detected_onsets_cnn[detected_onsets_cnn < 0] = 0
    
    # 基本邊界過濾：移除接近音頻開頭和結尾的onsets
    duration = len(audio) / sr
    boundary_threshold = 0.2  # 0.2秒邊界閾值
    
    filtered = []
    for onset in detected_onsets_cnn:
        # 過濾邊界onsets
        if onset < boundary_threshold or onset > (duration - boundary_threshold):
            continue
        filtered.append(onset)
    
    return filtered

def detect_hi_hats_onsets(audio_path, threshold=0.4, min_separation=0.08, cutoff=600.5530232910016):
    """檢測Hi-Hats onsets - 調整回來到較好的設定"""
    if not os.path.isfile(audio_path):
        return []
    
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # 調整cutoff frequency: 可調參數，參考hi-hats混音技巧
    # 根據hi-hats混音指南，主要頻率在1-2KHz以上
    cutoff_freq = cutoff  # 可調參數，適中的高頻過濾
    audio = apply_filter(audio, cutoff_freq, sr, 'highpass')
    
    # 加入 1 秒空白 (使用原本Drum_Transcription的方法)
    offset_sec = 1.0
    offset_samples = int(offset_sec * sr)
    padded_audio = np.concatenate([np.zeros(offset_samples), audio])
    padded_signal = Signal(padded_audio, sample_rate=sr)
    
    # 使用CNNOnsetProcessor (原本Drum_Transcription的方法)
    proc_CNN = CNNOnsetProcessor()  
    activation_cnn = proc_CNN(padded_signal)
    
    # 使用適中的peak picking參數
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
    """檢測Cymbals onsets - 回到剛剛的設定"""
    if not os.path.isfile(audio_path):
        return []
    
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # 回到剛剛的設定: 可調參數高頻過濾
    cutoff_freq = cutoff  # 可調參數，適中的高頻過濾
    audio = apply_filter(audio, cutoff_freq, sr, 'highpass')
    
    # 加入 1 秒空白 (使用原本Drum_Transcription的方法)
    offset_sec = 1.0
    offset_samples = int(offset_sec * sr)
    padded_audio = np.concatenate([np.zeros(offset_samples), audio])
    padded_signal = Signal(padded_audio, sample_rate=sr)
    
    # 改用CNNOnsetProcessor，更穩定的檢測
    proc_CNN = CNNOnsetProcessor()  
    activation_cnn = proc_CNN(padded_signal)
    
    # 使用適中的peak picking參數
    peak_picking = OnsetPeakPickingProcessor(threshold=threshold,
                                           pre_avg=0.03, post_avg=0.03, 
                                           pre_max=0.03, post_max=0.03)
    detected_onsets_cnn_padded = peak_picking(activation_cnn)
    detected_onsets_cnn = detected_onsets_cnn_padded - offset_sec
    detected_onsets_cnn[detected_onsets_cnn < 0] = 0
    
    # 基本邊界過濾：移除接近音頻開頭和結尾的onsets
    duration = len(audio) / sr
    boundary_threshold = 0.2  # 0.2秒邊界閾值
    
    filtered = []
    for onset in detected_onsets_cnn:
        # 過濾邊界onsets
        if onset < boundary_threshold or onset > (duration - boundary_threshold):
            continue
        filtered.append(onset)
    
    return filtered

def load_mdb_annotations(annotation_path):
    """載入MDB註釋文件並轉換為5-stem格式"""
    if not os.path.isfile(annotation_path):
        return defaultdict(list)
    
    stem_onsets = defaultdict(list)
    
    with open(annotation_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            
            timestamp = float(parts[0])
            mdb_label = parts[1].strip()
            
            # 轉換MDB標籤到5-stem
            if mdb_label in MDB_TO_STEM_MAPPING:
                stem = MDB_TO_STEM_MAPPING[mdb_label]
                stem_onsets[stem].append(timestamp)
    
    return stem_onsets

def save_predicted_onsets(all_results, output_dir):
    """
    保存預測的 onset 到 txt 檔案，格式與 MDB annotation 相同
    """
    # 創建預測結果目錄
    pred_dir = os.path.join(output_dir, 'mdb_prediction_results', 'msg-ld')
    os.makedirs(pred_dir, exist_ok=True)
    
    # 按檔案分組結果
    file_groups = defaultdict(list)
    for result in all_results:
        file_groups[result['base_name']].append(result)
    
    # 為每個檔案創建預測 annotation
    for base_name, results in file_groups.items():
        # 收集所有預測的 onset
        all_pred_onsets = []
        
        for result in results:
            stem = result['stem']
            pred_onsets = result['pred_onsets']
            start_time = result['start_time']
            
            # 將相對時間轉換為絕對時間
            for onset in pred_onsets:
                absolute_time = start_time + onset
                # 映射到 MDB 標籤格式
                if stem == 'kick':
                    label = 'KD'
                elif stem == 'snare':
                    label = 'SD'
                elif stem == 'toms':
                    label = 'TT'
                elif stem == 'hi_hats':
                    label = 'HH'
                elif stem == 'cymbals':
                    label = 'CY'
                else:
                    continue
                
                all_pred_onsets.append((absolute_time, label))
        
        # 按時間排序
        all_pred_onsets.sort(key=lambda x: x[0])
        
        # 保存到檔案
        output_file = os.path.join(pred_dir, f"{base_name}_predicted.txt")
        with open(output_file, 'w') as f:
            for timestamp, label in all_pred_onsets:
                f.write(f"{timestamp:.6f}\t{label}\n")
        
        print(f"預測 onset 已保存到: {output_file} ({len(all_pred_onsets)} onsets)")

def calculate_f1_score(pred_onsets, gt_onsets, tolerance_ms=20):
    """計算F1 score"""
    tolerance_sec = tolerance_ms / 1000.0
    
    # 特殊情況：GT=0且Pred=0時，應該是完美的預測（F1=1.0）
    if len(gt_onsets) == 0 and len(pred_onsets) == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0
    
    # 特殊情況：GT=0但Pred>0時，全部是false positives
    if len(gt_onsets) == 0:
        return 0.0, 0.0, 0.0, 0, len(pred_onsets), 0
    
    # 計算True Positives
    tp = 0
    matched_gt = set()
    
    for pred in pred_onsets:
        for i, gt in enumerate(gt_onsets):
            if i in matched_gt:
                continue
            if abs(pred - gt) <= tolerance_sec:
                tp += 1
                matched_gt.add(i)
                break
    
    # 計算Precision, Recall, F1
    fp = len(pred_onsets) - tp
    fn = len(gt_onsets) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1, precision, recall, tp, fp, fn

def extract_segment_info(filename):
    """從文件名提取segment信息"""
    # 例如: MusicDelta_Beatles_MIX_from_0.wav
    if '_from_' in filename:
        base_name = filename.split('_from_')[0]
        segment_str = filename.split('_from_')[1].replace('.wav', '')
        try:
            segment_idx = int(segment_str) // 10  # from_10 -> segment 1, from_20 -> segment 2
            start_time = segment_idx * 10.24  # segment 1表示從10.24秒開始，segment 2表示從20.48秒開始
            # 移除_MIX後綴來匹配註釋文件名
            if base_name.endswith('_MIX'):
                base_name = base_name[:-4]  # 移除_MIX
            return base_name, segment_idx, start_time
        except ValueError:
            return base_name, 0, 0.0
    return filename.replace('.wav', ''), 0, 0.0

def main():
    parser = argparse.ArgumentParser(description='MDB Drums Onset Detection Evaluation')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='輸入目錄路徑 (包含stem_0, stem_1, stem_2, stem_3, stem_4子目錄)')
    parser.add_argument('--annotation_dir', type=str, 
                       default='/workspace/MDBDrums/MDB Drums/test/annotation',
                       help='MDB註釋目錄路徑')
    parser.add_argument('--output_dir', type=str, default='./onset_evaluation_results',
                       help='輸出結果目錄')
    parser.add_argument('--tolerance_20ms', action='store_true', default=True,
                       help='計算20ms誤差的F1 score')
    parser.add_argument('--tolerance_50ms', action='store_true', default=True,
                       help='計算50ms誤差的F1 score')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 定義stem映射
    stem_dirs = {
        'kick': 'stem_0',
        'snare': 'stem_1', 
        'toms': 'stem_2',
        'hi_hats': 'stem_3',
        'cymbals': 'stem_4'
    }
    
    # 檢測函數映射 - 根據用戶反饋調整參數
    # detection_functions = {
    #     'kick': lambda x: detect_kick_onsets(x, threshold=0.7135728547393483, min_separation=0.10898222909192971, cutoff=2915.969206977644),    # Kick表現良好，保持不變
    #     'snare': lambda x: detect_snare_onsets(x, threshold=0.38753227655124056, min_separation=0.13411913739083473, cutoff=2830.6395423513873),  # 提高threshold，增加間隔避免擠在一起
    #     'toms': lambda x: detect_toms_onsets(x, threshold=0.5089708671850919, min_separation=0.2231750922666378, cutoff_low=299, cutoff_high=3217.6689426465955),     # 再不敏感一點
    #     'hi_hats': lambda x: detect_hi_hats_onsets(x, threshold=0.32803374320834777, min_separation=0.11616325049104322, cutoff=680.1679152297679),  # 調整回來到較好的設定
    #     'cymbals': lambda x: detect_cymbals_onsets(x, threshold=0.5503322287783717, min_separation=0.10970467528509326, cutoff=1510.1722196545102)   # 回到剛剛的設定
    # }
    detection_functions = {
        # 'kick': lambda x: detect_kick_onsets(x, threshold=0.7135728547393483, min_separation=0.10898222909192971, cutoff=2915.969206977644),
        'kick': lambda x: detect_kick_onsets(x, threshold=0.7123620356542087, min_separation=0.12605714451279332, cutoff=5659.969709057025),    # Kick表現良好，保持不變
        'snare': lambda x: detect_snare_onsets(x, threshold=0.24425729153651546, min_separation=0.059295203501098154, cutoff=1248.559112730783),  # 提高threshold，增加間隔避免擠在一起
        'toms': lambda x: detect_toms_onsets(x, threshold=0.5571172192068058, min_separation=0.19440705946102893, cutoff_low=299, cutoff_high=1185.8016508799908),     # 再不敏感一點
        'hi_hats': lambda x: detect_hi_hats_onsets(x, threshold=0.08597749010989753, min_separation=0.06948347430929988, cutoff=517.9637529565225),  # 調整回來到較好的設定
        'cymbals': lambda x: detect_cymbals_onsets(x, threshold=0.39457664043870916, min_separation=0.1796002881571787, cutoff=1115.370160765871)   # 回到剛剛的設定
    }
    # 收集所有結果
    all_results = []
    
    # 獲取所有音頻文件
    stem_0_dir = os.path.join(args.input_dir, 'stem_0')
    if not os.path.exists(stem_0_dir):
        print(f"錯誤: 找不到stem_0目錄: {stem_0_dir}")
        return
    
    audio_files = glob.glob(os.path.join(stem_0_dir, '*.wav'))
    
    print(f"找到 {len(audio_files)} 個音頻文件")
    
    # 靜默處理，不印出每個文件的詳細結果
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        base_name, segment_idx, start_time = extract_segment_info(filename)
        
        print(f"處理: {filename} (segment {segment_idx}, start_time: {start_time:.2f}s)")
        
        # 載入對應的註釋
        annotation_file = os.path.join(args.annotation_dir, f"{base_name}_class.txt")
        gt_onsets = load_mdb_annotations(annotation_file)
        
        # 過濾GT onsets到當前segment範圍
        segment_gt_onsets = defaultdict(list)
        for stem, onsets in gt_onsets.items():
            for onset in onsets:
                if start_time <= onset < start_time + 10.24:
                    segment_gt_onsets[stem].append(onset - start_time)
        
        # 對每個stem進行onset detection
        for stem, stem_dir in stem_dirs.items():
            stem_audio_file = os.path.join(args.input_dir, stem_dir, filename)
            
            if not os.path.exists(stem_audio_file):
                print(f"警告: 找不到文件 {stem_audio_file}")
                continue
            
            # 檢測onsets
            pred_onsets = detection_functions[stem](stem_audio_file)
            
            # 獲取對應的GT onsets
            gt_onsets_stem = segment_gt_onsets[stem]
            
            # 計算F1 scores
            f1_20, p_20, r_20, tp_20, fp_20, fn_20 = calculate_f1_score(pred_onsets, gt_onsets_stem, 20)
            f1_50, p_50, r_50, tp_50, fp_50, fn_50 = calculate_f1_score(pred_onsets, gt_onsets_stem, 50)
            
            # 保存結果
            result = {
                'filename': filename,
                'base_name': base_name,
                'segment_idx': segment_idx,
                'start_time': start_time,
                'stem': stem,
                'pred_onsets_count': len(pred_onsets),
                'gt_onsets_count': len(gt_onsets_stem),
                'f1_20ms': f1_20,
                'precision_20ms': p_20,
                'recall_20ms': r_20,
                'tp_20ms': tp_20,
                'fp_20ms': fp_20,
                'fn_20ms': fn_20,
                'f1_50ms': f1_50,
                'precision_50ms': p_50,
                'recall_50ms': r_50,
                'tp_50ms': tp_50,
                'fp_50ms': fp_50,
                'fn_50ms': fn_50,
                'pred_onsets': pred_onsets,
                'gt_onsets': gt_onsets_stem
            }
            all_results.append(result)
            
            print(f"  {stem}: F1@20ms={f1_20:.3f}, F1@50ms={f1_50:.3f} (pred:{len(pred_onsets)}, gt:{len(gt_onsets_stem)})")
    
    # 保存詳細結果
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(args.output_dir, 'detailed_results.csv')
    results_df.to_csv(results_file, index=False)
    # print(f"詳細結果已保存到: {results_file}")  # 靜默處理
    
    # 保存預測的 onset 到 txt 檔案
    save_predicted_onsets(all_results, args.output_dir)
    
    # 計算總體統計
    # print("\n=== 總體統計 ===")  # 靜默處理

    # 靜默處理所有中間統計輸出
    # print("\n=== 每個 Stem 指標彙整 @20ms (Micro vs Macro) ===")
    # for stem in stem_dirs.keys():
    #     stem_results = results_df[results_df['stem'] == stem]
    #     if len(stem_results) == 0:
    #         continue
    #     tp20 = stem_results['tp_20ms'].sum()
    #     fp20 = stem_results['fp_20ms'].sum()
    #     fn20 = stem_results['fn_20ms'].sum()
    #     micro_p20 = tp20 / (tp20 + fp20) if (tp20 + fp20) > 0 else 0.0
    #     micro_r20 = tp20 / (tp20 + fn20) if (tp20 + fn20) > 0 else 0.0
    #     micro_f120 = (2 * micro_p20 * micro_r20 / (micro_p20 + micro_r20)) if (micro_p20 + micro_r20) > 0 else 0.0
    #     macro_f1_20 = stem_results['f1_20ms'].mean()
    #     print(f"{stem.upper()}: Micro F1={micro_f120:.4f}, Micro P={micro_p20:.4f}, Micro R={micro_r20:.4f}, Macro F1={macro_f1_20:.4f}")

    # print("\n=== 每個 Stem 指標彙整 @50ms (Micro vs Macro) ===")
    # for stem in stem_dirs.keys():
    #     stem_results = results_df[results_df['stem'] == stem]
    #     if len(stem_results) == 0:
    #         continue
    #     tp50 = stem_results['tp_50ms'].sum()
    #     fp50 = stem_results['fp_50ms'].sum()
    #     fn50 = stem_results['fn_50ms'].sum()
    #     micro_p50 = tp50 / (tp50 + fp50) if (tp50 + fp50) > 0 else 0.0
    #     micro_r50 = tp50 / (tp50 + fn50) if (tp50 + fn50) > 0 else 0.0
    #     micro_f150 = (2 * micro_p50 * micro_r50 / (micro_p50 + micro_r50)) if (micro_p50 + micro_r50) > 0 else 0.0
    #     macro_f1_50 = stem_results['f1_50ms'].mean()
    #     print(f"{stem.upper()}: Micro F1={micro_f150:.4f}, Micro P={micro_p50:.4f}, Micro R={micro_r50:.4f}, Macro F1={macro_f1_50:.4f}")
    
    # 靜默處理按stem分組統計
    # for stem in stem_dirs.keys():
    #     stem_results = results_df[results_df['stem'] == stem]
    #     if len(stem_results) == 0:
    #         continue
    #         
    #     print(f"\n{stem.upper()}:")
    #     print(f"  文件數量: {len(stem_results)}")
    #     print(f"  F1@20ms: {stem_results['f1_20ms'].mean():.3f} ± {stem_results['f1_20ms'].std():.3f}")
    #     print(f"  F1@50ms: {stem_results['f1_50ms'].mean():.3f} ± {stem_results['f1_50ms'].std():.3f}")
    #     print(f"  平均預測onsets: {stem_results['pred_onsets_count'].mean():.1f}")
    #     print(f"  平均GT onsets: {stem_results['gt_onsets_count'].mean():.1f}")

    #     # Micro（以該 stem 的 TP/FP/FN 加總計算）
    #     tp20 = stem_results['tp_20ms'].sum()
    #     fp20 = stem_results['fp_20ms'].sum()
    #     fn20 = stem_results['fn_20ms'].sum()
    #     micro_p20 = tp20 / (tp20 + fp20) if (tp20 + fp20) > 0 else 0.0
    #     micro_r20 = tp20 / (tp20 + fn20) if (tp20 + fn20) > 0 else 0.0
    #     micro_f120 = (2 * micro_p20 * micro_r20 / (micro_p20 + micro_r20)) if (micro_p20 + micro_r20) > 0 else 0.0

    #     tp50 = stem_results['tp_50ms'].sum()
    #     fp50 = stem_results['fp_50ms'].sum()
    #     fn50 = stem_results['fn_50ms'].sum()
    #     micro_p50 = tp50 / (tp50 + fp50) if (tp50 + fp50) > 0 else 0.0
    #     micro_r50 = tp50 / (tp50 + fn50) if (tp50 + fn50) > 0 else 0.0
    #     micro_f150 = (2 * micro_p50 * micro_r50 / (micro_p50 + micro_r50)) if (micro_p50 + micro_r50) > 0 else 0.0

    #     print(f"  Micro@20ms: F1={micro_f120:.3f}, P={micro_p20:.3f}, R={micro_r20:.3f}")
    #     print(f"  Micro@50ms: F1={micro_f150:.3f}, P={micro_p50:.3f}, R={micro_r50:.3f}")
    
    # 靜默處理總體平均
    # print(f"\n總體平均:")
    # print(f"  F1@20ms: {results_df['f1_20ms'].mean():.3f} ± {results_df['f1_20ms'].std():.3f}")
    # print(f"  F1@50ms: {results_df['f1_50ms'].mean():.3f} ± {results_df['f1_50ms'].std():.3f}")

    # 總體 micro（跨所有 stems 加總 TP/FP/FN）
    total_tp20 = results_df['tp_20ms'].sum()
    total_fp20 = results_df['fp_20ms'].sum()
    total_fn20 = results_df['fn_20ms'].sum()
    total_tp50 = results_df['tp_50ms'].sum()
    total_fp50 = results_df['fp_50ms'].sum()
    total_fn50 = results_df['fn_50ms'].sum()

    overall_micro_p20 = total_tp20 / (total_tp20 + total_fp20) if (total_tp20 + total_fp20) > 0 else 0.0
    overall_micro_r20 = total_tp20 / (total_tp20 + total_fn20) if (total_tp20 + total_fn20) > 0 else 0.0
    overall_micro_f120 = (2 * overall_micro_p20 * overall_micro_r20 / (overall_micro_p20 + overall_micro_r20)) if (overall_micro_p20 + overall_micro_r20) > 0 else 0.0

    overall_micro_p50 = total_tp50 / (total_tp50 + total_fp50) if (total_tp50 + total_fp50) > 0 else 0.0
    overall_micro_r50 = total_tp50 / (total_tp50 + total_fn50) if (total_tp50 + total_fn50) > 0 else 0.0
    overall_micro_f150 = (2 * overall_micro_p50 * overall_micro_r50 / (overall_micro_p50 + overall_micro_r50)) if (overall_micro_p50 + overall_micro_r50) > 0 else 0.0

    # print(f"  Micro@20ms: F1={overall_micro_f120:.3f}, P={overall_micro_p20:.3f}, R={overall_micro_r20:.3f}")
    # print(f"  Micro@50ms: F1={overall_micro_f150:.3f}, P={overall_micro_p50:.3f}, R={overall_micro_r50:.3f}")
    
    # 保存總體統計
    summary_stats = {
        'overall': {
            'f1_20ms_mean': float(results_df['f1_20ms'].mean()),
            'f1_20ms_std': float(results_df['f1_20ms'].std()),
            'f1_50ms_mean': float(results_df['f1_50ms'].mean()),
            'f1_50ms_std': float(results_df['f1_50ms'].std()),
            'precision_20ms_mean': float(results_df['precision_20ms'].mean()),
            'precision_20ms_std': float(results_df['precision_20ms'].std()),
            'precision_50ms_mean': float(results_df['precision_50ms'].mean()),
            'precision_50ms_std': float(results_df['precision_50ms'].std()),
            'recall_20ms_mean': float(results_df['recall_20ms'].mean()),
            'recall_20ms_std': float(results_df['recall_20ms'].std()),
            'recall_50ms_mean': float(results_df['recall_50ms'].mean()),
            'recall_50ms_std': float(results_df['recall_50ms'].std()),
            'total_files': len(results_df),
            'micro_precision_20ms': float(overall_micro_p20),
            'micro_recall_20ms': float(overall_micro_r20),
            'micro_f1_20ms': float(overall_micro_f120),
            'micro_precision_50ms': float(overall_micro_p50),
            'micro_recall_50ms': float(overall_micro_r50),
            'micro_f1_50ms': float(overall_micro_f150)
        }
    }
    
    for stem in stem_dirs.keys():
        stem_results = results_df[results_df['stem'] == stem]
        if len(stem_results) > 0:
            # 該 stem 的 micro（僅在該 stem 內加總）
            st_tp20 = stem_results['tp_20ms'].sum()
            st_fp20 = stem_results['fp_20ms'].sum()
            st_fn20 = stem_results['fn_20ms'].sum()
            st_tp50 = stem_results['tp_50ms'].sum()
            st_fp50 = stem_results['fp_50ms'].sum()
            st_fn50 = stem_results['fn_50ms'].sum()
            st_mp20 = st_tp20 / (st_tp20 + st_fp20) if (st_tp20 + st_fp20) > 0 else 0.0
            st_mr20 = st_tp20 / (st_tp20 + st_fn20) if (st_tp20 + st_fn20) > 0 else 0.0
            st_mf120 = (2 * st_mp20 * st_mr20 / (st_mp20 + st_mr20)) if (st_mp20 + st_mr20) > 0 else 0.0
            st_mp50 = st_tp50 / (st_tp50 + st_fp50) if (st_tp50 + st_fp50) > 0 else 0.0
            st_mr50 = st_tp50 / (st_tp50 + st_fn50) if (st_tp50 + st_fn50) > 0 else 0.0
            st_mf150 = (2 * st_mp50 * st_mr50 / (st_mp50 + st_mr50)) if (st_mp50 + st_mr50) > 0 else 0.0

            summary_stats[stem] = {
                'f1_20ms_mean': float(stem_results['f1_20ms'].mean()),
                'f1_20ms_std': float(stem_results['f1_20ms'].std()),
                'f1_50ms_mean': float(stem_results['f1_50ms'].mean()),
                'f1_50ms_std': float(stem_results['f1_50ms'].std()),
                'precision_20ms_mean': float(stem_results['precision_20ms'].mean()),
                'precision_20ms_std': float(stem_results['precision_20ms'].std()),
                'precision_50ms_mean': float(stem_results['precision_50ms'].mean()),
                'precision_50ms_std': float(stem_results['precision_50ms'].std()),
                'recall_20ms_mean': float(stem_results['recall_20ms'].mean()),
                'recall_20ms_std': float(stem_results['recall_20ms'].std()),
                'recall_50ms_mean': float(stem_results['recall_50ms'].mean()),
                'recall_50ms_std': float(stem_results['recall_50ms'].std()),
                'file_count': len(stem_results),
                'micro_precision_20ms': float(st_mp20),
                'micro_recall_20ms': float(st_mr20),
                'micro_f1_20ms': float(st_mf120),
                'micro_precision_50ms': float(st_mp50),
                'micro_recall_50ms': float(st_mr50),
                'micro_f1_50ms': float(st_mf150)
            }
    
    summary_file = os.path.join(args.output_dir, 'summary_stats.json')
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    # print(f"總體統計已保存到: {summary_file}")  # 靜默處理

if __name__ == "__main__":
    main()