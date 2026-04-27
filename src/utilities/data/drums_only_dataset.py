import torch
import torchaudio
import numpy as np
import os
import utilities.audio as Audio
from torch.utils.data import Dataset

class DrumsOnlyDataset(Dataset):
    """
    Dedicated dataset handling for folders containing only drums.wav
    Used for MDB test set inference
    """
    def __init__(self, dataset_path, label_path, config, train=True, factor=1.0, whole_track=False):
        super().__init__()
        self.train = train
        self.config = config
        self.whole_track = whole_track
        
        # Audio processing parameters
        self.melbins = config["preprocessing"]["mel"]["n_mel_channels"]
        self.freqm = config["preprocessing"]["mel"]["freqm"]
        self.timem = config["preprocessing"]["mel"]["timem"]
        self.mixup = config["augmentation"]["mixup"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = config["preprocessing"]["stft"]["hop_length"]
        self.target_length = config["preprocessing"]["mel"]["target_length"]
        self.use_blur = config["preprocessing"]["mel"]["blur"]
        self.segment_length = int(self.target_length * self.hopsize)
        
        # Onset-pianoroll parameters
        self.hop_length = 160  # corresponds to 0.25s hop length
        self.num_stems = 5  # kick, snare, toms, hi_hats, cymbals
        
        # Read data
        self.data = self.read_datafile(dataset_path, label_path, train)
            
        print(f"Drums Only Dataset: {len(self.data)} tracks loaded")
        
        # Control dataset size via factor
        self.factor = factor
        self.total_len = int(len(self.data) * factor)
        print(f"Drums Only Dataset: {len(self.data)} tracks loaded, factor={factor}, total_len={self.total_len}")
        
        # STFT settings
        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )
        
        if not train:
            self.mixup = 0.0
            self.freqm = 0
            self.timem = 0

    def get_duration_sec(self, file, cache=False):
        if not os.path.exists(file):
            return 0
        try:
            # Attempt to read cached duration from a file
            with open(file + ".dur", "r") as f:
                duration = float(f.readline().strip("\n"))
        except FileNotFoundError:
            # If cached duration is not found, use torchaudio to find the actual duration
            audio_info = torchaudio.info(file)
            duration = audio_info.num_frames / audio_info.sample_rate
            if cache:
                # Cache the duration for future use
                with open(file + ".dur", "w") as f:
                    f.write(str(duration) + "\n")
        return duration

    def filter(self, tracks, audio_files_dir):
        # Keep only tracks that contain drums.wav
        keep = []
        durations = []
        for track in tracks:
            track_dir = os.path.join(audio_files_dir, track)
            drums_file = os.path.join(track_dir, "drums.wav")
            
            if not os.path.exists(drums_file):
                continue  # skip if drums.wav is missing
            
            duration = self.get_duration_sec(drums_file, cache=True) * self.config['preprocessing']['audio']['sampling_rate']
            
            # Duration filtering
            if (duration / self.config['preprocessing']['audio']['sampling_rate'] < 10.0):
                continue
            if (duration / self.config['preprocessing']['audio']['sampling_rate'] >= 640.0):
                print("skiping_file:", track)
                continue
                
            keep.append(track)
            durations.append(duration)
            
        print(f"sr={self.config['preprocessing']['audio']['sampling_rate']}, min: {10}, max: {600}")
        print(f"Keeping {len(keep)} of {len(tracks)} tracks")
        return keep, durations, np.cumsum(np.array(durations))

    def read_datafile(self, dataset_path, label_path, train):
        data = []
        # Load list of tracks and starts/durations
        tracks = os.listdir(dataset_path)
        print(f"Found {len(tracks)} tracks.")
        keep, durations, cumsum = self.filter(tracks, dataset_path)

        # Assuming keep, durations, and cumsum are lists of the same length
        for idx in range(len(keep)):
            # Construct a dictionary for each track with its name, duration, and cumulative sum
            track_info = {
                'wav_path': os.path.join(dataset_path, keep[idx]),
                'duration': durations[idx],
                'cumsum': cumsum[idx],
                'track_name': keep[idx]
            }
            data.append(track_info)
        return data

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        # Cycle through data
        index = index % len(self.data)
        track_info = self.data[index]
        
        # Read drums.wav
        drums_path = os.path.join(track_info['wav_path'], 'drums.wav')
        
        # Load audio
        waveform, sr = torchaudio.load(drums_path)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to target sampling rate
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            waveform = resampler(waveform)
        
        # Compute mel-spectrogram
        mel_spec = self.STFT.mel_spectrogram(waveform.squeeze(0).numpy())
        mel_spec = torch.from_numpy(mel_spec).float()
        
        # Create placeholders for 5 stems (all using the same drums audio)
        # In a real application, the model should separate drums.wav here
        fbank_stems = torch.stack([mel_spec] * self.num_stems)  # [5, mel_bins, time_frames]
        waveform_stems = torch.stack([waveform.squeeze(0)] * self.num_stems)  # [5, time_samples]
        
        # Create onset labels (all zeros, as there is no ground truth)
        onset_pianoroll = torch.zeros(self.num_stems, self.target_length)
        
        # Create timbre labels (all zeros, as there is no ground truth)
        timbre_features = torch.zeros(self.num_stems, 7)  # 7 timbre features
        
        return {
            'fbank': mel_spec,
            'fbank_stems': fbank_stems,
            'waveform': waveform.squeeze(0),
            'waveform_stems': waveform_stems,
            'onset_pianoroll': onset_pianoroll,
            'timbre_features': timbre_features,
            'track_name': track_info['track_name'],
            'wav_path': drums_path
        }