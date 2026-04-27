import sys

sys.path.append(
    "/PATH/TO/MUSICLDM_EXT/src"
)
import csv
import json
import wave
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import utilities.audio as Audio
import librosa
import os
import torchvision
import yaml
import pandas as pd
import omegaconf
import glob

STEM_JSON_MAP = {
    'kick': ["Bass Drum 1"],
    'snare': ["Acoustic Snare"],
    'toms': ["High Floor Tom", "Low-Mid Tom", "High Tom"],
    'hi-hats': ["Closed Hi Hat", "Open Hi Hat"],
    'cymbals': ["Crash Cymbal 1", "Ride Cymbal 1"]
}

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row["mid"]] = row["index"]
            line_count += 1
    return index_lookup

class TextDataset(Dataset):
    def __init__(self, data, logfile):
        super().__init__()
        self.data = data
        self.logfile = logfile
    def __getitem__(self, index):
        data_dict = {}
         # construct dict
        data_dict['fname'] = f"infer_file_{index}"
        data_dict['fbank'] = np.zeros((1024,64))
        data_dict['waveform'] = np.zeros((32000))
        data_dict['text'] = self.data[index]
        if index == 0:
            with open(os.path.join(self.logfile), 'w') as f:
                f.write(f"{data_dict['fname']}: {data_dict['text']}")
        else:
            with open(os.path.join(self.logfile), 'a') as f:
                f.write(f"\n{data_dict['fname']}: {data_dict['text']}")
        return data_dict


    def __len__(self):
        return len(self.data)


class AudiostockDataset(Dataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__()

        self.train = train
        self.config = config

        # self.read_datafile(dataset_path, label_path, train)

        self.melbins = config["preprocessing"]["mel"]["n_mel_channels"]
        self.freqm = config["preprocessing"]["mel"]["freqm"]
        self.timem = config["preprocessing"]["mel"]["timem"]
        self.mixup = config["augmentation"]["mixup"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = config["preprocessing"]["stft"]["hop_length"]
        self.target_length = config["preprocessing"]["mel"]["target_length"]
        self.use_blur = config["preprocessing"]["mel"]["blur"]
        self.segment_length = int(self.target_length * self.hopsize)
        self.whole_track = whole_track

        self.data = []
        if type(dataset_path) is str:
            self.data = self.read_datafile(dataset_path, label_path, train) 

        elif type(dataset_path) is list or type(dataset_path) is omegaconf.listconfig.ListConfig:
            for datapath in dataset_path:
                self.data +=  self.read_datafile(datapath, label_path, train) 
   
        else:
            raise Exception("Invalid data format")
        print("Data size: {}".format(len(self.data)))

        self.total_len = int(len(self.data) * factor)

        try:
            self.segment_size = config["preprocessing"]["audio"]["segment_size"]
            self.target_length = int(self.segment_size / self.hopsize)
            self.segment_length = int(self.target_length * self.hopsize)
            assert self.segment_size % self.hopsize == 0
            print("Use segment size of %s." % self.segment_size)
        except:
            self.segment_size = None
        
        if not train:
            self.mixup = 0.0
            self.freqm = 0
            self.timem = 0

        aug_cfg = config.get('augmentation', {}).get('wave', {}) if isinstance(config.get('augmentation', {}), dict) else {}
        self.wave_aug = {
            'enable': bool(aug_cfg.get('enable', True if train else False)),
            'p_apply': float(aug_cfg.get('p_apply', 0.5)),  # 50% no augmentation
            # common gain for all stems (Remix/RX)
            'rx_prob': float(aug_cfg.get('rx_prob', 0.3)),
            'rx_gain_db': tuple(aug_cfg.get('rx_gain_db', (-20.0, 0.0))),
            # independent pitch shift for each stem (keep length unchanged)
            'ps_prob': float(aug_cfg.get('ps_prob', 0.3)),
            'ps_semitones': tuple(aug_cfg.get('ps_semitones', (-3, 3))),  # integer semitone
            # independent saturation (tanh) for each stem
            'st_prob': float(aug_cfg.get('st_prob', 0.3)),
            'st_beta': tuple(aug_cfg.get('st_beta', (1.0, 5.0))),
            # slight phase/polarity flip
            'polarity_prob': float(aug_cfg.get('polarity_prob', 0.1)),
        }

        self.return_all_wav = False
        if self.mixup > 0:
            self.tempo_map = np.load(config["path"]["tempo_map"], allow_pickle=True).item()
            self.tempo_folder = config["path"]["tempo_data"]
        
        if self.mixup > 1:
            self.return_all_wav = config["augmentation"]["return_all_wav"] 

        print("Use mixup rate of %s; Use SpecAug (T,F) of (%s, %s); Use blurring effect or not %s" % (self.mixup, self.timem, self.freqm, self.use_blur))

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

        print(f'| Audiostock Dataset Length:{len(self.data)} | Epoch Length: {self.total_len}')

    def read_datafile(self, dataset_path, label_path, train):
        data = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
        if (not train) and len(data) > 2000:
            data_dict = {}
            filelist = [os.path.basename(f).split('.')[0].split('_') for f in self.data]
            for f,idx in filelist:
                if f not in data_dict:
                    data_dict[f] = int(idx)
                else:
                    data_dict[f] = max(int(idx), data_dict[f])
            data = [os.path.join(dataset_path, f'{k}_{data_dict[k] // 2}.wav') for k in data_dict.keys()] + \
                [os.path.join(dataset_path, f'{k}_0.wav') for k in data_dict.keys()] + \
                [os.path.join(dataset_path, f'{k}_{data_dict[k]}.wav') for k in data_dict.keys()]

        
        self.label = []
        if label_path is not None:
            for d in data:
                lp = os.path.join(label_path, os.path.basename(d).split('.')[0] + '.json')
                assert os.path.exists(lp), f'the label file {lp} does not exists.'
                self.label.append(lp)   

        return data 
                
    def random_segment_wav(self, x):
        wav_len = x.shape[-1]
        assert wav_len > 100, "Waveform is too short, %s" % wav_len
        if self.whole_track:
            return x
        if wav_len - self.segment_length > 0:
            if self.train:
                sta = random.randint(0, wav_len -self.segment_length)
            else:
                sta = (wav_len - self.segment_length) // 2
            x = x[:, sta: sta + self.segment_length]
        return x
    
    def normalize_wav(self, x):
        x = x[0]
        x = x - x.mean()
        x = x / (torch.max(x.abs()) + 1e-8)
        x = x * 0.5
        x = x.unsqueeze(0)
        return x

    def read_wav(self, filename):
        y, sr = torchaudio.load(filename)
        # resample
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, sr, self.sampling_rate)
        # normalize
        y = self.normalize_wav(y)
        # segment
        y = self.random_segment_wav(y)
        # pad
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y

    def get_mel(self, filename, mix_filename = None):
        # mixup
        if mix_filename is None:
            y = self.read_wav(filename)
        else:
            # get name 
            anchor_name = os.path.basename(filename)
            target_name = os.path.basename(mix_filename)
            # load wav
            anchor_wav, asr = torchaudio.load(filename)
            target_wav, tsr = torchaudio.load(mix_filename)
            assert asr == tsr, f'mixup sample rate should be the same {asr} vs. {tsr}'
            # get downbeat
            anchor_downbeat = np.load(os.path.join(self.tempo_folder, f'{anchor_name.split(".")[0]}_downbeat_pred.npy'), allow_pickle=True)
            target_downbeat = np.load(os.path.join(self.tempo_folder, f'{target_name.split(".")[0]}_downbeat_pred.npy'), allow_pickle=True)
            
            if len(anchor_downbeat) > 1 and len(target_downbeat) > 1:
                adp = int(anchor_downbeat[np.random.randint(0, len(anchor_downbeat) - 1)] * asr) 
                tdp = int(target_downbeat[np.random.randint(0, len(target_downbeat) - 1)] * tsr)
                anchor_wav = anchor_wav[..., adp:]
                target_wav = target_wav[..., tdp:]
                mix_len = min(anchor_wav.size(-1), target_wav.size(-1))
                if mix_len <= 100:
                    mix_wav, _ = torchaudio.load(filename)
                    anchor_wav = mix_wav[::]
                    target_wav = mix_wav[::]
                else:
                    anchor_wav = anchor_wav[..., :mix_len]
                    target_wav = target_wav[..., :mix_len]
                    p = np.random.beta(5,5)
                    mix_wav = p * anchor_wav + (1-p) * target_wav
            else:
                mix_wav = anchor_wav
                # normalize
            if self.return_all_wav:
                anchor_wav = self.normalize_wav(anchor_wav)
                target_wav = self.normalize_wav(target_wav)
                anchor_wav = anchor_wav[..., :self.segment_length]
                target_wav = target_wav[..., :self.segment_length]
                anchor_wav = torch.nn.functional.pad(anchor_wav, (0, self.segment_length - anchor_wav.size(1)), 'constant', 0.)
                target_wav = torch.nn.functional.pad(target_wav, (0, self.segment_length - target_wav.size(1)), 'constant', 0.)
                # get mel
                anchor_melspec, _, _ = self.STFT.mel_spectrogram(anchor_wav)
                anchor_melspec = anchor_melspec[0].T
                target_melspec, _, _ = self.STFT.mel_spectrogram(target_wav)
                target_melspec = target_melspec[0].T

                if anchor_melspec.size(0) < self.target_length:
                    anchor_melspec = torch.nn.functional.pad(anchor_melspec, (0,0,0,self.target_length - anchor_melspec.size(0)), 'constant', 0.)
                else:
                    anchor_melspec = anchor_melspec[0: self.target_length, :]
                
                if anchor_melspec.size(-1) % 2 != 0:
                    anchor_melspec = anchor_melspec[:, :-1]
                
                if target_melspec.size(0) < self.target_length:
                    target_melspec = torch.nn.functional.pad(target_melspec, (0,0,0,self.target_length - target_melspec.size(0)), 'constant', 0.)
                else:
                    target_melspec = target_melspec[0: self.target_length, :]

                if target_melspec.size(-1) % 2 != 0:
                    target_melspec = target_melspec[:, :-1]
                
                mix_wav, _ = torchaudio.load(filename) # unmix one for latent mixup

            y = self.normalize_wav(mix_wav)
            y = self.random_segment_wav(y)
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        
        # get mel
        y.requires_grad=False
        melspec, _, _ = self.STFT.mel_spectrogram(y)
        melspec = melspec[0].T
        if melspec.size(0) < self.target_length:
            melspec = torch.nn.functional.pad(melspec, (0,0,0,self.target_length - melspec.size(0)), 'constant', 0.)
        else:
            if not self.whole_track:
                melspec = melspec[0: self.target_length, :]
        if melspec.size(-1) % 2 != 0:
            melspec = melspec[:, :-1]
        
        if self.return_all_wav:
            if mix_filename is None:
                anchor_melspec = melspec
                target_melspec = melspec

            return y[0].numpy(), melspec.numpy(), anchor_melspec.numpy(), target_melspec.numpy()
        else:
            return y[0].numpy(), melspec.numpy()


    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]
        lf = self.label[idx] if len(self.label) > 0 else None
        # mixup
        if random.random() < self.mixup:
            wav_folder = os.path.dirname(f)
            anchor_name = os.path.basename(f)
            tempo_group = self.tempo_map['tempo'][self.tempo_map['map'][anchor_name]]
            if len(tempo_group) <= 1:
                mix_f = None
            else:
                mix_f = np.random.choice(tempo_group)
                mix_f = os.path.join(wav_folder, mix_f)
        else:
            mix_f = None
        # get data
        if self.return_all_wav:
            waveform, fbank,fbank_1, fbank_2 = self.get_mel(f, mix_f)
            data_dict['fbank_1'] = fbank_1
            data_dict['fbank_2'] = fbank_2
        else:
            waveform, fbank = self.get_mel(f, mix_f)
        if lf is not None:
            with open(lf, 'r') as lff:
                label_data = json.load(lff)
                text = label_data['text'][0]
        else:
            text = ""
        # construct dict
        data_dict['fname'] = os.path.basename(f).split('.')[0]
        data_dict['fbank'] = fbank
        data_dict['waveform'] = waveform
        data_dict['text'] = text


        ### adding this just to make it artificially compatible with multicanel
        audio_list = []
        fbank_list = []
        for stem in self.config["path"]["stems"]:
            audio_list.append(np.zeros_like(waveform)[np.newaxis, :])  # Expand dims for audio
            fbank_list.append(np.zeros_like(fbank)[np.newaxis, :])  # Expand dims for fbank

        
        # construct dict
        data_dict['fbank_stems'] = np.concatenate(fbank_list, axis=0)
        data_dict['waveform_stems'] = np.concatenate(audio_list, axis=0)

        return data_dict

    def __len__(self):
        return self.total_len



class DS_10283_2325_Dataset(AudiostockDataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__(dataset_path, label_path, config, train = train, factor = factor, whole_track = whole_track)  

    def read_datafile(self, dataset_path, label_path, train):
        file_path = dataset_path
        # Open the file and read lines
        with open(file_path, "r") as fp:
            data_json = json.load(fp)
            data = data_json["data"]

        dataset_directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        # dataset_name = "train" #filename.split('-')[1]

        wav_directory = os.path.abspath(os.path.join(dataset_directory, "wav_files"))

        for entry in data:

            # get audio data path
            prompt_file_path = os.path.join(wav_directory, entry["audio_prompt"])
            response_file_path = os.path.join(wav_directory, entry["audio_response"])
            entry['wav'] = prompt_file_path
            entry["response"] = response_file_path

        # self.data = data
        self.label = data
        return data

    def read_wav(self, filename, frame_offset):

        audio_data, sr =torchaudio.load(filename, frame_offset =  frame_offset*48000, num_frames = 480000)

        # resample
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(audio_data, sr, self.sampling_rate)
        # normalize
        y = self.normalize_wav(y)
        # segment
        y = self.random_segment_wav(y)
        # pad
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y
    
    def get_mel(self, filename, mix_filename = None, frame_offset = 0):
        # mixup
        y = self.read_wav(filename, frame_offset)
        
        # get mel
        y.requires_grad=False
        melspec, _, _ = self.STFT.mel_spectrogram(y)
        melspec = melspec[0].T
        if melspec.size(0) < self.target_length:
            melspec = torch.nn.functional.pad(melspec, (0,0,0,self.target_length - melspec.size(0)), 'constant', 0.)
        else:
            if not self.whole_track:
                melspec = melspec[0: self.target_length, :]
        if melspec.size(-1) % 2 != 0:
            melspec = melspec[:, :-1]
        
        if self.return_all_wav:
            if mix_filename is None:
                anchor_melspec = melspec
                target_melspec = melspec

            return y[0].numpy(), melspec.numpy(), anchor_melspec.numpy(), target_melspec.numpy()
        else:
            return y[0].numpy(), melspec.numpy()
            
    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]
        lf = self.label[idx]
        if lf is not None:
            text = lf['text']
        else:
            text = ""        
        
        
        prompt, fbank_prompt = self.get_mel(f["wav"], None, f["frame_offset"])
        response, fbank_response = self.get_mel(f["response"], None, f["frame_offset"])

        # construct dict
        data_dict['fname'] = os.path.basename(lf['text']).split('.')[0]+"_from_"+str(f["frame_offset"])
        data_dict['fbank_prompt'] = fbank_prompt
        data_dict['prompt'] = prompt
        data_dict['text'] = text

        data_dict['fbank'] = fbank_response
        data_dict['waveform'] = response

        return data_dict



class Audiostock_splited_Dataset(DS_10283_2325_Dataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__(dataset_path, label_path, config, train = train, factor = factor, whole_track = whole_track)  

    def read_datafile(self, dataset_path, label_path, train):
        file_path = dataset_path
        # Open the file and read lines
        data = []
        with open(file_path, "r") as fp:
            data_json = json.load(fp)

        for key, inner_dict in data_json.items():
            new_dict = inner_dict.copy()  # Create a copy of the inner dictionary
            new_dict['id'] = key  # Add the key from the outer dictionary
            data.append(new_dict)        

        prompt = self.config["path"]["prompt"]
        response = self.config["path"]["response"]

        # here we need logic taht only leaves in data entries that have nothe prompr and reposonse


        dataset_directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        dataset_name = filename.split('_')[0]

        wav_directory = os.path.abspath(os.path.join(dataset_directory, dataset_name+"_splited_16khz"))

        # Filter out entries where prompt or response is 0
        data = [entry for entry in data if entry[prompt+".wav"] != 0 and entry[response+".wav"] != 0]

        # iterate to get wav directories and append lable info
        for entry in data:
            prompt_file_path = os.path.join(wav_directory, entry["id"], prompt+".wav")

            response_file_path = os.path.join(wav_directory, entry["id"], response+".wav")
            entry['prompt'] = prompt_file_path
            entry["response"] = response_file_path

            if label_path is not None:
                lp = os.path.join(label_path, entry["id"] + '.json')
                assert os.path.exists(lp), f'the label file {lp} does not exists.'
                with open(lp, "r") as fp:
                    label_json = json.load(fp)
                entry.update(label_json)   

        new_data = []
        entries_to_remove = []  # List to store entries to be removed
        for entry in data:
            entry['frame_offset'] = 0
            # cut long files and take 10 seconds. first 30 seconds only if available
            duration = entry['original_data']["audio_size"]
            if duration > 10:  
                num_copies = int((min(duration,600)-10) / 10)
                for i in range(num_copies):
                    new_entry = entry.copy()
                    new_entry['frame_offset'] = (i+1) * 10
                    new_data.append(new_entry)

            # find very short files
            if duration < 0.2:  
                entries_to_remove.append(entry)
        
        # Remove the entries from data
        for entry in entries_to_remove:
            data.remove(entry)

        # add new entries
        data.extend(new_data)

        return data


    def read_wav(self, filename, frame_offset):

        y, _ =torchaudio.load(filename, frame_offset =  frame_offset*16000, num_frames = 160000)

        # normalize
        y = self.normalize_wav(y)
        # segment
        y = self.random_segment_wav(y)
        # pad
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y

    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]
       
        text = f['text']
        
        prompt, fbank_prompt = self.get_mel(f["prompt"], None, f["frame_offset"])
        response, fbank_response = self.get_mel(f["response"], None, f["frame_offset"])

        # construct dict
        data_dict['fname'] = os.path.basename(f['id']).split('.')[0]+"_from_"+str(f["frame_offset"])
        data_dict['fbank_prompt'] = fbank_prompt
        data_dict['prompt'] = prompt
        data_dict['text'] = text

        data_dict['fbank'] = fbank_response
        data_dict['waveform'] = response

        return data_dict


class Slakh_Dataset(DS_10283_2325_Dataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__(dataset_path, label_path, config, train = train, factor = factor, whole_track = whole_track)  

    def read_datafile(self, dataset_path, label_path, train):

        data = []

        # Iterate over entries in the dataset
        for entry in os.listdir(dataset_path):
            entry_path = os.path.join(dataset_path, entry)
            
            # Check if metadata.yaml file exists
            lp = os.path.join(entry_path, 'metadata_updated.yaml')
            if os.path.exists(lp):
                pass
            else:
                continue

            # Read and load the YAML file
            with open(lp, "r") as fp:
                label_yaml = yaml.safe_load(fp)
            
            # Append the loaded data to the list
            data.append(label_yaml)
        
        filtered_data = []

        prompt = self.config["path"]["prompt"]
        response = self.config["path"]["response"]

        # Create pairs of prompts and responses
        for entry in data:

            prompts = []
            responses = []

            wav_directory = os.path.join(dataset_path, entry['audio_dir'])

            # Collect all prompts and responses
            for name, stem in entry['stems'].items():
                file_path = os.path.join(wav_directory, name + ".flac")
                
                if os.path.exists(file_path):
                    if stem['inst_class'] == prompt:
                        prompts.append({'path': file_path, 'duration': stem["duration"], "active_segments": stem["active_segments"]})

                    elif stem['inst_class'] == response:
                        responses.append({'path': file_path, 'duration': stem["duration"], "active_segments": stem["active_segments"]})
                else:
                    continue
               
            # Pair each prompt with each response
            for prompt_entry in prompts:
                for response_entry in responses:
                    
                    # Compare active segments
                    prompt_segments = set(prompt_entry['active_segments'])
                    response_segments = set(response_entry['active_segments'])
                    shared_segments = sorted(prompt_segments.intersection(response_segments))
                    
                    # Create a new entry for each shared segment
                    if shared_segments:
                        for segment in shared_segments:
                            new_entry = entry.copy()
                            
                            new_entry['prompt'] = prompt_entry['path']
                            new_entry['response'] = response_entry['path']
                            new_entry['frame_offset'] = segment
                            filtered_data.append(new_entry)
                    else:
                        pass

        return filtered_data


    def read_wav(self, filename, frame_offset):

        y, sr =torchaudio.load(filename, frame_offset =  frame_offset*44100, num_frames = 441000)

        # resample
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, sr, self.sampling_rate)

        # normalize
        y = self.normalize_wav(y)
        # segment
        y = self.random_segment_wav(y)
        # pad
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y

    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]
        
        prompt, fbank_prompt = self.get_mel(f["prompt"], None, int(f["frame_offset"]))
        response, fbank_response = self.get_mel(f["response"], None, int(f["frame_offset"]))

        # construct dict
        data_dict['fname'] = f['audio_dir'].split('/')[0]+"_from_"+str(f["frame_offset"])
        data_dict['fbank_prompt'] = fbank_prompt
        data_dict['prompt'] = prompt
        # data_dict['text'] = text

        data_dict['fbank'] = fbank_response
        data_dict['waveform'] = response

        return data_dict


class MultiSource_Slakh_Dataset(DS_10283_2325_Dataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__(dataset_path, label_path, config, train = train, factor = factor, whole_track = whole_track)  

        self.text_prompt = config.get('path', {}).get('text_prompt', None)
        self.stem_masking = config.get('augmentation', {}).get('masking', False)

        mapping_path = config.get('model', {}).get('params', {}).get('track_mapping_path', '/PATH/TO/PROJECT/data/StemGMD_org/track_mapping.csv')
        if os.path.exists(mapping_path):
            self.trackid2json = self._build_trackid2json(mapping_path)
        else:
            print(f"Warning: track_mapping.csv does not exist at {mapping_path}")
            self.trackid2json = {}

    def get_duration_sec(self, file, cache=False):

        if not os.path.exists(file):
            # File doesn't exist, return 0
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
        # Remove files too short or too long
        keep = []
        durations = []
        for track in tracks:
            track_dir = os.path.join(audio_files_dir, track)
            files = [os.path.join(track_dir, stem + ".wav") for stem in self.config["path"]["stems"]]
            # only count the existing files
            exist_files = [f for f in files if os.path.exists(f)]
            
            # special handling: if the expected stems are not found, but drums.wav is found, also keep it
            if not exist_files:
                drums_file = os.path.join(track_dir, "drums.wav")
                if os.path.exists(drums_file):
                    exist_files = [drums_file]
                else:
                    continue  # this track all stems do not exist, skip
            
            durations_track = np.array([self.get_duration_sec(file, cache=True) * self.config['preprocessing']['audio']['sampling_rate'] for file in exist_files])
            # use the duration of the first existing stem
            duration = durations_track[0]
            # only keep if there is at least one stem with a suitable length
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
                'wav_path': os.path.join(dataset_path,keep[idx]),
                'duration': durations[idx],
                # 'cumsum': cumsum[idx]
            }
            # Append the dictionary to the data list
            data.append(track_info)

        entries_to_remove = []  # List to store entries to be removed
        max_samples = 640.0 * self.config['preprocessing']['audio']['sampling_rate']

        # Temporary list to hold all data including new segments
        temp_data = []

        for entry in data:
            entry['frame_offset'] = 0
            duration = entry['duration']

            # Always add the original entry to temp_data
            temp_data.append(entry)

            # Handle long files by adding new segments immediately after the original entry
            if duration > self.segment_length:
                num_copies = int((min(duration, max_samples) - self.segment_length) / self.segment_length)
                for i in range(num_copies):
                    new_entry = entry.copy()
                    new_entry['frame_offset'] = (i + 1) * self.segment_length
                    temp_data.append(new_entry)  # Add new segment right after the original entry

            # Mark very short files for removal
            if duration < 0.2:
                entries_to_remove.append(entry)

        # Remove the short entries directly from temp_data
        temp_data = [entry for entry in temp_data if entry not in entries_to_remove]

        # Now, temp_data has all the data in the desired order
        data = temp_data

        return data


    def read_wav(self, filename, frame_offset):

        # y, sr =torchaudio.load(filename, frame_offset =  int (frame_offset*44100), num_frames = self.segment_length) # use the segment_length in the config
        y, sr =torchaudio.load(filename, frame_offset =  int (frame_offset*44100), num_frames = int(44100*10.24))
        # keep the original number of channels, Stable Audio supports dual-channel input
        # if stereo, keep dual-channel; if mono, keep mono

        # resample
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, sr, self.sampling_rate)

        # pad
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y

    def get_index_offset(self, item):

        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.segment_length // 2
        shift = np.random.randint(-half_interval, half_interval) if self.train else 0
        offset = item["frame_offset"] + shift  # Note we centred shifts, so adding now
        
        start, end = 0.0, item["duration"]  # start and end of current song

        if offset > end - self.segment_length:  # Going over song
            offset = max(start, offset - half_interval)  # Now should fit
        if offset < start:  # Going under zero
            offset = 0.0 # Now should fit
        
        offset = offset/self.config['preprocessing']['audio']['sampling_rate']
        return item, offset

    def get_mel_from_waveform(self, waveform):
        # waveform
        y = torch.tensor(waveform).unsqueeze(0) #self.read_wav(filename, frame_offset)
        
        # get mel
        y.requires_grad=False
        melspec, _, _ = self.STFT.mel_spectrogram(y)
        melspec = melspec[0].T
        if melspec.size(0) < self.target_length:
            melspec = torch.nn.functional.pad(melspec, (0,0,0,self.target_length - melspec.size(0)), 'constant', 0.)
        else:
            if not self.whole_track:
                melspec = melspec[0: self.target_length, :]
        if melspec.size(-1) % 2 != 0:
            melspec = melspec[:, :-1]

        return melspec.numpy()

    def mask_audio_channels(self, audio, fbank):
        """
        Randomly masks 0, 1, 2, or 3 channels in a 4-channel audio input and updates the corresponding Mel spectrograms.
        
        Parameters:
        audio (list of np.ndarray): 4-channel audio input, where each sublist represents a channel.
        fbank (list): List to store the Mel spectrograms corresponding to the masked audio.
        
        Returns:
        tuple: (masked_audio, fbank)
            masked_audio (list of np.ndarray): Audio input with randomly masked channels.
            fbank (list of np.ndarray): Updated Mel spectrograms corresponding to the masked audio.
        """
        num_channels = len(audio)
        assert num_channels == 4, "Audio input must have 4 channels."
        
        # Determine the number of channels to mask (0, 1, 2, or 3)
        num_channels_to_mask = random.choice(range(num_channels))
        
        # Select the channels to mask
        channels_to_mask = random.sample(range(num_channels), num_channels_to_mask)
        
        # Create a copy of the audio list to avoid modifying the original input
        masked_audio = [channel.copy() for channel in audio]
        
        # Apply the mask to the selected channels
        for channel in channels_to_mask:
            masked_audio[channel] = np.zeros_like(masked_audio[channel])
        
        # Update the Mel spectrograms in the fbank for the masked channels
        for channel in channels_to_mask:
            fbank[channel] = np.expand_dims(self.get_mel_from_waveform(masked_audio[channel][0]), axis=0)
        
        return masked_audio, fbank

    def _build_trackid2json(self, mapping_path):
        df = pd.read_csv(mapping_path)
        trackid2json = {}
        for _, row in df.iterrows():
            track_id = row['track_id']
            midi_json = os.path.basename(row['stems_dir'])
            trackid2json[track_id] = midi_json
        return trackid2json

    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]

        index, frame_offset = self.get_index_offset(f)

        stems_list = self.config["path"]["stems"]
        # check if it is 5 stems mode
        if stems_list == ['kick', 'snare', 'toms', 'hi_hats', 'cymbals']:
            # first read all 9 stems
            all_stem_names = ['crash', 'hh_closed', 'hh_open', 'hi_tom', 'kick', 'low_tom', 'mid_tom', 'ride', 'snare']
            all_audio = {}
            all_fbank = {}
            first_valid_audio = None
            first_valid_fbank = None
            for stem in all_stem_names:
                stem_path = os.path.join(f["wav_path"], stem + ".wav")
                if os.path.exists(stem_path):
                    audio, fbank = self.get_mel(stem_path, None, frame_offset)
                    if first_valid_audio is None:
                        first_valid_audio = audio
                    if first_valid_fbank is None:
                        first_valid_fbank = fbank
                else:
                    audio = np.zeros_like(first_valid_audio) if first_valid_audio is not None else None
                    fbank = np.zeros_like(first_valid_fbank) if first_valid_fbank is not None else None
                all_audio[stem] = audio
                all_fbank[stem] = fbank
            # merge into 5 stems
            audio_list = [
                all_audio['kick'][np.newaxis, :] if all_audio['kick'] is not None else np.zeros_like(first_valid_audio)[np.newaxis, :],
                all_audio['snare'][np.newaxis, :] if all_audio['snare'] is not None else np.zeros_like(first_valid_audio)[np.newaxis, :],
                ( (all_audio['hi_tom'] if all_audio['hi_tom'] is not None else 0)
                + (all_audio['mid_tom'] if all_audio['mid_tom'] is not None else 0)
                + (all_audio['low_tom'] if all_audio['low_tom'] is not None else 0) )[np.newaxis, :],
                ( (all_audio['hh_closed'] if all_audio['hh_closed'] is not None else 0)
                + (all_audio['hh_open'] if all_audio['hh_open'] is not None else 0) )[np.newaxis, :],
                ( (all_audio['crash'] if all_audio['crash'] is not None else 0)
                + (all_audio['ride'] if all_audio['ride'] is not None else 0) )[np.newaxis, :],
            ]
            # merge waveform first then convert to mel
            fbank_list = [
                np.expand_dims(self.get_mel_from_waveform(all_audio['kick']), axis=0) if all_audio['kick'] is not None else np.zeros_like(first_valid_fbank)[np.newaxis, :],
                np.expand_dims(self.get_mel_from_waveform(all_audio['snare']), axis=0) if all_audio['snare'] is not None else np.zeros_like(first_valid_fbank)[np.newaxis, :],
                np.expand_dims(self.get_mel_from_waveform(
                    (all_audio['hi_tom'] if all_audio['hi_tom'] is not None else 0)
                    + (all_audio['mid_tom'] if all_audio['mid_tom'] is not None else 0)
                    + (all_audio['low_tom'] if all_audio['low_tom'] is not None else 0)
                ), axis=0),
                np.expand_dims(self.get_mel_from_waveform(
                    (all_audio['hh_closed'] if all_audio['hh_closed'] is not None else 0)
                    + (all_audio['hh_open'] if all_audio['hh_open'] is not None else 0)
                ), axis=0),
                np.expand_dims(self.get_mel_from_waveform(
                    (all_audio['crash'] if all_audio['crash'] is not None else 0)
                    + (all_audio['ride'] if all_audio['ride'] is not None else 0)
                ), axis=0),
            ]
        else:
            # original 9 stems logic
            audio_list = []
            fbank_list = []
            first_valid_audio = None
            first_valid_fbank = None
            stems_found = 0
            stems_total = len(self.config["path"]["stems"])
            for stem in self.config["path"]["stems"]:
                stem_path = os.path.join(f["wav_path"], stem + ".wav")
                if os.path.exists(stem_path):
                    audio, fbank = self.get_mel(stem_path, None, frame_offset)
                    if first_valid_audio is None:
                        first_valid_audio = audio
                    if first_valid_fbank is None:
                        first_valid_fbank = fbank
                    stems_found += 1
                else:
                    if first_valid_audio is not None and first_valid_fbank is not None:
                        audio = np.zeros_like(first_valid_audio)
                        fbank = np.zeros_like(first_valid_fbank)
                    else:
                        audio = None
                        fbank = None
                if audio is not None and fbank is not None:
                    audio_list.append(audio[np.newaxis, :])
                    fbank_list.append(fbank[np.newaxis, :])
            while len(audio_list) < stems_total:
                if first_valid_audio is not None and first_valid_fbank is not None:
                    audio_list.append(np.zeros_like(first_valid_audio)[np.newaxis, :])
                    fbank_list.append(np.zeros_like(first_valid_fbank)[np.newaxis, :])
                else:
                    raise RuntimeError(f"All stems missing for sample: {f['wav_path']}")

        if self.stem_masking and self.train:
            audio_list, fbank_list = self.mask_audio_channels(audio_list, fbank_list)

        # construct dict
        data_dict['fname'] = f['wav_path'].split('/')[-1]+"_from_"+str(int(frame_offset))
        data_dict['fbank_stems'] = np.concatenate(fbank_list, axis=0)
        data_dict['waveform_stems'] = np.concatenate(audio_list, axis=0)

        data_dict['waveform'] = np.clip(np.sum(data_dict['waveform_stems'], axis=0), -1, 1)
        data_dict['fbank'] = self.get_mel_from_waveform(data_dict['waveform'])

        # ====== add onset_pianoroll ======
        # 1. get track id
        track_id = data_dict['fname'].split('_')[0]  # e.g. Track00001
        # 2. find corresponding json file
        json_file = self.trackid2json.get(track_id, None)
        # use the path in the config, if not, use the default path
        midi_onset_dir = getattr(self, 'onset_data_path', '/PATH/TO/PROJECT/midi_onset_gt')
        stem_names = self.config['path']['stems']  # ['kick', 'snare', ...]
        num_stems = len(stem_names)
        target_length = self.config['preprocessing']['mel']['target_length']
        hop_length = self.config['preprocessing']['stft']['hop_length']
        sampling_rate = self.config['preprocessing']['audio']['sampling_rate']

        pianoroll = np.zeros((num_stems, target_length), dtype=np.float32)
        if json_file is not None:
            json_path = os.path.join(midi_onset_dir, json_file)
            if os.path.exists(json_path):
                with open(json_path, 'r') as f_json:
                    onset_dict = json.load(f_json)
                # 3. convert the onset list of each stem to pianoroll
                for i, stem in enumerate(stem_names):
                    json_keys = STEM_JSON_MAP.get(stem, [])
                    for key in json_keys:
                        if key in onset_dict:
                            for onset_sec in onset_dict[key]:
                                frame = int(float(onset_sec) * sampling_rate / hop_length)
                                if 0 <= frame < target_length:
                                    pianoroll[i, frame] = 1.0
        data_dict['onset_pianoroll'] = pianoroll
        # ==================================


        if self.text_prompt is not None:
           data_dict["text"] = self.text_prompt

        return data_dict


class MultiSource_Slakh_Inference_Dataset(MultiSource_Slakh_Dataset):
    """
    MultiSource_Slakh_Dataset for inference, without ground truth stems
    """
    
    def __init__(self, dataset_path, label_path, config, train=True, factor=1.0, whole_track=False) -> None:
        super().__init__(dataset_path, label_path, config, train=train, factor=factor, whole_track=whole_track)
        print(f"[Inference Dataset] initialize inference dataset, {len(self.data)} samples")
    
    def __getitem__(self, index):
        """
        inference version: only return mix and fbank, without ground truth stems
        """
        data_dict = {}
        
        # get original stems and generate mix
        audio_list = []
        fbank_list = []
        
        for stem in self.config["path"]["stems"]:
            stem_path = os.path.join(self.data[index]["audio_path"], stem + ".wav")
            
            if os.path.exists(stem_path):
                # read stem
                wav = self.read_wav(stem_path, self.data[index]["offset"])
                mel = self.get_mel_from_waveform(wav)
                
                audio_list.append(wav)
                fbank_list.append(mel)
            else:
                # if stem does not exist, use zero padding
                wav = torch.zeros(1, self.segment_length)
                mel = torch.zeros(1, self.target_length, self.melbins)
                
                audio_list.append(wav)
                fbank_list.append(mel)
        
        # merge all stems
        data_dict['waveform_stems'] = np.concatenate(audio_list, axis=0)  # [num_stems, T]
        data_dict['fbank_stems'] = np.concatenate(fbank_list, axis=0)    # [num_stems, F, L]
        
        # generate mix (sum of all stems)
        data_dict['waveform'] = np.clip(np.sum(data_dict['waveform_stems'], axis=0), -1, 1)  # [T]
        data_dict['fbank'] = self.get_mel_from_waveform(data_dict['waveform'])  # [F, L]
        
        # basic information
        data_dict['fname'] = self.data[index]["fname"]
        data_dict['text'] = self.data[index].get("text", "")
        
        # inference does not need ground truth stems, so remove these keys
        # only keep mix related data for model use
        
        return data_dict


class Dataset(Dataset):
    def __init__(
        self,
        preprocess_config=None,
        train_config=None,
        samples_weight=None,
        train=True,
        shuffle=None,
    ):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.preprocess_config = preprocess_config
        self.train_config = train_config
        self.datapath = (
            preprocess_config["path"]["train_data"]
            if (train)
            else preprocess_config["path"]["test_data"]
        )

        self.data = []
        if type(self.datapath) is str:
            with open(self.datapath, "r") as fp:
                data_json = json.load(fp)
            self.data = data_json["data"]
        elif type(self.datapath) is list:
            for datapath in self.datapath:
                with open(datapath, "r") as fp:
                    data_json = json.load(fp)
                self.data += data_json["data"]
        else:
            raise Exception("Invalid data format")
        print("Data size: {}".format(len(self.data)))

        self.samples_weight = samples_weight

        self.melbins = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.freqm = preprocess_config["preprocessing"]["mel"]["freqm"]
        self.timem = preprocess_config["preprocessing"]["mel"]["timem"]
        self.mixup = train_config["augmentation"]["mixup"]

        # No augmentation during evaluation
        if train == False:
            self.mixup = 0.0
            self.freqm = 0
            self.timem = 0

        self.sampling_rate = preprocess_config["preprocessing"]["audio"][
            "sampling_rate"
        ]

        self.hopsize = self.preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.target_length = self.preprocess_config["preprocessing"]["mel"][
            "target_length"
        ]
        self.use_blur = self.preprocess_config["preprocessing"]["mel"]["blur"]

        self.segment_length = int(self.target_length * self.hopsize)

        try:
            self.segment_size = self.preprocess_config["preprocessing"]["audio"][
                "segment_size"
            ]
            self.target_length = int(self.segment_size / self.hopsize)
            assert self.segment_size % self.hopsize == 0
            print("Use segment size of %s." % self.segment_size)
        except:
            self.segment_size = None


        print(
            "Use mixup rate of %s; Use SpecAug (T,F) of (%s, %s); Use blurring effect or not %s"
            % (self.mixup, self.timem, self.freqm, self.use_blur)
        )

        self.skip_norm = False
        self.noise = False
        if self.noise == True:
            print("now use noise augmentation")

        self.index_dict = make_index_dict(
            preprocess_config["path"]["class_label_index"]
        )
        self.label_num = len(self.index_dict)
        print("number of classes is {:d}".format(self.label_num))
        self.STFT = Audio.stft.TacotronSTFT(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )

        self.id2label, self.id2num, self.num2label = self.build_id_to_label()

    def build_id_to_label(self):
        ret = {}
        id2num = {}
        num2label = {}
        df = pd.read_csv(self.preprocess_config["path"]["class_label_index"])
        for _, row in df.iterrows():
            index, mid, display_name = row["index"], row["mid"], row["display_name"]
            ret[mid] = display_name
            id2num[mid] = index
            num2label[index] = display_name
        return ret, id2num, num2label

    def resample(self, waveform, sr):
        if sr == 16000:
            return waveform
        if sr == 32000 and self.sampling_rate == 16000:
            waveform = waveform[::2]
            return waveform
        if sr == 48000 and self.sampling_rate == 16000:
            waveform = waveform[::3]
            return waveform
        else:
            raise ValueError(
                "We currently only support 16k audio generation. You need to resample you audio file to 16k, 32k, or 48k: %s, %s"
                % (sr, self.sampling_rate)
            )

    def normalize_wav(self, waveform):
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform * 0.5

    def random_segment_wav(self, waveform):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        # Too short
        if (waveform_length - self.segment_length) <= 0:
            return waveform

        random_start = int(
            self.random_uniform(0, waveform_length - self.segment_length)
        )
        return waveform[:, random_start : random_start + self.segment_length]

    def pad_wav(self, waveform):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        if waveform_length == self.segment_length:
            return waveform

        # Pad
        temp_wav = np.zeros((1, self.segment_length))
        rand_start = int(self.random_uniform(0, self.segment_length - waveform_length))

        temp_wav[:, rand_start : rand_start + waveform_length] = waveform
        return temp_wav

    def trim_wav(self, waveform):
        if np.max(np.abs(waveform)) < 0.0001:
            return waveform

        def detect_leading_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = 0
            while start + chunk_size < waveform_length:
                if np.max(np.abs(waveform[start : start + chunk_size])) < threshold:
                    start += chunk_size
                else:
                    break
            return start

        def detect_ending_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = waveform_length
            while start - chunk_size > 0:
                if np.max(np.abs(waveform[start - chunk_size : start])) < threshold:
                    start -= chunk_size
                else:
                    break
            if start == waveform_length:
                return start
            else:
                return start + chunk_size

        start = detect_leading_silence(waveform)
        end = detect_ending_silence(waveform)

        return waveform[start:end]

    def read_wav_file(self, filename):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform.numpy()[0, ...]

        waveform = self.resample(waveform, sr)
        waveform = self.normalize_wav(waveform)
        waveform = self.trim_wav(waveform)

        waveform = waveform[None, ...]

        waveform = self.random_segment_wav(waveform)
        waveform = self.pad_wav(waveform)

        return waveform

    def _wav2fbank(self, filename, filename2=None):
        if filename2 == None:
            waveform = self.read_wav_file(filename)
        else:
            waveform1 = self.read_wav_file(filename)
            waveform2 = self.read_wav_file(filename2)


            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(5, 5)
            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = self.normalize_wav(mix_waveform)

        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        fbank, log_magnitudes_stft, energy = Audio.tools.get_mel_from_wav(
            waveform, self.STFT
        )

        fbank = torch.FloatTensor(fbank.T)
        log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

        fbank, log_magnitudes_stft = self._pad_spec(fbank), self._pad_spec(
            log_magnitudes_stft
        )

        if filename2 == None:
            return fbank, log_magnitudes_stft, 0, waveform
        else:
            return fbank, log_magnitudes_stft, mix_lambda, waveform

    def _pad_spec(self, fbank):
        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0 : self.target_length, :]

        if fbank.size(-1) % 2 != 0:
            fbank = fbank[..., :-1]

        return fbank

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        (
            fbank,
            log_magnitudes_stft,
            waveform,
            label_indices,
            clip_label,
            fname,
            (datum, mix_datum),
        ) = self.feature_extraction(index)

        text = self.label_indices_to_text(datum, label_indices)
        if mix_datum is not None:
            text += self.label_indices_to_text(mix_datum, label_indices)

        t_step = fbank.size(0)
        waveform = waveform[..., : int(self.hopsize * t_step)]

        return (
            fbank.float(),
            log_magnitudes_stft.float(),
            label_indices.float(),
            fname,
            waveform.float(),
            text,
        ) 

    def make_decision(self, probability):
        if float(torch.rand(1)) < probability:
            return True
        else:
            return False

    def label_indices_to_text(self, datum, label_indices):
        if "caption" in datum.keys():
            return datum["caption"]
        name_indices = torch.where(label_indices > 0.1)[0]
        description_header = ""
        labels = ""
        for id, each in enumerate(name_indices):
            if id == len(name_indices) - 1:
                labels += "%s." % self.num2label[int(each)]
            else:
                labels += "%s, " % self.num2label[int(each)]
        return description_header + labels

    def feature_extraction(self, index):
        if index > len(self.data) - 1:
            print(
                "The index of the dataloader is out of range: %s/%s"
                % (index, len(self.data))
            )
            index = random.randint(0, len(self.data) - 1)

        # Read wave file and extract feature
        while True:
            try:
                if random.random() < self.mixup:
                    datum = self.data[index]
                    ###########################################################
                    mix_sample_idx = random.randint(0, len(self.data) - 1)
                    mix_datum = self.data[mix_sample_idx]
                    ###########################################################
                    # get the mixed fbank
                    fbank, log_magnitudes_stft, mix_lambda, waveform = self._wav2fbank(
                        datum["wav"], mix_datum["wav"]
                    )
                    # initialize the label
                    label_indices = np.zeros(self.label_num)
                    for label_str in datum["labels"].split(","):
                        label_indices[int(self.index_dict[label_str])] += mix_lambda
                    for label_str in mix_datum["labels"].split(","):
                        label_indices[int(self.index_dict[label_str])] += (
                            1.0 - mix_lambda
                        )

                else:
                    datum = self.data[index]
                    label_indices = np.zeros(self.label_num)
                    fbank, log_magnitudes_stft, mix_lambda, waveform = self._wav2fbank(
                        datum["wav"]
                    )
                    for label_str in datum["labels"].split(","):
                        label_indices[int(self.index_dict[label_str])] = 1.0

                    mix_datum = None
                label_indices = torch.FloatTensor(label_indices)
                break
            except Exception as e:
                index = (index + 1) % len(self.data)
                print("feature_extraction", e)
                continue

        # The filename of the wav file
        fname = datum["wav"]

        clip_label = None

        return (
            fbank,
            log_magnitudes_stft,
            waveform,
            label_indices,
            clip_label,
            fname,
            (datum, mix_datum),
        )

    def aug(self, fbank):
        assert torch.min(fbank) < 0
        fbank = fbank.exp()
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.use_blur:
            fbank = self.blur(fbank)
        if self.freqm != 0:
            fbank = self.frequency_masking(fbank, self.freqm)
        if self.timem != 0:
            fbank = self.time_masking(fbank, self.timem)  # self.timem=0
        fbank = (fbank + 1e-7).log()
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)
        if self.noise == True:
            fbank = (
                fbank
                + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            )
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        return fbank


    def __len__(self):
        return len(self.data)

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def blur(self, fbank):
        assert torch.min(fbank) >= 0
        kernel_size = int(self.random_uniform(1, self.melbins))
        fbank = torchvision.transforms.functional.gaussian_blur(
            fbank, kernel_size=[kernel_size, kernel_size]
        )
        return fbank

    def frequency_masking(self, fbank, freqm):
        bs, freq, tsteps = fbank.size()
        mask_len = int(self.random_uniform(freqm // 8, freqm))
        mask_start = int(self.random_uniform(start=0, end=freq - mask_len))
        fbank[:, mask_start : mask_start + mask_len, :] *= 0.0
        return fbank

    def time_masking(self, fbank, timem):
        bs, freq, tsteps = fbank.size()
        mask_len = int(self.random_uniform(timem // 8, timem))
        mask_start = int(self.random_uniform(start=0, end=tsteps - mask_len))
        fbank[:, :, mask_start : mask_start + mask_len] *= 0.0
        return fbank


def balance_test():
    import torch
    from tqdm import tqdm
    from pytorch_lightning import Trainer, seed_everything

    from torch.utils.data import WeightedRandomSampler
    from torch.utils.data import DataLoader
    from utilities.data.dataset import Dataset as AudioDataset

    seed_everything(0)

    # train_json = "/mnt/fast/nobackup/users/hl01486/metadata/audioset_freesound_full/datafiles_extra_audio_files_2/audioset_bal_unbal_freesound_train_data.json"
    train_json = "/mnt/fast/nobackup/users/hl01486/metadata/audioset/datafiles/audioset_bal_unbal_train_data.json"

    samples_weight = np.loadtxt(train_json[:-5] + "_weight.csv", delimiter=",")

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    # dataset = AudioDataset(samples_weight = None, train=True)
    dataset = AudioDataset(samples_weight=samples_weight, train=True)

    loader = DataLoader(dataset, batch_size=10, num_workers=8, sampler=sampler)

    result = []

    label_indices_total = None

    for cnt, each in tqdm(enumerate(loader)):
        (
            fbank,
            log_magnitudes_stft,
            label_indices,
            fname,
            waveform,
            clip_label,
            text,
        ) = each
        if label_indices_total is None:
            label_indices_total = label_indices
        else:
            label_indices_total += label_indices

        if cnt > 2000:
            break

    np.save(
        "balanced_with_mixup_balance.npy",
        label_indices_total.cpu().detach().numpy() / 2000,
    )

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    dataset = AudioDataset(samples_weight=None, train=True)

    loader = DataLoader(dataset, batch_size=10, num_workers=8, sampler=sampler)

    result = []

    label_indices_total = None

    for cnt, each in tqdm(enumerate(loader)):
        (
            fbank,
            log_magnitudes_stft,
            label_indices,
            fname,
            waveform,
            clip_label,
            text,
        ) = each
        if label_indices_total is None:
            label_indices_total = label_indices
        else:
            label_indices_total += label_indices

        if cnt > 2000:
            break

    np.save(
        "balanced_with_no_mixup_balance.npy",
        label_indices_total.cpu().detach().numpy() / 2000,
    )

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    dataset = AudioDataset(samples_weight=None, train=True)

    loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=8,
        # sampler=sampler
    )

    result = []

    label_indices_total = None

    for cnt, each in tqdm(enumerate(loader)):
        (
            fbank,
            log_magnitudes_stft,
            label_indices,
            fname,
            waveform,
            clip_label,
            text,
        ) = each
        if label_indices_total is None:
            label_indices_total = label_indices
        else:
            label_indices_total += label_indices

        if cnt > 2000:
            break

    np.save("no_balance.npy", label_indices_total.cpu().detach().numpy() / 2000)


def check_batch(batch):
    import soundfile as sf
    import matplotlib.pyplot as plt

    save_path = "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/stable_diffusion_for_audio/output/temp"
    os.makedirs(save_path, exist_ok=True)
    fbank, log_magnitudes_stft, label_indices, fname, waveform, clip_label, text = batch
    for fb, wv, description in zip(fbank, waveform, text):
        sf.write(
            save_path + "/" + "%s.wav" % description.replace(" ", "_")[:30], wv, 16000
        )
        plt.imshow(np.flipud(fb.cpu().detach().numpy().T), aspect="auto")
        plt.savefig(save_path + "/" + "%s.png" % description.replace(" ", "_")[:30])


if __name__ == "__main__":

    import torch
    from tqdm import tqdm
    from pytorch_lightning import Trainer, seed_everything

    from torch.utils.data import WeightedRandomSampler
    from torch.utils.data import DataLoader
    from utilities.data.dataset import Dataset as AudioDataset

    seed_everything(0)

    preprocess_config = yaml.load(
        open(
            "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/stable_diffusion_for_audio/config/2023_01_06_v2_AC_F4_S_rolling_aug/preprocess.yaml",
            "r",
        ),
        Loader=yaml.FullLoader,
    )
    train_config = yaml.load(
        open(
            "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/stable_diffusion_for_audio/config/2023_01_06_v2_AC_F4_S_rolling_aug/train.yaml",
            "r",
        ),
        Loader=yaml.FullLoader,
    )

    # train_json = "/mnt/fast/nobackup/users/hl01486/metadata/audioset_freesound_full/datafiles_extra_audio_files_2/audioset_bal_unbal_freesound_train_data.json"
    train_json = "/mnt/fast/nobackup/users/hl01486/metadata/audiocaps/datafiles/audiocaps_train_label.json"

    samples_weight = np.loadtxt(train_json[:-5] + "_weight.csv", delimiter=",")

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    dataset = AudioDataset(
        samples_weight=samples_weight,
        train=True,
        train_config=train_config,
        preprocess_config=preprocess_config,
    )

    loader = DataLoader(dataset, batch_size=10, num_workers=8, sampler=sampler)

    result = []

    label_indices_total = None

    for cnt, each in tqdm(enumerate(loader)):
        (
            fbank,
            log_magnitudes_stft,
            label_indices,
            fname,
            waveform,
            clip_label,
            text,
        ) = each
        check_batch(each)
        break


class ADTOFDataset(torch.utils.data.Dataset):
    """ADTOF dataset class, for processing ADTOF format annotation files"""
    
    def __init__(self, adtof_root, config, train=False, whole_track=True):
        super().__init__()
        
        self.adtof_root = adtof_root
        self.config = config
        self.train = train
        self.whole_track = whole_track
        
        # audio processing parameters
        self.melbins = config["preprocessing"]["mel"]["n_mel_channels"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = config["preprocessing"]["stft"]["hop_length"]
        self.target_length = config["preprocessing"]["mel"]["target_length"]
        self.segment_length = int(self.target_length * self.hopsize)
        
        # path setting - corrected ADTOF structure
        # try multiple possible path structures
        possible_audio_paths = [
            os.path.join(adtof_root, "usable_dataset", "audio", "audio"),  # actual structure
            os.path.join(adtof_root, "dataset", "audio", "audio"),         # original structure
            os.path.join(adtof_root, "audio", "audio"),                    # simplified structure
        ]
        
        possible_annotation_paths = [
            os.path.join(adtof_root, "usable_dataset", "annotations", "aligned_drum"),  # actual structure
            os.path.join(adtof_root, "dataset", "annotations", "aligned_drum"),         # original structure
            os.path.join(adtof_root, "annotations", "aligned_drum"),                    # simplified structure
        ]
        
        # find the first existing audio directory
        self.audio_dir = None
        for path in possible_audio_paths:
            if os.path.exists(path):
                self.audio_dir = path
                print(f"found audio directory: {self.audio_dir}")
                break
        
        if not self.audio_dir:
            print(f"warning: audio directory not found, tried paths: {possible_audio_paths}")
        
        # find the first existing annotation directory
        self.annotation_dir = None
        for path in possible_annotation_paths:
            if os.path.exists(path):
                self.annotation_dir = path
                print(f"found annotation directory: {self.annotation_dir}")
                break
        
        if not self.annotation_dir:
            print(f"warning: annotation directory not found, tried paths: {possible_annotation_paths}")
        
        # load data
        self.data = self.load_data()
        print(f"ADTOF dataset loaded, {len(self.data)} files")
        
        # STFT setting
        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )
    
    def load_data(self):
        """load ADTOF data"""
        data = []
        
        if self.annotation_dir and os.path.exists(self.annotation_dir):
            # load from annotation directory
            annotation_files = [f for f in os.listdir(self.annotation_dir) if f.endswith('.txt')]
            print(f"found {len(annotation_files)} annotation files")
            
            for ann_file in annotation_files:
                # extract file name (remove .txt suffix)
                base_name = os.path.splitext(ann_file)[0]
                
                # build audio file path
                audio_path = None
                if self.audio_dir and os.path.exists(self.audio_dir):
                    # try different audio formats, including .ogg
                    for ext in ['.ogg', '.wav', '.mp3', '.flac']:
                        potential_path = os.path.join(self.audio_dir, base_name + ext)
                        if os.path.exists(potential_path):
                            audio_path = potential_path
                            break
                
                annotation_path = os.path.join(self.annotation_dir, ann_file)
                
                data.append({
                    'audio_path': audio_path,
                    'annotation_path': annotation_path,
                    'base_name': base_name
                })
        
        print(f"loaded {len(data)} data items")
        if len(data) > 0:
            print(f"example data item: {data[0]}")
        
        return data
    
    def parse_adtof_annotation(self, annotation_path):
        """parse ADTOF annotation file, convert to pianoroll format"""
        if not os.path.exists(annotation_path):
            return None
        
        # read annotation file
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        # parse timestamp and notes
        events = []
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        timestamp = float(parts[0])
                        note = int(parts[1])
                        events.append((timestamp, note))
                    except (ValueError, IndexError):
                        continue
        
        # convert to pianoroll format
        # here we simplify the processing, in reality, it may need more complex logic
        pianoroll = np.zeros((self.target_length, 128))  # 128 MIDI notes
        
        for timestamp, note in events:
            # convert timestamp to frame index
            frame_idx = int(timestamp * self.sampling_rate / self.hopsize)
            if 0 <= frame_idx < self.target_length and 0 <= note < 128:
                pianoroll[frame_idx, note] = 1.0
        
        return pianoroll
    
    def read_wav(self, filename):
        """read audio file"""
        if not filename or not os.path.exists(filename):
            # if no audio file, return zero audio
            print(f"warning: audio file not found, using zero audio: {filename}")
            return torch.zeros(1, self.segment_length)
        
        y, sr = torchaudio.load(filename)
        
        # resample
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, sr, self.sampling_rate)
        
        # convert to mono
        if y.shape[0] > 1:
            y = y.mean(dim=0, keepdim=True)
        
        # normalize
        y = y - y.mean()
        y = y / (torch.max(y.abs()) + 1e-8)
        y = y * 0.5
        
        # segment processing
        if not self.whole_track:
            y = self.random_segment_wav(y)
            # pad
            if y.shape[1] < self.segment_length:
                y = torch.nn.functional.pad(y, (0, self.segment_length - y.shape[1]), 'constant', 0.)
        else:
            # whole track
            if y.shape[1] < self.segment_length:
                y = torch.nn.functional.pad(y, (0, self.segment_length - y.shape[1]), 'constant', 0.)
            else:
                y = y[:, :self.segment_length]
        
        return y
    
    def random_segment_wav(self, x):
        """random segment audio"""
        wav_len = x.shape[-1]
        if wav_len <= self.segment_length:
            return x
        
        if self.train:
            sta = random.randint(0, wav_len - self.segment_length)
        else:
            sta = (wav_len - self.segment_length) // 2
        
        return x[:, sta:sta + self.segment_length]
    
    def get_mel(self, filename):
        """get mel spectrogram"""
        y = self.read_wav(filename)
        mel_output = self.STFT.mel_spectrogram(y)
        
        # handle the case where STFT.mel_spectrogram may return a tuple
        if isinstance(mel_output, tuple):
            mel = mel_output[0]  # take the first element
        else:
            mel = mel_output
            
        mel = mel.squeeze(0).transpose(0, 1)  # [n_mels, T] -> [T, n_mels]
        
        # adjust length
        if mel.shape[0] < self.target_length:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, self.target_length - mel.shape[0]))
        else:
            mel = mel[:self.target_length, :]
        
        return mel
    
    def __getitem__(self, index):
        """get data item"""
        item = self.data[index]
        
        # get audio features
        if item['audio_path']:
            mel = self.get_mel(item['audio_path'])
        else:
            # if no audio file, use zero features
            mel = torch.zeros(self.target_length, self.melbins)
        
        # get annotation
        pianoroll = self.parse_adtof_annotation(item['annotation_path'])
        if pianoroll is None:
            pianoroll = np.zeros((self.target_length, 128))
        
        # convert to tensor - modify to [1, 5, T, n_mels] to match model expectation
        mel = mel.unsqueeze(0).repeat(5, 1, 1)  # [5, T, n_mels]
        mel = mel.unsqueeze(0)  # [1, 5, T, n_mels] - add batch dimension
        pianoroll = torch.from_numpy(pianoroll).float()
        
        return {
            'fbank': mel,  # condition features [1, 5, T, n_mels]
            'waveform': mel,  # input features [1, 5, T, n_mels]
            'pianoroll': pianoroll,  # annotation
            'fname': item['base_name'],
            'annotation_path': item['annotation_path']  # add annotation path
        }
    
    def __len__(self):
        return len(self.data)


class ADTOFInferenceDataset(torch.utils.data.Dataset):
    """ADTOF inference dataset, support single file or batch inference"""
    
    def __init__(self, input_path, config, is_single_file=True):
        super().__init__()
        
        self.config = config
        self.is_single_file = is_single_file
        
        # audio processing parameters
        self.melbins = config["preprocessing"]["mel"]["n_mel_channels"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = config["preprocessing"]["stft"]["hop_length"]
        self.target_length = config["preprocessing"]["mel"]["target_length"]
        self.segment_length = int(self.target_length * self.hopsize)
        
        # load data
        if is_single_file:
            self.data = [{'audio_path': input_path, 'fname': os.path.basename(input_path)}]
        else:
            # batch mode: input_path is a directory
            self.data = []
            for ext in ['.wav', '.mp3', '.flac']:
                audio_files = glob.glob(os.path.join(input_path, f'*{ext}'))
                for audio_file in audio_files:
                    self.data.append({
                        'audio_path': audio_file,
                        'fname': os.path.basename(audio_file)
                    })
        
        print(f"inference dataset loaded, {len(self.data)} files")
        
        # STFT setting
        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )
    
    def read_wav(self, filename):
        """read audio file"""
        y, sr = torchaudio.load(filename)
        
        # resample
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, sr, self.sampling_rate)
        
        # convert to mono
        if y.shape[0] > 1:
            y = y.mean(dim=0, keepdim=True)
        
        # normalize
        y = y - y.mean()
        y = y / (torch.max(y.abs()) + 1e-8)
        y = y * 0.5
        
        # process length
        if y.shape[1] < self.segment_length:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.shape[1]), 'constant', 0.)
        else:
            y = y[:, :self.segment_length]
        
        return y
    
    def get_mel(self, filename):
        """get mel spectrogram"""
        y = self.read_wav(filename)
        mel = self.STFT.mel_spectrogram(y)
        mel = mel.squeeze(0).transpose(0, 1)  # [n_mels, T] -> [T, n_mels]
        
        # adjust length
        if mel.shape[0] < self.target_length:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, self.target_length - mel.shape[0]))
        else:
            mel = mel[:self.target_length, :]
        
        return mel
    
    def __getitem__(self, index):
        item = self.data[index]
        mel = self.get_mel(item['audio_path'])
        mel = mel.unsqueeze(0).repeat(5, 1, 1)  # [5, T, n_mels]
        mel = mel.unsqueeze(0)  # [1, 5, T, n_mels] - add batch dimension
        return {
            'fbank': mel,  # condition features [1, 5, T, n_mels]
            'waveform': mel,  # input features [1, 5, T, n_mels]
            'fname': item['fname']
        }
    
    def __len__(self):
        return len(self.data)


class MDBDrumsSingleFileDataset(torch.utils.data.Dataset):
    """dataset for single drum mix file"""
    
    def __init__(self, dataset_path, label_path, config, train=True, factor=1.0, whole_track=False):
        self.dataset_path = dataset_path
        self.label_path = label_path  # add label_path parameter
        self.config = config
        self.train = train
        self.factor = factor
        self.whole_track = whole_track

        # load wav files (support recursive). can use config["path"]["select_wav_name"] to select specific file (e.g. demucs's drums.wav)
        self.audio_files = []
        select_name = None
        try:
            select_name = (self.config or {}).get('path', {}).get('select_wav_name', None)
        except Exception:
            select_name = None
        if os.path.isdir(dataset_path):
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if not file.lower().endswith('.wav'):
                        continue
                    if select_name is not None and file != select_name:
                        continue
                    self.audio_files.append(os.path.join(root, file))
        else:
            # if it is a single file
            if dataset_path.lower().endswith('.wav'):
                if (select_name is None) or (os.path.basename(dataset_path) == select_name):
                    self.audio_files = [dataset_path]
        
        self.audio_files.sort()
        print(f"[MDBDrumsSingleFileDataset] found {len(self.audio_files)} audio files")
        
        # check file length
        self.valid_files = []
        for file in self.audio_files:
            duration = self.get_duration_sec(file)
            if 10.24 <= duration <= 640.0:  # same as original filter logic
                self.valid_files.append(file)
            else:
                print(f"skip file {file}, length {duration:.2f} seconds")
        
        print(f"[MDBDrumsSingleFileDataset] valid files {len(self.valid_files)}")

        # segment length: automatically match model training window length (target_length * hop / sr)
        sr_cfg = self.config['preprocessing']['audio']['sampling_rate']
        hop = self.config['preprocessing']['stft']['hop_length']
        tlen = self.config['preprocessing']['mel']['target_length']
        self.segment_sec = float(tlen * hop / sr_cfg)  # e.g. 1024*160/16000=10.24 seconds
        # step length is the same as segment length, to avoid many tail segments of zero padding; the name is still in seconds (floor) 0/10/20/...
        self.step_sec = self.segment_sec
        # optional: minimum effective ratio, to avoid very short tail segments (default no filtering)
        try:
            self.min_keep_ratio = float(self.config.get('preprocessing', {}).get('mel', {}).get('min_keep_ratio', 0.0))
        except Exception:
            self.min_keep_ratio = 0.0

        self.items = []  # each element: { 'audio_path': str, 'start_sec': int, 'read_start_sec': float }
        for file in self.valid_files:
            duration = self.get_duration_sec(file)
            if duration <= 0:
                continue
            start = 0.0
            # last segment align to tail, to avoid many tail segments of zero padding
            last_read_start = max(0.0, duration - self.segment_sec)
            while start < duration:
                # read actual start: use start for normal segments; use tail aligned start for near tail
                read_start = start if (start + self.segment_sec) <= duration else last_read_start
                # name still in seconds (floor), keep from_0/from_10/...
                name_start_sec = int(start)
                # filter by minimum effective ratio (can be disabled)
                effective = max(0.0, min(self.segment_sec, duration - read_start))
                ratio = (effective / self.segment_sec) if self.segment_sec > 0 else 1.0
                if ratio >= self.min_keep_ratio:
                    self.items.append({
                        'audio_path': file,
                        'start_sec': name_start_sec,
                        'read_start_sec': float(read_start),
                    })
                start += self.step_sec
    
    def get_duration_sec(self, file):
        """get audio file length (seconds)"""
        try:
            audio_info = torchaudio.info(file)
            duration = audio_info.num_frames / audio_info.sample_rate
            return duration
        except Exception as e:
            print(f"cannot read file {file}: {e}")
            return 0
    
    def read_wav_segment(self, filename, start_sec: float, segment_sec: float):
        """read wav segment from specified start and length, and resample and pad to fixed length."""
        try:
            info = torchaudio.info(filename)
            orig_sr = info.sample_rate
            frame_offset = int(round(start_sec * orig_sr))
            num_frames = int(round(segment_sec * orig_sr))
            waveform, sr = torchaudio.load(filename, frame_offset=frame_offset, num_frames=num_frames)
            
            # ensure mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # resample to target sampling rate
            target_sr = self.config['preprocessing']['audio']['sampling_rate']
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)

            # pad to fixed length (10.0 seconds)
            segment_length = int(round(self.segment_sec * target_sr))
            if waveform.shape[1] < segment_length:
                pad = segment_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad))
            elif waveform.shape[1] > segment_length:
                waveform = waveform[:, :segment_length]
            return waveform.squeeze(0).numpy()  # [T]
                
        except Exception as e:
            print(f"cannot read file {filename}: {e}")
            return np.zeros(16000)  # 1 second of silence
    
    def get_mel_from_waveform(self, waveform):
        """calculate mel-spectrogram from waveform"""
        # use the same mel calculation logic as original
        mel_config = self.config['preprocessing']['mel']
        stft_config = self.config['preprocessing']['stft']
        
        # convert to tensor
        waveform_tensor = torch.from_numpy(waveform).float()
        
        # calculate mel-spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config['preprocessing']['audio']['sampling_rate'],
            n_fft=stft_config['filter_length'],
            hop_length=stft_config['hop_length'],
            win_length=stft_config['win_length'],
            n_mels=mel_config['n_mel_channels'],
            f_min=mel_config['mel_fmin'],
            f_max=mel_config['mel_fmax']
        )
        
        mel_spec = mel_transform(waveform_tensor)
        mel_spec = torch.log(mel_spec + 1e-9)  # [F, T] log scale
        
        # adjust to target length (along time dimension)
        target_length = mel_config['target_length']
        if mel_spec.shape[1] > target_length:
            mel_spec = mel_spec[:, :target_length]
        elif mel_spec.shape[1] < target_length:
            pad_length = target_length - mel_spec.shape[1]
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_length))
        
        # align with other datasets as [T, F]
        return mel_spec.transpose(0, 1).numpy()
    
    def __getitem__(self, index):
        """get a sample: segment by 10 seconds, named from_<start_sec>."""
        item = self.items[index]
        audio_file = item['audio_path']
        start_sec = int(item['start_sec'])           # name using
        read_start_sec = float(item.get('read_start_sec', start_sec))  # actual read using

        # read fixed length waveform (tail segments will be automatically aligned to tail, to avoid many tail segments of zero padding)
        waveform = self.read_wav_segment(audio_file, start_sec=read_start_sec, segment_sec=self.segment_sec)

        # calculate segment mel-spectrogram
        fbank = self.get_mel_from_waveform(waveform)

        num_stems = 5  # same as num_stems in config
        
        # create zero-padded stems: same as model expectation, shape [5, T, F] and [5, T]
        fbank_stems = np.zeros((num_stems, fbank.shape[0], fbank.shape[1]), dtype=np.float32)
        waveform_stems = np.zeros((num_stems, len(waveform)), dtype=np.float32)
        
        # prepare output - same as MultiSource_Slakh_Dataset format
        # generate friendly name: <parent folder>_<file name without extension>, e.g. MusicDelta_Beatles_MIX_drums
        parent = os.path.basename(os.path.dirname(audio_file))
        base_no_ext = os.path.splitext(os.path.basename(audio_file))[0]
        friendly_name = f"{parent}_{base_no_ext}_from_{start_sec}"
        data_dict = {
            'fname': friendly_name,
            # align with other datasets: use stems version as main key
            'fbank_stems': fbank_stems,        # [5, T, F]
            'waveform_stems': waveform_stems,  # [5, T]
            # keep single channel key, for branches that need cond_stage_key='fbank'
            'fbank': fbank,                    # [T, F]
            'waveform': waveform,              # [T]
        }
        
        return data_dict
    
    def __len__(self):
        return len(self.items)


class StemGMDOnsetDataset(torch.utils.data.Dataset):
    """
    StemGMD dataset, support dynamic generation of onset-pianoroll labels
    for training onset-pianoroll prediction model
    """
    def __init__(self, dataset_path, label_path, config, train=True, factor=1.0, whole_track=False):
        super().__init__()
        self.train = train
        self.config = config
        self.whole_track = whole_track
        
        # audio processing parameters
        self.melbins = config["preprocessing"]["mel"]["n_mel_channels"]
        self.freqm = config["preprocessing"]["mel"]["freqm"]
        self.timem = config["preprocessing"]["mel"]["timem"]
        self.mixup = config["augmentation"]["mixup"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = config["preprocessing"]["stft"]["hop_length"]
        self.target_length = config["preprocessing"]["mel"]["target_length"]
        self.use_blur = config["preprocessing"]["mel"]["blur"]
        self.segment_length = int(self.target_length * self.hopsize)
        
        # onset-pianoroll parameters
        self.hop_length = 160  # 0.25 seconds hop length
        self.num_stems = 5  # kick, snare, toms, hi_hats, cymbals

        # read data

        self.data = self.read_datafile(dataset_path, label_path, train)
            
        print(f"StemGMD Onset Dataset: {len(self.data)} tracks loaded")
        
        # use factor parameter to control dataset size
        self.factor = factor
        self.total_len = int(len(self.data) * factor)
        print(f"StemGMD Onset Dataset: {len(self.data)} tracks loaded, factor={factor}, total_len={self.total_len}")
        
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

        # additional properties aligned with MultiSource_Slakh_Dataset
        self.text_prompt = config.get('path', {}).get('text_prompt', None)
        self.stem_masking = config.get('augmentation', {}).get('masking', False)
        # read track_mapping.csv, build track_id -> onset json filename mapping
        # use path in config, if not set, use default path
        mapping_path = config.get('model', {}).get('params', {}).get('track_mapping_path', '/PATH/TO/PROJECT/data/StemGMD_org/track_mapping.csv')
        if os.path.exists(mapping_path):
            self.trackid2json = self._build_trackid2json(mapping_path)
        else:
            print(f"warning: track_mapping.csv does not exist at {mapping_path}")
            self.trackid2json = {}
        
        # add timbre_data_path support
        self.timbre_data_path = config.get('model', {}).get('params', {}).get('timbre_data_path', None)
        if self.timbre_data_path:
            print(f"[StemGMDOnsetDataset] will load precomputed timbre features from {self.timbre_data_path}")
        
        # add onset_data_path support
        self.onset_data_path = config.get('model', {}).get('params', {}).get('onset_data_path', '/PATH/TO/PROJECT/midi_onset_gt')
        print(f"[StemGMDOnsetDataset] onset data path: {self.onset_data_path}")
    
    def read_datafile(self, dataset_path, label_path, train):
        """change to similar to MultiSource_Slakh_Dataset:
        - list tracks directly under dataset_path
        - filter by length
        - create multiple segments for long files (frame_offset)
        """
        data = []
        tracks = os.listdir(dataset_path)
        print(f"Found {len(tracks)} tracks.")
        keep, durations, _ = self._filter_tracks_like_slakh(tracks, dataset_path)
        for idx in range(len(keep)):
            entry = {
                'wav_path': os.path.join(dataset_path, keep[idx]),
                'duration': durations[idx],
            }
            data.append(entry)

        # generate list with multiple segments
        temp_data = []
        max_samples = 640.0 * self.sampling_rate
        for entry in data:
            entry['frame_offset'] = 0
            duration = entry['duration']
            temp_data.append(entry)
            if duration > self.segment_length:
                num_copies = int((min(duration, max_samples) - self.segment_length) / self.segment_length)
                for i in range(num_copies):
                    new_entry = entry.copy()
                    new_entry['frame_offset'] = (i + 1) * self.segment_length
                    temp_data.append(new_entry)

        data = temp_data
        print(f"StemGMD Onset Dataset: {len(data)} tracks loaded, factor={1.0}, total_len={len(data)}")
        return data

    def _get_duration_sec_like_slakh(self, file, cache=False):
        if not os.path.exists(file):
            return 0
        try:
            with open(file + ".dur", "r") as f:
                duration = float(f.readline().strip("\n"))
        except FileNotFoundError:
            audio_info = torchaudio.info(file)
            duration = audio_info.num_frames / audio_info.sample_rate
            if cache:
                with open(file + ".dur", "w") as f:
                    f.write(str(duration) + "\n")
        return duration

    def _filter_tracks_like_slakh(self, tracks, audio_files_dir):
        keep = []
        durations = []
        for track in tracks:
            track_dir = os.path.join(audio_files_dir, track)
            files = [os.path.join(track_dir, stem + ".wav") for stem in self.config["path"]["stems"]]
            exist_files = [f for f in files if os.path.exists(f)]
            
            # special handling: if no expected stems are found, but drums.wav is found, also keep it
            if not exist_files:
                drums_file = os.path.join(track_dir, "drums.wav")
                if os.path.exists(drums_file):
                    exist_files = [drums_file]
                else:
                    continue
            
            durations_track = np.array([self._get_duration_sec_like_slakh(file, cache=True) * self.sampling_rate for file in exist_files])
            duration = durations_track[0]
            if (duration / self.sampling_rate < 10.0):
                continue
            if (duration / self.sampling_rate >= 640.0):
                continue
            keep.append(track)
            durations.append(duration)
        print(f"sr={self.sampling_rate}, min: {10}, max: {600}")
        print(f"Keeping {len(keep)} of {len(tracks)} tracks")
        return keep, durations, np.cumsum(np.array(durations))
    
    def normalize_wav(self, x):
        """normalize waveform"""
        x = x[0]
        x = x - x.mean()
        x = x / (torch.max(x.abs()) + 1e-8)
        x = x * 0.5
        x = x.unsqueeze(0)
        return x
    
    def random_segment_wav(self, x):
        """randomly segment waveform"""
        wav_len = x.shape[-1]
        assert wav_len > 100, "Waveform is too short, %s" % wav_len
        
        if self.whole_track:
            return x
            
        if wav_len - self.segment_length > 0:
            if self.train:
                sta = random.randint(0, wav_len - self.segment_length)
            else:
                sta = (wav_len - self.segment_length) // 2
            x = x[:, sta: sta + self.segment_length]
        return x
    
    def read_wav(self, filename, frame_offset_sec=0.0):
        """read wav segment by seconds (aligned with MultiSource_Slakh)"""
        start_frame = int(frame_offset_sec * 44100)
        y, sr = torchaudio.load(filename, frame_offset=start_frame, num_frames=int(44100 * 10.24))
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, sr, self.sampling_rate)
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y
    
    def get_mel(self, filename, mix_filename=None, frame_offset=0.0):
        """compatible interface with MultiSource_Slakh: return (waveform, mel)"""
        y = self.read_wav(filename, frame_offset)
        y.requires_grad = False
        
        # fix AssertionError: ensure waveform is in [-1, 1] range
        y = torch.clamp(y, -1.0, 1.0)
        
        melspec, _, _ = self.STFT.mel_spectrogram(y)
        melspec = melspec[0].T
        if melspec.size(0) < self.target_length:
            melspec = torch.nn.functional.pad(melspec, (0, 0, 0, self.target_length - melspec.size(0)), 'constant', 0.)
        else:
            if not self.whole_track:
                melspec = melspec[0: self.target_length, :]
        if melspec.size(-1) % 2 != 0:
            melspec = melspec[:, :-1]
        return y[0].numpy(), melspec.numpy()

    def get_mel_from_waveform(self, waveform):
        """input single stem waveform (shape [T] or [1, T]), return mel (shape [T_frames, melbins])"""
        if isinstance(waveform, np.ndarray):
            if waveform.ndim == 1:
                y = torch.from_numpy(waveform).unsqueeze(0)
            elif waveform.ndim == 2 and waveform.shape[0] == 1:
                y = torch.from_numpy(waveform)
            else:
                # revert: take first channel
                y = torch.from_numpy(waveform)
                y = y[:1, :]
        else:
            # tensor input
            y = waveform
            if y.dim() == 1:
                y = y.unsqueeze(0)
            elif y.dim() > 2:
                y = y[:1, :]
        y = y.float()
        y.requires_grad = False
        
        # fix AssertionError: ensure waveform is in [-1, 1] range
        y = torch.clamp(y, -1.0, 1.0)
        
        melspec, _, _ = self.STFT.mel_spectrogram(y)
        melspec = melspec[0].T
        if melspec.size(0) < self.target_length:
            melspec = torch.nn.functional.pad(melspec, (0, 0, 0, self.target_length - melspec.size(0)), 'constant', 0.)
        else:
            if not self.whole_track:
                melspec = melspec[0: self.target_length, :]
        if melspec.size(-1) % 2 != 0:
            melspec = melspec[:, :-1]
        return melspec.cpu().numpy()

    def _apply_wave_augmentations(self, waveform_stems: np.ndarray) -> np.ndarray:
        """apply augmentations to waveform_stems before calculating mel
        limitation: no time stretch/reordering, to avoid breaking GT onset; only do amplitude/spectrum preserving duration conversion.
        parameters: waveform_stems shape [S, 1, T], range [-1,1]
        return same shape ndarray.
        """
        if (not getattr(self, 'train', False)):
            return waveform_stems
        # guard: if wave_aug is missing or not enabled, return directly
        if (not hasattr(self, 'wave_aug')) or (not isinstance(self.wave_aug, dict)) or (not self.wave_aug.get('enable', False)):
            return waveform_stems
        import math
        S, C, T = waveform_stems.shape
        if random.random() >= self.wave_aug['p_apply']:
            return waveform_stems

        out = waveform_stems.copy()

        # 1) gain for all stems (RX)
        if random.random() < self.wave_aug['rx_prob']:
            lo, hi = self.wave_aug['rx_gain_db']
            gain_db = random.uniform(float(lo), float(hi))
            gain = float(10.0 ** (gain_db / 20.0))
            out = np.clip(out * gain, -1.0, 1.0)

        # 2) pitch shift (keep length) and tanh saturation for each stem
        for s in range(S):
            w = out[s]  # [1, T]
            # pitch shift
            if random.random() < self.wave_aug['ps_prob']:
                try:
                    import torch
                    import torchaudio
                    semi_lo, semi_hi = self.wave_aug['ps_semitones']
                    n_steps = int(random.randint(int(semi_lo), int(semi_hi)))
                    if n_steps != 0:
                        wt = torch.from_numpy(w.copy()).float()
                        wt = torchaudio.functional.pitch_shift(wt, sample_rate=int(self.sampling_rate), n_steps=float(n_steps))
                        w = wt.numpy()
                except Exception:
                    pass
            # saturation
            if random.random() < self.wave_aug['st_prob']:
                beta_lo, beta_hi = self.wave_aug['st_beta']
                beta = float(random.uniform(float(beta_lo), float(beta_hi)))
                w = np.tanh(beta * w) / math.tanh(beta)
            # polarity flip (no change in time sequence)
            if random.random() < self.wave_aug['polarity_prob']:
                w = -w
            out[s] = np.clip(w, -1.0, 1.0)

        return out

    def get_index_offset(self, item):
        """similar to MultiSource_Slakh: convert frame_offset (samples) + random shift to seconds."""
        half_interval = self.segment_length // 2
        shift = np.random.randint(-half_interval, half_interval) if self.train else 0
        offset = item["frame_offset"] + shift
        start, end = 0.0, item["duration"]
        if offset > end - self.segment_length:
            offset = max(start, offset - half_interval)
        if offset < start:
            offset = 0.0
        offset = offset / self.sampling_rate
        return item, offset
    
    def load_onset_data(self, onset_json_path):
        """load onset data"""
        if not os.path.exists(onset_json_path):
            print(f"Warning: Onset file not found: {onset_json_path}")
            return None
        
        with open(onset_json_path, 'r') as f:
            onset_data = json.load(f)
        
        return onset_data
    
    def create_onset_pianoroll(self, onset_data, duration_sec):
        """create onset pianoroll"""
        if onset_data is None:
            # if no onset data, return all zero matrix
            num_frames = int(duration_sec * self.sampling_rate / self.hop_length)
            return np.zeros((self.num_stems, num_frames))
        
        # calculate number of frames
        num_frames = int(duration_sec * self.sampling_rate / self.hop_length)
        onset_pianoroll = np.zeros((self.num_stems, num_frames))
        
        # stem mapping: according to STEM_JSON_MAP merge rules
        stem_mapping = {
            # Kick (stem_0)
            'kick': 0, 'Bass Drum 1': 0, 'Bass Drum': 0,
            # Snare (stem_1) 
            'snare': 1, 'Acoustic Snare': 1, 'Snare': 1,
            # Toms (stem_2)
            'hi_tom': 2, 'mid_tom': 2, 'low_tom': 2,
            'High Tom': 2, 'Mid Tom': 2, 'Low Tom': 2,
            'High Floor Tom': 2, 'Low-Mid Tom': 2, 'Floor Tom': 2,
            # Hi-Hats (stem_3)
            'hh_closed': 3, 'hh_open': 3,
            'Closed Hi Hat': 3, 'Open Hi Hat': 3, 'Hi Hat': 3,
            # Cymbals (stem_4)
            'crash': 4, 'ride': 4, 
            'Crash Cymbal 1': 4, 'Ride Cymbal 1': 4,
            'Crash Cymbal': 4, 'Ride Cymbal': 4
        }
        
        # fill onset data
        mapped_count = 0
        for stem_name, onsets in onset_data.items():
            if stem_name in stem_mapping:
                stem_idx = stem_mapping[stem_name]
                mapped_count += 1
                for onset_time in onsets:
                    frame_idx = int(onset_time * self.sampling_rate / self.hop_length)
                    if 0 <= frame_idx < num_frames:
                        onset_pianoroll[stem_idx, frame_idx] = 1.0
            else:
                print(f"no mapping found: {stem_name}")
        
        return onset_pianoroll
    
    def __getitem__(self, index):
        """similar to MultiSource_Slakh_Dataset: fbank_stems, waveform_stems, fbank, waveform, onset_pianoroll."""
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]
        entry, frame_offset = self.get_index_offset(f)
        stems_list = self.config["path"]["stems"]
        all_stem_names = ['crash', 'hh_closed', 'hh_open', 'hi_tom', 'kick', 'low_tom', 'mid_tom', 'ride', 'snare']
        all_audio = {}
        first_valid_audio = None
        
        # check if drums.wav exists (Demucs output)
        drums_path = os.path.join(entry["wav_path"], "drums.wav")
        has_drums_wav = os.path.exists(drums_path)
        
        if has_drums_wav:
            # if drums.wav exists, read it as mix, create empty stems for inference
            drums_audio, _ = self.get_mel(drums_path, None, frame_offset)
            first_valid_audio = drums_audio
            # create empty stems, so the model can infer separation based on mix
            for stem in all_stem_names:
                all_audio[stem] = np.zeros_like(drums_audio)
        else:
            # original logic: read each stem
            for stem in all_stem_names:
                stem_path = os.path.join(entry["wav_path"], stem + ".wav")
                if os.path.exists(stem_path):
                    audio, _ = self.get_mel(stem_path, None, frame_offset)
                    if first_valid_audio is None:
                        first_valid_audio = audio
                else:
                    audio = np.zeros_like(first_valid_audio) if first_valid_audio is not None else None
                all_audio[stem] = audio

        if stems_list == ['kick', 'snare', 'toms', 'hi_hats', 'cymbals']:
            # ensure all audio are numpy arrays, avoid None values
            kick_audio = all_audio['kick'] if all_audio['kick'] is not None else np.zeros_like(first_valid_audio)
            snare_audio = all_audio['snare'] if all_audio['snare'] is not None else np.zeros_like(first_valid_audio)
            hi_tom_audio = all_audio['hi_tom'] if all_audio['hi_tom'] is not None else np.zeros_like(first_valid_audio)
            mid_tom_audio = all_audio['mid_tom'] if all_audio['mid_tom'] is not None else np.zeros_like(first_valid_audio)
            low_tom_audio = all_audio['low_tom'] if all_audio['low_tom'] is not None else np.zeros_like(first_valid_audio)
            hh_closed_audio = all_audio['hh_closed'] if all_audio['hh_closed'] is not None else np.zeros_like(first_valid_audio)
            hh_open_audio = all_audio['hh_open'] if all_audio['hh_open'] is not None else np.zeros_like(first_valid_audio)
            crash_audio = all_audio['crash'] if all_audio['crash'] is not None else np.zeros_like(first_valid_audio)
            ride_audio = all_audio['ride'] if all_audio['ride'] is not None else np.zeros_like(first_valid_audio)
            audio_list = [
                kick_audio[np.newaxis, :],
                snare_audio[np.newaxis, :],
                (hi_tom_audio + mid_tom_audio + low_tom_audio)[np.newaxis, :],
                (hh_closed_audio + hh_open_audio)[np.newaxis, :],
                (crash_audio + ride_audio)[np.newaxis, :],
            ]
        else:
            # revert: collect stems_list one by one
            audio_list = []
            for stem in stems_list:
                wav = all_audio.get(stem, None)
                if wav is None:
                    wav = np.zeros_like(first_valid_audio)
                audio_list.append(wav[np.newaxis, :])

        waveform_stems = np.concatenate(audio_list, axis=0)
        waveform_stems = self._apply_wave_augmentations(waveform_stems)
        
        # if drums.wav exists, use it as mix_waveform
        if has_drums_wav:
            mix_waveform = drums_audio
        else:
            mix_waveform = np.clip(np.sum(waveform_stems, axis=0), -1, 1)

        # generate mel for each stem (from enhanced waveform)
        fbank_list = [np.expand_dims(self.get_mel_from_waveform(waveform_stems[i]), axis=0) for i in range(waveform_stems.shape[0])]
        fbank_stems = np.concatenate(fbank_list, axis=0)
        mix_fbank = self.get_mel_from_waveform(mix_waveform)


        F, T = mix_fbank.shape[1], mix_fbank.shape[0]
        guides_raw = np.zeros((4, F, T), dtype=np.float32)  # 4 guide features
        film_vec = np.zeros(4, dtype=np.float32)
        
        # check if guide features are enabled
        enable_guide_features = getattr(self.config, 'enable_guide_features', False)
        if hasattr(self.config, 'model') and hasattr(self.config.model, 'params'):
            enable_guide_features = getattr(self.config.model.params, 'enable_guide_features', enable_guide_features)
        
        if enable_guide_features:
            try:
                # import guide features calculation function
                import sys
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'audio'))
                from guides import compute_guides_from_mel
                
                # get config parameters
                guide_config = {}
                try:
                    if hasattr(self.config, 'model') and hasattr(self.config.model, 'params'):
                        guide_config = getattr(self.config.model.params, 'guide_features', {})
                except Exception:
                    pass
                
                F_med = guide_config.get('F_med', 17)
                T_med = guide_config.get('T_med', 9)
                p_mask = guide_config.get('p_mask', 2)
                hf_threshold = guide_config.get('hf_threshold', 6000)
                lf_threshold = guide_config.get('lf_threshold', 200)
                
                # convert mix_fbank from [T, F] to [F, T] and convert to power format
                # mix_fbank is currently in log-mel format, need to convert to power format
                mel_power = np.exp(mix_fbank)  # log -> power
                mel_power = mel_power.T  # [T, F] -> [F, T]
                
                # calculate guide features
                guides_raw, film_vec, _ = compute_guides_from_mel(
                    mel_power=mel_power,
                    sr=self.sampling_rate,
                    hop_length=self.hopsize,
                    mel_fmin=0.0,
                    mel_fmax=8000.0,
                    F_med=F_med,
                    T_med=T_med,
                    p_mask=p_mask,
                    hf_threshold=hf_threshold,
                    lf_threshold=lf_threshold
                )
                
                # ensure data types are correct
                guides_raw = guides_raw.astype(np.float32)
                film_vec = film_vec.astype(np.float32)
                
            except Exception as e:
                print(f"calculate guide features failed: {e}")
                # if calculation fails, keep default values (already initialized above)
                pass

        # timbre_features: load from file if possible, otherwise calculate on the fly
        track_id = os.path.basename(entry['wav_path'])
        timbre_loaded = False
        
        # try to load from precomputed files
        if hasattr(self, 'timbre_data_path') and self.timbre_data_path:
            # construct file paths according to naming rules
            # possible file names:
            # 1. {track_id}_5stems_timbre_from_{offset}.npy (e.g. IDMT)
            # 2. {track_id}_5stems_timbre.npy (e.g. StemGMD)
            # 3. {track_id}.npy (simplified version)
            
            possible_files = [
                os.path.join(self.timbre_data_path, f"{track_id}_5stems_timbre_from_{int(frame_offset)}.npy"),
                os.path.join(self.timbre_data_path, f"{track_id}_5stems_timbre.npy"),
                os.path.join(self.timbre_data_path, f"{track_id}.npy"),
            ]
            
            # try to load the first existing file
            for timbre_file in possible_files:
                if os.path.exists(timbre_file):
                    try:
                        # load precomputed timbre features
                        timbre_features = np.load(timbre_file).astype(np.float32)
                        
                        # check if shape is correct (should be 5x7)
                        if timbre_features.shape != (5, 7):
                            print(f"warning: timbre features shape incorrect {timbre_features.shape}, expected (5, 7)")
                            continue
                        
                        # normalize to 0~1 range
                        timbre_features = timbre_features / 100.0
                        timbre_loaded = True
                        # only print info for the first few loads to avoid too much output
                        if index % 100 == 0:
                            print(f"successfully loaded timbre features: {os.path.basename(timbre_file)}")
                        break
                    except Exception as e:
                        print(f"failed to load timbre file {timbre_file}: {e}")
                        continue
        
        # if not loaded, calculate on the fly
        if not timbre_loaded:
            S, T, F = fbank_stems.shape
            groups = 7
            edges = np.linspace(0, F, groups + 1, dtype=int)
            feat = np.zeros((S, groups), dtype=np.float32)
            # use log mel (already log) to average, and normalize each stem by max value
            for s in range(S):
                for g in range(groups):
                    a, b = edges[g], edges[g + 1]
                    band = fbank_stems[s, :, a:b]
                    if band.size == 0:
                        feat[s, g] = 0.0
                    else:
                        feat[s, g] = float(np.mean(band))
            mx = np.max(np.abs(feat), axis=1, keepdims=True) + 1e-8
            timbre_features = (feat / mx).astype(np.float32)

        # onset pianoroll (same as MultiSource_Slakh)
        # track_id is already defined above
        json_file = self.trackid2json.get(track_id, None)
        # use path from config, if not set, use default path
        midi_onset_dir = getattr(self, 'onset_data_path', '/PATH/TO/PROJECT/midi_onset_gt')
        stem_names = stems_list
        num_stems = len(stem_names)
        target_length = self.target_length
        hop_length = self.hopsize
        sampling_rate = self.sampling_rate
        pianoroll = np.zeros((num_stems, target_length), dtype=np.float32)
        if json_file is not None:
            json_path = os.path.join(midi_onset_dir, json_file)
            if os.path.exists(json_path):
                with open(json_path, 'r') as f_json:
                    onset_dict = json.load(f_json)
                for i, stem in enumerate(stem_names):
                    json_keys = STEM_JSON_MAP.get(stem, [])
                    for key in json_keys:
                        if key in onset_dict:
                            for onset_sec in onset_dict[key]:
                                frame = int(float(onset_sec) * sampling_rate / hop_length)
                                if 0 <= frame < target_length:
                                    pianoroll[i, frame] = 1.0

        data_dict['fname'] = track_id + "_from_" + str(int(frame_offset))
        data_dict['fbank_stems'] = fbank_stems
        data_dict['waveform_stems'] = waveform_stems
        data_dict['waveform'] = mix_waveform
        data_dict['fbank'] = mix_fbank
        data_dict['onset_pianoroll'] = pianoroll
        data_dict['timbre_features'] = timbre_features
        
        # add guide features
        data_dict['guides'] = guides_raw
        data_dict['film_vec'] = film_vec
        if self.text_prompt is not None:
            data_dict['text'] = self.text_prompt
        return data_dict

    def _build_trackid2json(self, mapping_path):
        import pandas as pd
        df = pd.read_csv(mapping_path)
        trackid2json = {}
        for _, row in df.iterrows():
            track_id = row.get('track_id')
            if 'take' in row:
                midi_json = f"{row['take']}_onsets.json"
            elif 'stems_dir' in row:
                midi_json = os.path.basename(row['stems_dir'])
            else:
                midi_json = None
            if track_id is not None and midi_json is not None:
                trackid2json[track_id] = midi_json
        return trackid2json
    
    def __len__(self):
        return self.total_len


class ENSTOnsetDataset(torch.utils.data.Dataset):
    """
    ENST-Drums dataset (Drummer1~3 cross-validation), aligned with StemGMDOnsetDataset:
      - fbank_stems: [S, T, F]
      - waveform_stems: [S, T]
      - fbank: [T, F] (mel of mix)
      - waveform: [T] (mix)
      - onset_pianoroll: [S, T_frames] (S=5: kick/snare/toms/hi_hats/cymbals)

    Assume data root directory structure similar to ENST official:
      dataset_path/
        audio_wet/Drummer1/<track_dir>/*.wav
        audio_wet/Drummer2/...
        audio_wet/Drummer3/...
        annotations/Drummer1/<track_dir>/*.txt
        ...
    """

    def __init__(self, dataset_path, label_path, config, train=True, factor=1.0, whole_track=False) -> None:
        super().__init__()
        self.train = train
        self.config = config
        self.whole_track = whole_track

        # audio/feature parameters
        self.melbins = config["preprocessing"]["mel"]["n_mel_channels"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = config["preprocessing"]["stft"]["hop_length"]
        self.target_length = config["preprocessing"]["mel"]["target_length"]
        self.segment_length = int(self.target_length * self.hopsize)

        # STFT
        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

        # stems and mapping (5 classes)
        self.stems = ['kick', 'snare', 'toms', 'hi_hats', 'cymbals']
        self.num_stems = len(self.stems)

        # cross-validation settings
        self.cv_fold = int(self.config.get('path', {}).get('enst_cv_fold', 0))
        self.cv_fold = max(0, min(2, self.cv_fold))
        self.drummers = ['Drummer1', 'Drummer2', 'Drummer3']
        self.audio_roots = [
            os.path.join(dataset_path, 'audio_wet'),
            os.path.join(dataset_path, 'audio_sum'),
            os.path.join(dataset_path, 'audio'),
            dataset_path,
        ]
        self.ann_roots = [
            os.path.join(dataset_path, 'annotations'),
            os.path.join(dataset_path, 'annotation'),
            dataset_path,
        ]

        # build list
        self.data = self._scan_tracks()
        # factor
        self.total_len = int(len(self.data) * float(factor)) if factor is not None else len(self.data)
        if self.total_len <= 0:
            self.total_len = len(self.data)
        print(f"ENSTOnsetDataset: found {len(self.data)} items (train={self.train}, fold={self.cv_fold})")

    # -------------- scan and parse --------------
    def _pick_root(self, roots):
        for r in roots:
            if os.path.exists(r):
                return r
        return None

    def _list_drummers(self, base):
        if not base or not os.path.exists(base):
            return []
        subs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        cand = [d for d in subs if d.lower().startswith('drummer')]
        if len(cand) == 0:
            # allow direct track directory (no drummer hierarchy)
            return ['.']
        # only keep Drummer1~3
        out = []
        for d in self.drummers:
            if d in cand:
                out.append(d)
        return out if len(out) > 0 else cand

    def _selected_drummers(self, all_drummers):
        if all_drummers == ['.']:
            return ['.']
        if self.train:
            return [d for i, d in enumerate(self.drummers) if i != self.cv_fold and d in all_drummers]
        else:
            d = self.drummers[self.cv_fold]
            return [d] if d in all_drummers else []

    def _scan_tracks(self):
        data = []
        audio_base = self._pick_root(self.audio_roots)
        ann_base = self._pick_root(self.ann_roots)
        if audio_base is None:
            print("[ENST] audio root not found.")
            return data
        drummers = self._list_drummers(audio_base)
        sel = self._selected_drummers(drummers)
        for d in sel:
            d_audio = os.path.join(audio_base, d) if d != '.' else audio_base
            d_ann = os.path.join(ann_base, d) if (ann_base and d != '.') else ann_base
            if not os.path.exists(d_audio):
                continue
            tracks = [t for t in os.listdir(d_audio) if os.path.isdir(os.path.join(d_audio, t))]
            # if no subfolders, use current directory as track container
            if len(tracks) == 0:
                tracks = ['.']
            for t in tracks:
                tdir = os.path.join(d_audio, t)
                # find mix file
                wav = self._find_mix_wav(tdir)
                if wav is None:
                    # fallback: grab the first .wav
                    wav = self._find_any_wav(tdir)
                if wav is None:
                    continue
                # find annotation
                ann = self._find_annotation(d_ann, t)
                item = {
                    'drummer': d,
                    'track': t,
                    'audio_path': wav,
                    'ann_path': ann,
                    'frame_offset': 0,
                }
                # estimate length
                try:
                    info = torchaudio.info(wav)
                    item['duration'] = info.num_frames
                except Exception:
                    item['duration'] = self.segment_length
                data.append(item)
        return data

    def _find_mix_wav(self, tdir):
        cands = []
        for root, _, files in os.walk(tdir):
            for f in files:
                if f.lower().endswith('.wav'):
                    fp = os.path.join(root, f)
                    name = f.lower()
                    if ('mix' in name) or ('sum' in name) or ('full' in name):
                        cands.append(fp)
        if len(cands) == 0:
            return None
        # select the longest one
        try:
            lens = []
            for fp in cands:
                info = torchaudio.info(fp)
                lens.append(info.num_frames)
            idx = int(np.argmax(np.array(lens)))
            return cands[idx]
        except Exception:
            return cands[0]

    def _find_any_wav(self, tdir):
        for root, _, files in os.walk(tdir):
            for f in files:
                if f.lower().endswith('.wav'):
                    return os.path.join(root, f)
        return None

    def _find_annotation(self, ann_base, track_name):
        if not ann_base or not os.path.exists(ann_base):
            return None
        # find .txt / .csv in corresponding track directory
        tdir = os.path.join(ann_base, track_name)
        cands = []
        if os.path.exists(tdir):
            for f in os.listdir(tdir):
                if f.lower().endswith(('.txt', '.csv', '.tsv', '.lab')):
                    cands.append(os.path.join(tdir, f))
        # fallback: global search for annotations containing track_name
        if len(cands) == 0:
            for root, _, files in os.walk(ann_base):
                for f in files:
                    if f.lower().endswith(('.txt', '.csv', '.tsv', '.lab')) and track_name in f:
                        cands.append(os.path.join(root, f))
        return cands[0] if len(cands) > 0 else None

    # -------------- audio/feature --------------
    def read_wav(self, filename, frame_offset_sec=0.0):
        start_frame = int(frame_offset_sec * 44100)
        y, sr = torchaudio.load(filename, frame_offset=start_frame, num_frames=int(44100 * 10.24))
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, sr, self.sampling_rate)
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y

    def get_mel_from_waveform(self, waveform):
        y = waveform
        if isinstance(y, np.ndarray):
            if y.ndim == 1:
                y = torch.from_numpy(y).unsqueeze(0)
            elif y.ndim == 2 and y.shape[0] != 1:
                y = torch.from_numpy(y[:1, :])
            else:
                y = torch.from_numpy(y)
        y = y.float()
        y.requires_grad = False
        melspec, _, _ = self.STFT.mel_spectrogram(y)
        melspec = melspec[0].T
        if melspec.size(0) < self.target_length:
            melspec = torch.nn.functional.pad(melspec, (0, 0, 0, self.target_length - melspec.size(0)), 'constant', 0.)
        else:
            if not self.whole_track:
                melspec = melspec[0: self.target_length, :]
        if melspec.size(-1) % 2 != 0:
            melspec = melspec[:, :-1]
        return melspec.cpu().numpy()

    # -------------- annotation to pianoroll --------------
    def _parse_enst_annotation(self, ann_path):
        if ann_path is None or (not os.path.exists(ann_path)):
            return None
        events = {k: [] for k in self.stems}
        try:
            with open(ann_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = [p for p in line.replace(',', ' ').split() if len(p) > 0]
                    # try to parse: first one that can be parsed as time
                    t = None
                    tag = None
                    for p in parts:
                        try:
                            t = float(p)
                            break
                        except Exception:
                            continue
                    # label: find non-numeric token
                    for p in parts[::-1]:
                        if not self._is_float(p):
                            tag = p.lower()
                            break
                    if t is None or tag is None:
                        continue
                    stem = self._map_tag_to_stem(tag)
                    if stem is not None:
                        events[stem].append(t)
        except Exception:
            return None
        return events

    def _is_float(self, s):
        try:
            _ = float(s)
            return True
        except Exception:
            return False

    def _map_tag_to_stem(self, tag: str):
        # map ENST common abbreviations to 5 stems
        cym_tags = {"cr", "c1", "cr1", "cr2", "c4", "c", "rc", "rc1", "rc2", "rc3", "rc4", "ch", "ch1", "ch5", "spl", "spl2", "cb"}
        tom_tags = {"lft", "lt", "lmt", "mt", "tom", "tt"}
        hat_tags = {"chh", "ohh", "hh", "hho", "hhc"}
        snr_tags = {"sd", "sd-", "rs", "cs", "snare"}
        kick_tags = {"bd", "kick"}
        if tag in cym_tags:
            return 'cymbals'
        if tag in tom_tags:
            return 'toms'
        if tag in hat_tags:
            return 'hi_hats'
        if tag in snr_tags:
            return 'snare'
        if tag in kick_tags:
            return 'kick'
        return None

    def _events_to_pianoroll(self, events: dict, duration_sec: float):
        num_frames = int(self.sampling_rate * duration_sec / self.hopsize)
        pianoroll = np.zeros((self.num_stems, self.target_length), dtype=np.float32)
        name_to_idx = {n: i for i, n in enumerate(self.stems)}
        for name, ts in (events or {}).items():
            si = name_to_idx.get(name, None)
            if si is None:
                continue
            for t in ts:
                f = int(float(t) * self.sampling_rate / self.hopsize)
                if 0 <= f < self.target_length:
                    pianoroll[si, f] = 1.0
        return pianoroll

    # -------------- __getitem__ --------------
    def __getitem__(self, index):
        idx = index % len(self.data)
        item = self.data[idx]
        # read audio segment
        wav = self.read_wav(item['audio_path'], 0.0)
        # mix features
        mix_waveform = wav[0].numpy()
        mix_fbank = self.get_mel_from_waveform(wav)
        # 5 stems use mix (if real stems are available, replace with real stems)
        waveform_stems = np.stack([mix_waveform for _ in range(self.num_stems)], axis=0)
        fbank_stems = np.stack([mix_fbank for _ in range(self.num_stems)], axis=0)
        # pianoroll
        events = self._parse_enst_annotation(item.get('ann_path', None))
        # duration estimate: use 10.24 second window
        pianoroll = self._events_to_pianoroll(events, duration_sec=self.target_length * self.hopsize / self.sampling_rate)

        return {
            'fname': f"{item['drummer']}_{item['track']}",
            'fbank_stems': fbank_stems,
            'waveform_stems': waveform_stems,
            'fbank': mix_fbank,
            'waveform': mix_waveform,
            'onset_pianoroll': pianoroll,
        }

    def __len__(self):
        return self.total_len

class MultiSource_Slakh_Waveform_Dataset(MultiSource_Slakh_Dataset):
    """
    Dataset class for Stable Audio VAE
    Skip STFT and mel calculation, only process waveform data
    """
    def __init__(self, dataset_path, label_path, config, train=True, factor=1.0, whole_track=False) -> None:
        # directly call AudiostockDataset's __init__, skip DS_10283_2325_Dataset's STFT initialization
        super(AudiostockDataset, self).__init__()
        
        self.config = config
        self.train = train
        self.factor = factor
        self.whole_track = whole_track
        
        # audio parameters - copied from AudiostockDataset
        self.melbins = config["preprocessing"]["mel"]["n_mel_channels"]
        self.freqm = config["preprocessing"]["mel"]["freqm"]
        self.timem = config["preprocessing"]["mel"]["timem"]
        self.mixup = config["augmentation"]["mixup"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = config["preprocessing"]["stft"]["hop_length"]
        self.target_length = config["preprocessing"]["mel"]["target_length"]
        self.use_blur = config["preprocessing"]["mel"]["blur"]
        self.segment_length = int(self.target_length * self.hopsize)
        
        # force set to 47 second audio length (2072700 samples @ 44.1kHz)
        self.segment_length = 2072700  # 47 seconds * 44100 Hz
        
        # data loading
        self.data = []
        if type(dataset_path) is str:
            self.data = self.read_datafile(dataset_path, label_path, train) 
        elif type(dataset_path) is list or type(dataset_path) is omegaconf.listconfig.ListConfig:
            for datapath in dataset_path:
                self.data += self.read_datafile(datapath, label_path, train) 
        else:
            raise Exception("Invalid data format")
        print(f"Data size: {len(self.data)}")

        self.total_len = int(len(self.data) * factor)

        try:
            self.segment_size = config["preprocessing"]["audio"]["segment_size"]
            self.target_length = int(self.segment_size / self.hopsize)
            self.segment_length = int(self.target_length * self.hopsize)
            assert self.segment_size % self.hopsize == 0
            print("Use segment size of %s." % self.segment_size)
        except:
            self.segment_size = None
        
        if not train:
            self.mixup = 0.0
            self.freqm = 0
            self.timem = 0

        self.return_all_wav = False
        if self.mixup > 0:
            self.tempo_map = np.load(config["path"]["tempo_map"], allow_pickle=True).item()
            self.tempo_folder = config["path"]["tempo_data"]
        
        if self.mixup > 1:
            self.return_all_wav = config["augmentation"]["return_all_wav"] 

        print(f"Use mixup rate of {self.mixup}; Use SpecAug (T,F) of ({self.timem}, {self.freqm}); Use blurring effect or not {self.use_blur}")
        
        # text prompt (Do not use here)
        self.text_prompt = config.get('path', {}).get('text_prompt', None)
        self.stem_masking = config.get('augmentation', {}).get('masking', False)

        # read track_mapping.csv, build track_id -> onset json filename mapping
        mapping_path = '/PATH/TO/PROJECT/track_mapping.csv'
        self.trackid2json = self._build_trackid2json(mapping_path)
        
        print(f'| MultiSource_Slakh_Waveform_Dataset Length:{len(self.data)} | Epoch Length: {self.total_len}')
    
    def get_mel_from_waveform(self, waveform):
        """
        for compatibility, return an empty mel spectrogram
        because Stable Audio VAE does not need mel spectrogram
        """
        # return an empty mel spectrogram, shape (64, 1024)
        return np.zeros((64, 1024), dtype=np.float32)
    
    def get_mel(self, filename, mix_filename=None, frame_offset=0):
        """
        override get_mel method, only return waveform, not calculate mel
        """
        # read waveform
        waveform = self.read_wav(filename, frame_offset)
        
        # for compatibility, return an empty mel spectrogram
        mel = self.get_mel_from_waveform(waveform)
        
        return waveform, mel
    
    def __getitem__(self, index):
        """
        override __getitem__ method, only return waveform data
        """
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]

        index, frame_offset = self.get_index_offset(f)

        stems_list = self.config["path"]["stems"]
        # check if it is 5 stems mode
        if stems_list == ['kick', 'snare', 'toms', 'hi_hats', 'cymbals']:
            # first read all 9 stems
            all_stem_names = ['crash', 'hh_closed', 'hh_open', 'hi_tom', 'kick', 'low_tom', 'mid_tom', 'ride', 'snare']
            all_audio = {}
            first_valid_audio = None
            for stem in all_stem_names:
                stem_path = os.path.join(f["wav_path"], stem + ".wav")
                if os.path.exists(stem_path):
                    audio, _ = self.get_mel(stem_path, None, frame_offset)
                    if first_valid_audio is None:
                        first_valid_audio = audio
                else:
                    audio = np.zeros_like(first_valid_audio) if first_valid_audio is not None else None
                all_audio[stem] = audio
            
            # merge into 5 stems
            audio_list = [
                all_audio['kick'][np.newaxis, :] if all_audio['kick'] is not None else np.zeros_like(first_valid_audio)[np.newaxis, :],
                all_audio['snare'][np.newaxis, :] if all_audio['snare'] is not None else np.zeros_like(first_valid_audio)[np.newaxis, :],
                ( (all_audio['hi_tom'] if all_audio['hi_tom'] is not None else 0)
                + (all_audio['mid_tom'] if all_audio['mid_tom'] is not None else 0)
                + (all_audio['low_tom'] if all_audio['low_tom'] is not None else 0) )[np.newaxis, :],
                ( (all_audio['hh_closed'] if all_audio['hh_closed'] is not None else 0)
                + (all_audio['hh_open'] if all_audio['hh_open'] is not None else 0) )[np.newaxis, :],
                ( (all_audio['crash'] if all_audio['crash'] is not None else 0)
                + (all_audio['ride'] if all_audio['ride'] is not None else 0) )[np.newaxis, :],
            ]
        else:
            # original 9 stems logic
            audio_list = []
            first_valid_audio = None
            stems_found = 0
            stems_total = len(self.config["path"]["stems"])
            for stem in self.config["path"]["stems"]:
                stem_path = os.path.join(f["wav_path"], stem + ".wav")
                if os.path.exists(stem_path):
                    audio, _ = self.get_mel(stem_path, None, frame_offset)
                    if first_valid_audio is None:
                        first_valid_audio = audio
                    stems_found += 1
                else:
                    if first_valid_audio is not None:
                        audio = np.zeros_like(first_valid_audio)
                    else:
                        audio = None
                if audio is not None:
                    audio_list.append(audio[np.newaxis, :])
            while len(audio_list) < stems_total:
                if first_valid_audio is not None:
                    audio_list.append(np.zeros_like(first_valid_audio)[np.newaxis, :])
                else:
                    raise RuntimeError(f"All stems missing for sample: {f['wav_path']}")

        if self.stem_masking and self.train:
            # simplified masking, only process audio_list
            audio_list = self.mask_audio_channels_waveform(audio_list)

        # construct dict
        data_dict['fname'] = f['wav_path'].split('/')[-1]+"_from_"+str(int(frame_offset))
        data_dict['waveform_stems'] = np.concatenate(audio_list, axis=0)
        
        # for compatibility, create empty mel data
        data_dict['fbank_stems'] = np.zeros((len(audio_list), 64, 1024), dtype=np.float32)

        data_dict['waveform'] = np.clip(np.sum(data_dict['waveform_stems'], axis=0), -1, 1)
        data_dict['fbank'] = np.zeros((64, 1024), dtype=np.float32)  # empty mel spectrogram

        # ====== add onset_pianoroll ======
        # 1. get track id
        track_id = data_dict['fname'].split('_')[0]  # e.g. Track00001
        # 2. find corresponding json filename
        json_file = self.trackid2json.get(track_id, None)
        # use path from config, if not set, use default path
        midi_onset_dir = getattr(self, 'onset_data_path', '/PATH/TO/PROJECT/midi_onset_gt')
        stem_names = self.config['path']['stems']  # ['kick', 'snare', ...]
        num_stems = len(stem_names)
        target_length = self.config['preprocessing']['mel']['target_length']
        hop_length = self.config['preprocessing']['stft']['hop_length']
        sampling_rate = self.config['preprocessing']['audio']['sampling_rate']

        pianoroll = np.zeros((num_stems, target_length), dtype=np.float32)
        if json_file is not None:
            json_path = os.path.join(midi_onset_dir, json_file)
            if os.path.exists(json_path):
                with open(json_path, 'r') as f_json:
                    onset_dict = json.load(f_json)
                # 3. convert onset list of each stem to pianoroll
                for i, stem in enumerate(stem_names):
                    json_keys = STEM_JSON_MAP.get(stem, [])
                    for key in json_keys:
                        if key in onset_dict:
                            for onset_sec in onset_dict[key]:
                                frame = int(float(onset_sec) * sampling_rate / hop_length)
                                if 0 <= frame < target_length:
                                    pianoroll[i, frame] = 1.0
        data_dict['onset_pianoroll'] = pianoroll
        # ==================================

        if self.text_prompt is not None:
           data_dict["text"] = self.text_prompt

        return data_dict
    
    def mask_audio_channels_waveform(self, audio_list):
        """
        simplified masking, only process waveform data
        """
        if not self.train:
            return audio_list
        
        # randomly mask some stems
        num_stems = len(audio_list)
        mask_prob = 0.3  # 30% probability to mask a stem
        
        for i in range(num_stems):
            if np.random.random() < mask_prob:
                # set stem to silence
                audio_list[i] = np.zeros_like(audio_list[i])
        
        return audio_list