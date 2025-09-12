import os, ast, torch, torchaudio, random
import pandas as pd
from torch.utils.data import Dataset
from MyModel.utils.utils import *

import pdb

LANG2ID = load_json('lang2id.json')

class EELDDataset(Dataset):
    def __init__(self, folder_path, noise_path, rir_path, mode='train', add_rir=False, add_noise=False, snr_db=[5,10,15,20]):
        
        assert (mode in ['train', 'dev', 'test'])
        self.csv_path = os.path.join(folder_path, f'{mode}.csv')
        self.data = pd.read_csv(self.csv_path)
        self.data = self.data[self.data['duration']<20]
        self.data = self.data.dropna().reset_index(drop=True)
        self.add_noise = add_noise
        self.add_rir = add_rir
        self.snr_db = snr_db
        
        with open(noise_path, 'r', encoding='utf-8') as f:
            self.noise_files = [line.strip() for line in f if line.strip()]
            
        with open(rir_path, 'r', encoding='utf-8') as f:
            self.rir_files = [line.strip() for line in f if line.strip()]


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx]['file path']
        audio, sr = torchaudio.load(audio_path)
        assert (sr == 16000)
        audio = normalize_audio(audio)
        
        if self.add_noise:
            if random.random() > 0.5:
                num_frames = audio.shape[-1]
                noise_path = random.choice(self.noise_files)
                info = torchaudio.info(noise_path)
                total_frames = info.num_frames
                assert total_frames > num_frames
                max_offset = total_frames - num_frames
                frame_offset = random.randint(0, max_offset)
                noise, _ = torchaudio.load(noise_path, frame_offset=frame_offset, num_frames=num_frames)
                audio = add_noise(audio, noise, snr_db=random.choice(self.snr_db))
            
        if self.add_rir:
            if random.random() > 0.5:
                rir_path = random.choice(self.rir_files)
                rir, _ = torchaudio.load(rir_path)
                audio = apply_rir(audio, rir)

        l_flag = ast.literal_eval(self.data.iloc[idx]['language flag'])
        v_flag = ast.literal_eval(self.data.iloc[idx]['vad flag'])

        mask = torch.zeros(compute_output_length(audio.shape[-1]), dtype=torch.long)
        mask = apply_lang_segments(l_flag, mask, LANG2ID, return_embedding=True)
        mask = apply_lang_segments(v_flag, mask, LANG2ID, return_embedding=True)
            

        audio = audio.squeeze()
        mask = relabel_mask(mask)
            
        mask = mask.squeeze()
        
        return audio, mask
    
class DISPLACEDataset(Dataset):
    def __init__(self, folder_path, base_audio_dir='/your/path/to/displacedataset/samples', official_split='Dev', mode='train'):
        
        assert (mode in ['train', 'dev', 'test'])
        self.csv_path = os.path.join(folder_path, f'{mode}.csv')
        self.data = pd.read_csv(self.csv_path)
        self.data = self.data.dropna().reset_index(drop=True)
        self.mode = mode
        self.base_audio_dir = base_audio_dir
        self.official_split = official_split

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio_path = os.path.join(self.base_audio_dir, self.official_split, 'Track2_LD', self.data.iloc[idx]['filename'])
        start = self.data.iloc[idx]['start']
        end = self.data.iloc[idx]['end']
        num_frames = end - start
        audio, sr = torchaudio.load(audio_path, frame_offset=start, num_frames=num_frames)
        audio = normalize_audio(audio)
        assert sr == 16000
        
        flag = ast.literal_eval(self.data.iloc[idx]['flag'])
        v_flag = ast.literal_eval(self.data.iloc[idx]['vad flag'])
        for lang in flag:
            for seg in flag[lang]:
                seg['start'] -= start
                seg['end'] -= start
        mask = torch.zeros(compute_output_length(audio.shape[-1]), dtype=torch.long)
        mask = apply_lang_segments_ft(flag, mask, return_embedding=True)
        mask = apply_lang_segments_ft_vad(v_flag, mask, return_embedding=True)
        mask[mask >= 2] = 2 # Ignore overlap
        
        audio = audio.squeeze()
        mask = mask.squeeze()
        
        return audio, mask
    
class AfricanSoapDataset(Dataset):
    def __init__(self, folder_path, base_audio_dir='/your/path/to/africansoapdataset/samples', mode='train'):
        
        assert (mode in ['train', 'dev', 'test'])
        self.csv_path = os.path.join(folder_path, f'{mode}.csv')
        self.data = pd.read_csv(self.csv_path)
        self.data = self.data.dropna().reset_index(drop=True)
        self.base_audio_dir = base_audio_dir

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        langpair = self.data.iloc[idx]['language pair']
        audio_path = os.path.join(self.base_audio_dir, langpair, 'audio', self.data.iloc[idx]['audio'])
        audio, sr = torchaudio.load(audio_path)
        audio = normalize_audio(audio)
        assert sr == 16000
        
        flag = ast.literal_eval(self.data.iloc[idx]['flag'])
        v_flag = ast.literal_eval(self.data.iloc[idx]['vad flag'])
        mask = torch.zeros(compute_output_length(audio.shape[-1]), dtype=torch.long)
        mask = apply_lang_segments_ft(flag, mask, return_embedding=True)
        mask = apply_lang_segments_ft_vad(v_flag, mask, return_embedding=True)
        
        audio = audio.squeeze()
        mask = mask.squeeze()
        
        return audio, mask
    