import os
import torch
import torchaudio
from torch.utils.data import Dataset
from preprocessing import mel_spectrogram
import glob
import re


class DCASE_Dataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=mel_spectrogram, target_seconds=12):
        self.root_dir = root_dir
        self.transform = transform
        self.target_seconds = target_seconds
        
        if mode == 'train':
            self.file_list = glob.glob(os.path.join(root_dir, 'train', '**', '*.wav'), recursive=True)
        elif mode == 'test':
            self.file_list = glob.glob(os.path.join(root_dir, 'test', '**', '*.wav'), recursive=True)

        self.combination_label_map = self._create_combination_labels()
        self.num_classes = len(self.combination_label_map)

    def _get_label_from_path(self, audio_path):

        path_parts = audio_path.replace('\\', '/').split('/')
        
        machine_type_str = path_parts[-2]
        
        return machine_type_str

    def _create_combination_labels(self):
        all_files = glob.glob(os.path.join(self.root_dir, '**', '*.wav'), recursive=True)
        unique_labels = sorted(list(set([self._get_label_from_path(p) for p in all_files])))
        return {label: i for i, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]

        path_parts = audio_path.replace('\\', '/').split('/')
        machine_type_str = path_parts[-2]

        combination_label_str = self._get_label_from_path(audio_path)
        label = self.combination_label_map[combination_label_str]

        is_normal = "normal" in os.path.basename(audio_path)

        waveform, sr = torchaudio.load(audio_path)
        target_length = sr * self.target_seconds
        length = waveform.size(1)
        if length < target_length:
            waveform = torch.cat([waveform, torch.zeros(waveform.size(0), target_length - length)], dim=1)
        else:
            waveform = waveform[:, :target_length]

        spec = self.transform(waveform, sr) if self.transform else waveform
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)

        return spec.float(), torch.tensor(label, dtype=torch.long), is_normal, machine_type_str
