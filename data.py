import os
import random

import numpy as np
import torch


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, speaker_dict, melsp_dict, seq_len=128):

        self.seq_len = seq_len
        self.speaker_dict = speaker_dict

        self.melsp_dict = {}
        for k, v in melsp_dict.items():
            label = self.speaker_dict[k]
            self.melsp_dict[label] = {
                "mean": torch.tensor(v['mean'], dtype=torch.float),
                "std": torch.tensor(v['std'], dtype=torch.float)
            }

        self.data = self.read_data(data_dir)

    def melsp_normalize(self, melsp, label):
        speaker_dict = self.melsp_dict[label]
        mean, std = speaker_dict['mean'], speaker_dict['std']
        melsp = (melsp - mean) / std

        return melsp

    def read_data(self, data_dir):
        data = []

        for speaker in os.listdir(data_dir):
            speaker_label = self.speaker_dict[speaker]
            speaker_dir = data_dir / speaker

            for uttr in os.listdir(speaker_dir):
                uttr_dir = speaker_dir / uttr

                if not os.path.isdir(uttr_dir):
                    continue

                melsp = torch.load(uttr_dir / "melsp.pt")
                melsp = self.melsp_normalize(melsp, speaker_label)
                data.append((melsp, speaker_label))

        return data

    def __getitem__(self, index):
        melsp, label = self.data[index]

        max_start = melsp.size()[0] - 1
        idx1 = random.randint(0, max_start)
        idx2 = random.randint(0, max_start)
        melsp1 = melsp[idx1]
        melsp2 = melsp[idx2]

        label = torch.from_numpy(np.array(label)).long()

        return melsp1, melsp2, label

    def __len__(self):
        return len(self.data)