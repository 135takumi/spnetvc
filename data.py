import os
import random

import numpy as np
import torch


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, speaker_dict, mcep_dict, seq_len=128):

        self.seq_len = seq_len
        self.speaker_dict = speaker_dict

        self.mcep_dict = {}
        for k, v in mcep_dict.items():
            label = self.speaker_dict[k]
            self.mcep_dict[label] = {
                "mean": np.array(v['mean'])[None, :],
                "std": np.array(v['std'])[None, :]
            }

        self.data = self.read_data(data_dir)

    def mcep_normalize(self, mcep, label):
        speaker_dict = self.mcep_dict[label]
        mean, std = speaker_dict['mean'], speaker_dict['std']
        mcep = (mcep - mean) / std

        return mcep

    def read_data(self, data_dir):
        data = []

        for speaker in os.listdir(data_dir):
            speaker_label = self.speaker_dict[speaker]
            speaker_dir = data_dir / speaker

            for uttr in os.listdir(speaker_dir):
                uttr_dir = os.path.join(speaker_dir, uttr)

                if not os.path.isdir(uttr_dir):
                    continue

                mcep = np.load(os.path.join(uttr_dir, "mcep.npy"))
                mcep = self.mcep_normalize(mcep, speaker_label)
                data.append((mcep, speaker_label))

        return data

    def __getitem__(self, index):
        mcep, label = self.data[index]

        max_start = np.shape(mcep)[0] - 1
        idx1 = random.randint(0, max_start)
        idx2 = random.randint(0, max_start)
        mcep1 = mcep[idx1]
        mcep2 = mcep[idx2]

        mcep1 = torch.from_numpy(mcep1).float()
        mcep2 = torch.from_numpy(mcep2).float()

        label = torch.from_numpy(np.array(label)).long()

        return mcep1, mcep2, label

    def __len__(self):
        return len(self.data)
