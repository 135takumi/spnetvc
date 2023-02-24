import os
import random

import numpy as np
import torch


from utils import mcep_normalize


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, spkr_dct, m_statistics_dct, seq_len=128):

        self.seq_len = seq_len
        self.spkr_dct = spkr_dct
        self.m_statistics_dct = {}

        # TODO: もっとエレガントな実装がありそう
        # spkr_name:mcep_meanをspkr_id:mcep_meanに変える
        for k, v in m_statistics_dct.items():
            spkr_id = self.spkr_dct[k]
            self.m_statistics_dct[spkr_id] = {
                "mean": np.array(v['mean'])[None, :],
                "std": np.array(v['std'])[None, :]
            }

        self.data = self.read_data(data_dir)


    def read_data(self, data_dir):
        data = []

        for spkr in os.listdir(data_dir):
            spkr_id  = self.spkr_dct[spkr]
            spkr_dir = data_dir / spkr

            for uttr in os.listdir(spkr_dir):
                uttr_dir = os.path.join(spkr_dir, uttr)

                if not os.path.isdir(uttr_dir):
                    continue

                mcep = np.load(os.path.join(uttr_dir, "mcep.npy"))
                mcep = mcep_normalize(mcep, spkr_id, self.m_statistics_dct)
                data.append((mcep, spkr_id))

        return data

    def __getitem__(self, index):
        mcep, spkr_id = self.data[index]

        last_head = np.shape(mcep)[0] - 1

        idx1 = random.randint(0, last_head)
        idx2 = random.randint(0, last_head)
        mcep1 = mcep[idx1]
        mcep2 = mcep[idx2]

        mcep1 = torch.from_numpy(mcep1).float()
        mcep2 = torch.from_numpy(mcep2).float()

        spkr_id = torch.from_numpy(np.array(spkr_id)).long()

        return mcep1, mcep2, spkr_id

    def __len__(self):
        return len(self.data)
