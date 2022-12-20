import argparse
import json
import os
import pandas as pd


import matplotlib.pyplot as plt
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import hyper_parameters as hp
from data import AudioDataset
from model import SplitterVC


def main():
    """
    学習したモデルを使って声質変換を行う
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=hp.exp_name)
    parser.add_argument("--weight", type=str, default="VAEVC-latest.pth")

    args = parser.parse_args()
    writer = SummaryWriter(hp.session_dir/ "log" / "z_check" / args.exp_name)

    # モデルの読み込み
    model = SplitterVC(hp.seen_speaker_num).to(hp.device)
    model.load_state_dict(torch.load(hp.tng_result_dir / args.exp_name / "spnetvc" / args.weight,
                          map_location=hp.device)["model"])
    model.eval()

    with open(hp.session_dir / "seen_speaker.json", 'r') as f:
        speaker_dict = json.load(f)
    with open(hp.session_dir / "melsp_statistics.json", 'r') as f:
        melsp_dict = json.load(f)

    valid_data = AudioDataset(hp.val_data_dir, speaker_dict, melsp_dict, seq_len=hp.seq_len)
    valid_loader = DataLoader(valid_data, batch_size=hp.batch_size, shuffle=True)

    save_dir = hp.session_dir / 'img' / args.exp_name
    os.makedirs(save_dir, exist_ok=True)

    for i in range(7):
        for j, (x_t, _, label) in enumerate(valid_loader):
            x_t = x_t.to(hp.device)
            label = label.to(hp.device)

            if i == 0 and j < 5:
                with torch.no_grad():
                    _, cts_mean, _ = model.cts_encode(x_t)
                    atr_z, _, _ = model.atr_encode(x_t)
                    atr_z = torch.mean(atr_z, 0, keepdim=True)
                    reconst = model.decode(cts_mean, atr_z)

                plt.figure()

                plt.subplot(1, 2, 1)
                plt.imshow(x_t[0, 0].T.detach().cpu().numpy(), origin='lower')
                plt.xlabel('input')
                plt.colorbar()

                plt.subplot(1, 2, 2)
                plt.imshow(reconst[0,0].T.cpu(), origin='lower')
                plt.xlabel('reconst')
                plt.colorbar()

                #fig.colorbar()
                plt.savefig(save_dir / ('melsp_visualize'+str(j)+'.png'))


if __name__ == '__main__':
    main()
