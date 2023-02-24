import argparse
import json
import os
import sys


import torch.utils.data
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_


import hyper_parameters as hp
from data import AudioDataset
from model import LatentDiscriminator
from utils import load_trained_model, save_checkpoint


def train(model, clf, optimizer, loader, writer, epoch, debug, cts_atr):
    clf.train()

    loss_sum = 0
    counter = 1

    for x_t, _, label in loader:
        x_t = x_t.to(hp.device)
        label = label.to(hp.device)

        with torch.no_grad():
            if cts_atr == 'cts':
                z, _, _ = model.cts_encode(x_t)
            elif cts_atr == 'atr':
                z, _, _ = model.atr_encode(x_t)
            else:
                print("引数「cts_atr」には文字列\"cts\"か\"atr\"を入れてください")
                sys.exit(1)

        clf_result = clf(z)

        loss = f.cross_entropy(clf_result, label)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        loss_sum += loss.item()

        counter += 1

        if counter > 3 and debug:
            break

    batch_num = len(loader)
    writer.add_scalar(("train/" + cts_atr + "/model"), loss_sum / batch_num, epoch)


def valid(model, clf, loader, writer, epoch, debug, cts_atr):
    model.eval()

    loss_sum = 0
    accu_sum = 0
    counter      = 1

    for x_t, _, label in loader:
        x_t   = x_t.to(hp.device)
        label = label.to(hp.device)


        with torch.no_grad():
            if cts_atr == 'cts':
                z, _, _ = model.cts_encode(x_t)
            elif cts_atr == 'atr':
                z, _, _ = model.atr_encode(x_t)
            else:
                sys.exit(1)

            clf_result = clf(z)

        estimated_label = f.softmax(clf_result, dim=1)
        loss = f.cross_entropy(clf_result, label)
        accu = (estimated_label.max(1)[1] == label)
        accu = torch.mean(accu.float())

        loss_sum += loss.item()
        accu_sum += accu.item()

        counter += 1

        if counter > 3 and debug:
            break

    batch_num = len(loader)
    writer.add_scalar(("valid/" + cts_atr + "/model"), loss_sum / batch_num, epoch)
    writer.add_scalar(("valid/" + cts_atr + "/accu"),  accu_sum / batch_num, epoch)


def main():
    """
    学習したモデルを使って声質変換を行う
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=hp.exp_name)
    parser.add_argument("--weight", type=str, default="SPNETVC-latest.pth")

    args = parser.parse_args()
    writer = SummaryWriter(hp.session_dir/ "log" / "z_check" / args.exp_name)

    # モデルの読み込み
    model = load_trained_model(args)

    # z_clfの作成
    c_z_clf = LatentDiscriminator(hp.seen_spkr_num).to(hp.device)
    a_z_clf = LatentDiscriminator(hp.seen_spkr_num).to(hp.device)

    c_z_clf_optimizer = torch.optim.Adam(c_z_clf.parameters(), lr=0.0002)
    a_z_clf_optimizer = torch.optim.Adam(a_z_clf.parameters(), lr=0.0002)

    with open(hp.session_dir / "seen_speaker.json", 'r') as f:
        spkr_dct = json.load(f)
    with open(hp.session_dir / "mcep_statistics.json", 'r') as f:
        m_statistics_dct = json.load(f)

    t_data = AudioDataset(hp.tng_data_dir, spkr_dct, m_statistics_dct,
                          seq_len=hp.seq_len)
    v_data = AudioDataset(hp.val_data_dir, spkr_dct, m_statistics_dct,
                          seq_len=hp.seq_len)
    t_loader = DataLoader(t_data, batch_size=hp.batch_size, shuffle=True)
    v_loader = DataLoader(v_data, batch_size=hp.batch_size, shuffle=True)

    save_dir = hp.tng_result_dir / args.exp_name / "z_clf"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, hp.clf_epochs+1):
        train(model, c_z_clf, c_z_clf_optimizer, t_loader, writer, epoch,
              hp.debug, cts_atr='cts')
        train(model, a_z_clf, a_z_clf_optimizer, t_loader, writer, epoch,
              hp.debug, cts_atr='atr')
        valid(model, c_z_clf, v_loader, writer, epoch, hp.debug,
              cts_atr='cts')
        valid(model, a_z_clf, v_loader, writer, epoch, hp.debug,
              cts_atr='atr')

        if epoch == hp.clf_epochs:
            c_z_clf_save_path = save_dir / "CTS_CLF-latest.pth"
            a_z_clf_save_path = save_dir / "ATR_CLF-latest.pth"
            save_checkpoint(c_z_clf_save_path, c_z_clf, c_z_clf_optimizer,
                            epoch)
            save_checkpoint(a_z_clf_save_path, a_z_clf, a_z_clf_optimizer,
                            epoch)
        elif epoch % hp.save_interval == 0:
            c_z_clf_save_path = save_dir / f"CTS_CLF-{epoch:03}.pth"
            a_z_clf_save_path = save_dir / f"ATR_CLF-{epoch:03}.pth"
            save_checkpoint(c_z_clf_save_path, c_z_clf, c_z_clf_optimizer,
                            epoch)
            save_checkpoint(a_z_clf_save_path, a_z_clf, a_z_clf_optimizer,
                            epoch)

    c_data_lst   = []
    a_data_lst   = []
    label_lst    = []

    for i in range(4):
        for x_t, _, label in v_loader:
            x_t   = x_t.to(hp.device)
            label = label.to(hp.device)

            with torch.no_grad():
                c_z, _, _ = model.cts_encode(x_t)
                a_z, _, _ = model.atr_encode(x_t)

            label_lst.append(label)
            c_data_lst.append(c_z)
            a_data_lst.append(a_z)

    label_data = torch.cat(label_lst)
    data_c     = torch.cat(c_data_lst)
    data_a     = torch.cat(a_data_lst)

    writer.add_embedding(data_c,   label_data, tag='cts')
    writer.add_embedding(data_a,   label_data, tag='atr')

    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()
