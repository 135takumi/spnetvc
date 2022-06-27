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
from model import SplitterVC, LatentDiscriminator


def save_checkpoint(filepath, model, optimizer, epoch):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, filepath)


def train(model, clf, optimizer, train_loader, writer, epoch, debug, cts_atr):
    clf.train()

    running_loss = 0
    counter = 1

    for x_t, _, label in train_loader:
        x_t = x_t.to(hp.device)
        label = label.to(hp.device)

        with torch.no_grad():
            if cts_atr == 'cts':
                z, _, _ = model.cts_encode(x_t)
            elif cts_atr == 'atr':
                z, _, _ = model.atr_encode(x_t)
            else:
                sys.exit(1)

        clf_result = clf(z)

        loss = f.cross_entropy(clf_result, label)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        running_loss += loss.item()

        counter += 1

        if counter > 3 and debug == True:
            break

    denominator = len(train_loader)

    writer.add_scalar(("train/" + cts_atr + "/model"), running_loss / denominator, epoch)


def valid(model, clf, valid_loader, writer, epoch, debug, cts_atr):
    model.eval()

    running_loss = 0
    running_accu = 0
    counter = 1

    for x_t, _, label in valid_loader:
        x_t = x_t.to(hp.device)
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

        running_loss += loss.item()
        running_accu += accu.item()

        counter += 1

        if counter > 3 and debug == True:
            break

    denominator = len(valid_loader)

    writer.add_scalar(("valid/" + cts_atr + "/model"), running_loss / denominator, epoch)
    writer.add_scalar(("valid/" + cts_atr + "/accu"), running_accu / denominator, epoch)


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
    model = SplitterVC(hp.seen_speaker_num, hp.emb_num, hp.emb_dim).to(hp.device)
    model.load_state_dict(torch.load(hp.tng_result_dir / args.exp_name / "spnetvc" / args.weight,
                          map_location=hp.device)["model"])
    model.eval()

    # z_clfの作成
    cts_z_clf = LatentDiscriminator(hp.seen_speaker_num).to(hp.device)
    atr_z_clf = LatentDiscriminator(hp.seen_speaker_num).to(hp.device)

    cts_z_clf_optimizer = torch.optim.Adam(cts_z_clf.parameters(), lr=0.0002)
    atr_z_clf_optimizer = torch.optim.Adam(atr_z_clf.parameters(), lr=0.0002)

    with open(hp.session_dir / "seen_speaker.json", 'r') as f:
        speaker_dict = json.load(f)
    with open(hp.session_dir / "mcep_statistics.json", 'r') as f:
        mcep_dict = json.load(f)

    train_data = AudioDataset(hp.tng_data_dir, speaker_dict, mcep_dict, seq_len=hp.seq_len)
    valid_data = AudioDataset(hp.val_data_dir, speaker_dict, mcep_dict, seq_len=hp.seq_len)
    train_loader = DataLoader(train_data, batch_size=hp.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=hp.batch_size, shuffle=True)

    save_dir = hp.tng_result_dir / args.exp_name / "z_clf"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, hp.clf_epochs+1):
        train(model, cts_z_clf, cts_z_clf_optimizer, train_loader, writer, epoch, hp.debug,
              cts_atr='cts')
        train(model, atr_z_clf, atr_z_clf_optimizer, train_loader, writer, epoch, hp.debug,
              cts_atr='atr')
        valid(model, cts_z_clf, valid_loader, writer, epoch, hp.debug, cts_atr='cts')
        valid(model, atr_z_clf, valid_loader, writer, epoch, hp.debug, cts_atr='atr')

        if epoch == hp.clf_epochs:
            cts_z_clf_save_path = save_dir / "CTS_CLF-latest.pth"
            atr_z_clf_save_path = save_dir / "ATR_CLF-latest.pth"
            save_checkpoint(cts_z_clf_save_path, cts_z_clf, cts_z_clf_optimizer, epoch)
            save_checkpoint(atr_z_clf_save_path, atr_z_clf, atr_z_clf_optimizer, epoch)
        elif epoch % hp.save_interval == 0:
            cts_z_clf_save_path = save_dir / f"CTS_CLF-{epoch:03}.pth"
            atr_z_clf_save_path = save_dir / f"ATR_CLF-{epoch:03}.pth"
            save_checkpoint(cts_z_clf_save_path, cts_z_clf, cts_z_clf_optimizer, epoch)
            save_checkpoint(atr_z_clf_save_path, atr_z_clf, atr_z_clf_optimizer, epoch)

    list_c_data = []
    list_a_data = []
    list_label = []

    for i in range(4):
        for x_t, _, label in valid_loader:
            x_t = x_t.to(hp.device)
            label = label.to(hp.device)

            with torch.no_grad():
                cts_z, _, _ = model.cts_encode(x_t)
                atr_z, _, _ = model.atr_encode(x_t)

            list_label.append(label)
            list_c_data.append(cts_z)
            list_a_data.append(atr_z)

    label_data = torch.cat(list_label, 0)
    data_cts = torch.cat(list_c_data, 0)
    data_atr = torch.cat(list_a_data, 0)

    writer.add_embedding(data_cts, label_data,tag='cts')
    writer.add_embedding(data_atr, label_data,tag='atr')

    writer.flush()
    writer.close()

if __name__ == '__main__':
    main()
