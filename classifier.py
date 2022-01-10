import argparse
import json
import os


import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_


import hyper_parameters as hp
from data import AudioDataset
from model import Classifier


def save_checkpoint(filepath, model, optimizer, epoch):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(state, filepath)


def train(model, optimizer, train_loader, writer, epoch, debug):
    model.train()

    running_loss = 0
    counter = 1

    for x_t, _, label in train_loader:
        x_t = x_t.to(hp.device)
        label = label.to(hp.device)

        clf_result = model(x_t)

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
    writer.add_scalar("train/model", running_loss / denominator, epoch)


def valid(model, valid_loader, writer, epoch, debug):
    model.eval()

    running_loss = 0
    running_accu = 0
    counter = 1

    for x_t, _, label in valid_loader:
        x_t = x_t.to(hp.device)
        label = label.to(hp.device)


        with torch.no_grad():
            clf_result = model(x_t)

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

    writer.add_scalar("valid/model", running_loss / denominator, epoch)
    writer.add_scalar("valid/accu", running_accu / denominator, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=hp.clf_name)
    parser.add_argument("--debug", type=bool, default=hp.debug)

    args = parser.parse_args()

    # ロガーの作成
    writer = SummaryWriter(hp.session_dir / "log" / "clf" / args.model_name)

    # モデル
    model = Classifier(hp.seen_speaker_num).to(hp.device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with open(hp.session_dir / "seen_speaker.json", 'r') as f:
        speaker_dict = json.load(f)

    with open(hp.session_dir / "mcep_statistics.json", 'r') as f:
        mcep_dict = json.load(f)

    # Create data loaders
    train_data = AudioDataset(hp.tng_data_dir, speaker_dict, mcep_dict, seq_len=hp.seq_len)
    valid_data = AudioDataset(hp.val_data_dir, speaker_dict, mcep_dict, seq_len=hp.seq_len)
    train_loader = DataLoader(train_data, batch_size=hp.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=hp.batch_size, shuffle=True)

    save_dir = hp.tng_result_dir / args.model_name
    os.makedirs(save_dir, exist_ok=True)

    # 学習
    for epoch in range(1, hp.clf_epochs + 1):

        train(model, optimizer, train_loader, writer, epoch, args.debug)
        valid(model, valid_loader, writer, epoch, args.debug)

        if epoch == hp.clf_epochs:
            save_path = save_dir / "CLF-latest.pth"
            save_checkpoint(save_path, model, optimizer, epoch)
        elif epoch % hp.save_interval == 0:
            save_path = save_dir / f"CLF-{epoch:03}.pth"
            save_checkpoint(save_path, model, optimizer, epoch)

if __name__ == "__main__":
    main()
