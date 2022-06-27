import argparse
import json
import os
import sys

import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import hyper_parameters as hp
from data import AudioDataset
from model import SplitterVC, Classifier


def save_checkpoint(filepath, model, optimizer, epoch):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(state, filepath)


def kl_div(mean, logvar):
    kld = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    return kld


def get_lambda(ld_lambda, n_iter, lambda_schedule):
    if lambda_schedule == 0:
        return ld_lambda
    else:
        return ld_lambda * float(min(n_iter, lambda_schedule)) / lambda_schedule


def calc_ld_loss(model, x_t, label, optimizer, cts_atr="cts"):
    # ldの学習

    if cts_atr == "cts":
        with torch.no_grad():
            z, _, _ = model.cts_encode(x_t)
        discriminator_result = model.cts_discriminate(z)
    elif cts_atr == "atr":
        with torch.no_grad():
            z, _, _ = model.atr_encode(x_t)
        discriminator_result = model.atr_discriminate(z)
    else:
        sys.exit(1)

    loss = f.cross_entropy(discriminator_result, label)

    optimizer.zero_grad()
    loss.backward()
    if cts_atr == "cts":
        clip_grad_norm_(model.cts_ld.parameters(), 5)
    elif cts_atr == "atr":
        clip_grad_norm_(model.atr_ld.parameters(), 5)
    else:
        sys.exit(1)
    optimizer.step()


def train(model, vae_optimizer, cts_ld_optimizer, atr_ld_optimizer, vae_train_loader,
          cts_ld_train_loader, atr_ld_trian_loader, writer, epoch, debug):
    model.train()

    running_loss = 0
    running_loss_rec = 0
    running_loss_mse_atr = 0
    running_loss_atr_kl = 0
    running_loss_cts_ld = 0
    running_loss_atr_ld = 0
    running_loss_commitment = 0
    running_loss_emb = 0
    counter = 1

    for (vae_x_t1, vae_x_t2, vae_label), (cts_ld_x_t, _, cts_ld_label), \
        (atr_ld_x_t, _, atr_ld_label) \
            in zip(vae_train_loader, cts_ld_train_loader, atr_ld_trian_loader):

        vae_x_t1, vae_x_t2, vae_label = vae_x_t1.to(hp.device), vae_x_t2.to(hp.device), \
                                        vae_label.to(hp.device)
        cts_ld_x_t, cts_ld_label = cts_ld_x_t.to(hp.device), cts_ld_label.to(hp.device)
        atr_ld_x_t, atr_ld_label = atr_ld_x_t.to(hp.device), atr_ld_label.to(hp.device)

        # cts_ldの学習
        calc_ld_loss(model, cts_ld_x_t, cts_ld_label, cts_ld_optimizer, cts_atr="cts")

        # atr_ldの学習
        calc_ld_loss(model, atr_ld_x_t, atr_ld_label, atr_ld_optimizer, cts_atr="atr")

        # vaeの学習
        total_iter = (epoch - 1) * len(vae_train_loader) + counter

        cts_z, commitment_loss, emb_loss = model.cts_encode(vae_x_t1)
        atr_z1, atr_mean, atr_logvar = model.atr_encode(vae_x_t1)
        atr_z2, _, _ = model.atr_encode(vae_x_t2)
        x_recon_t = model.decode(cts_z, atr_z1)

        cts_discriminator_result = model.cts_discriminate(cts_z)
        atr_discriminator_result = model.atr_discriminate(atr_z1)

        rec_loss = 0.5 * f.mse_loss(x_recon_t, vae_x_t1)
        mse_atr_loss = 0.5 * f.mse_loss(atr_z1, atr_z2)
        atr_kl_loss = kl_div(atr_mean, atr_logvar)
        cts_ld_loss = f.cross_entropy(cts_discriminator_result, vae_label)
        cts_lambda = get_lambda(hp.cts_ld_lambda, total_iter, hp.lambda_schedule)
        atr_ld_loss = f.cross_entropy(atr_discriminator_result, vae_label)
        atr_lambda = get_lambda(hp.atr_ld_lambda, total_iter, hp.lambda_schedule)

        # vaeのloss
        loss = hp.rec_lambda * rec_loss + hp.mse_atr_lambda * mse_atr_loss \
               + hp.atr_kl_lambda * atr_kl_loss + atr_lambda * atr_ld_loss \
               - cts_lambda * cts_ld_loss + hp.commitment_lambda * commitment_loss \
               + hp.emb_lambda * emb_loss

        vae_optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.vae.parameters(), 5)
        vae_optimizer.step()

        running_loss += loss.item()
        running_loss_rec += rec_loss.item()
        running_loss_mse_atr += mse_atr_loss.item()
        running_loss_atr_kl += atr_kl_loss.item()
        running_loss_cts_ld += cts_ld_loss.item()
        running_loss_atr_ld += atr_ld_loss.item()
        running_loss_commitment += commitment_loss.item()
        running_loss_emb += emb_loss.item()

        counter += 1

        if counter > 3 and debug == True:
            break

    denominator = len(vae_train_loader)
    writer.add_scalar("train/model", running_loss / denominator, epoch)
    writer.add_scalar("train/reconstract", running_loss_rec / denominator, epoch)
    writer.add_scalar("train/mse_atr", running_loss_mse_atr / denominator, epoch)
    writer.add_scalar("train/atr_kl", running_loss_atr_kl / denominator, epoch)
    writer.add_scalar("train/cts_ld", running_loss_cts_ld / denominator, epoch)
    writer.add_scalar("train/atr_ld", running_loss_atr_ld / denominator, epoch)
    writer.add_scalar("train/commitment", running_loss_commitment / denominator, epoch)
    writer.add_scalar("train/emb", running_loss_emb / denominator, epoch)


def valid(model, clf, valid_loader, writer, epoch, debug):
    model.eval()

    running_loss = 0
    running_accu = 0
    running_loss_rec = 0
    running_loss_mse_atr = 0
    running_loss_atr_kl = 0
    running_loss_cts_ld = 0
    running_loss_atr_ld = 0
    running_loss_commitment = 0
    running_loss_emb = 0
    counter = 1

    with torch.no_grad():
        for x_t1, x_t2, label in valid_loader:
            x_t1, x_t2, label = x_t1.to(hp.device), x_t2.to(hp.device), label.to(hp.device)

            cts_z, commitment_loss, emb_loss = model.cts_encode(x_t1)
            atr_z1, atr_mean, atr_logvar = model.atr_encode(x_t1)
            atr_z2, _, _ = model.atr_encode(x_t2)

            cts_discriminator_result = model.cts_discriminate(cts_z)
            atr_discriminator_result = model.atr_discriminate(atr_z1)
            x_recon_t = model.decode(cts_z, atr_z1)

            clf_result = clf(x_recon_t)
            estimated_label = f.softmax(clf_result, dim=1)
            accu = (estimated_label.max(1)[1] == label)
            accu = torch.mean(accu.float())

            rec_loss = 0.5 * f.mse_loss(x_recon_t, x_t1)
            mse_atr_loss = 0.5 * f.mse_loss(atr_z1, atr_z2)
            atr_kl_loss = kl_div(atr_mean, atr_logvar)
            cts_ld_loss = f.cross_entropy(cts_discriminator_result, label)
            atr_ld_loss = f.cross_entropy(atr_discriminator_result, label)

            # vaeのloss
            loss = hp.rec_lambda * rec_loss + hp.mse_atr_lambda * mse_atr_loss \
                   + hp.atr_kl_lambda * atr_kl_loss - hp.cts_ld_lambda * cts_ld_loss \
                   + hp.atr_ld_lambda * atr_ld_loss + hp.commitment_lambda * commitment_loss \
                   + hp.emb_lambda * emb_loss

            running_loss += loss.item()
            running_accu += accu.item()
            running_loss_rec += rec_loss.item()
            running_loss_mse_atr += mse_atr_loss.item()
            running_loss_atr_kl += atr_kl_loss.item()
            running_loss_cts_ld += cts_ld_loss.item()
            running_loss_atr_ld += atr_ld_loss.item()
            running_loss_commitment += commitment_loss.item()
            running_loss_emb += emb_loss.item()

            counter += 1

            if counter > 3 and debug == True:
                break

    denominator = len(valid_loader)
    writer.add_scalar("valid/model", running_loss / denominator, epoch)
    writer.add_scalar("valid/accu", running_accu / denominator, epoch)
    writer.add_scalar("valid/reconstract", running_loss_rec / denominator, epoch)
    writer.add_scalar("valid/mse_atr", running_loss_mse_atr / denominator, epoch)
    writer.add_scalar("valid/atr_kl", running_loss_atr_kl / denominator, epoch)
    writer.add_scalar("valid/cts_ld", running_loss_cts_ld / denominator, epoch)
    writer.add_scalar("valid/commitment", running_loss_cts_ld / denominator, epoch)
    writer.add_scalar("valid/emb", running_loss_cts_ld / denominator, epoch)
    writer.add_scalar("valid/atr_ld", running_loss_atr_ld / denominator, epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=hp.exp_name)
    parser.add_argument("--debug", type=bool, default=hp.debug)

    args = parser.parse_args()

    # ロガーの作成
    writer = SummaryWriter(hp.session_dir / "log" / "spnetvc" / args.exp_name)

    # モデル
    model = SplitterVC(hp.seen_speaker_num, hp.emb_num, hp.emb_dim).to(hp.device)

    clf = Classifier(hp.seen_speaker_num).to(hp.device)
    clf.load_state_dict(torch.load(hp.tng_result_dir / hp.clf_name / "CLF-latest.pth",
                                   map_location=hp.device)["model"])

    # optimizer
    vae_optimizer = torch.optim.Adam(model.vae.parameters(), lr=hp.lr)
    cts_ld_optimizer = torch.optim.Adam(model.cts_ld.parameters(), lr=0.0002)
    atr_ld_optimizer = torch.optim.Adam(model.atr_ld.parameters(), lr=0.0002)

    with open(hp.session_dir / "seen_speaker.json", 'r') as f:
        speaker_dict = json.load(f)

    with open(hp.session_dir / "mcep_statistics.json", 'r') as f:
        mcep_dict = json.load(f)

    # Create data loaders
    train_data = AudioDataset(hp.tng_data_dir, speaker_dict, mcep_dict, seq_len=hp.seq_len)
    valid_data = AudioDataset(hp.val_data_dir, speaker_dict, mcep_dict, seq_len=hp.seq_len)

    vae_train_loader = DataLoader(train_data, batch_size=hp.batch_size, shuffle=True)
    cts_ld_train_loader = DataLoader(train_data, batch_size=hp.batch_size, shuffle=True)
    atr_ld_train_loader = DataLoader(train_data, batch_size=hp.batch_size, shuffle=True)

    valid_loader = DataLoader(valid_data, batch_size=hp.batch_size, shuffle=True)

    save_dir = hp.tng_result_dir / args.exp_name / "spnetvc"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(1, hp.spnetvc_epochs + 1)):
        train(model, vae_optimizer, cts_ld_optimizer, atr_ld_optimizer, vae_train_loader,
              cts_ld_train_loader, atr_ld_train_loader, writer, epoch, args.debug)
        valid(model, clf, valid_loader, writer, epoch, args.debug)

        if epoch == hp.spnetvc_epochs:
            save_path = save_dir / "VAEVC-latest.pth"
            save_checkpoint(save_path, model, vae_optimizer, epoch)
        elif epoch % hp.save_interval == 0:
            save_path = save_dir / f"VAEVC-{epoch:03}.pth"
            save_checkpoint(save_path, model, vae_optimizer, epoch)


if __name__ == "__main__":
    main()
