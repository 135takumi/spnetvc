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
from utils import save_checkpoint


def kl_div(mean, logvar):
    kld = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    return kld


def get_lambda(l, n_iter, schedule):
    """
    discriminatorのクロスエントロピーの重みを段階的に増やすための関数
    FaderNetworksの元論文では学習の安定化のため使用
    """
    if schedule == 0:
        return l
    else:
        return l * float(min(n_iter, schedule)) / schedule


def calc_ld_loss(model, x_t, label,  optimizer, cts_atr="cts"):
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
        print("引数「cts_atr」には文字列\"cts\"か\"atr\"を入れてください")
        sys.exit(1)

    loss = f.cross_entropy(discriminator_result, label)
    optimizer.zero_grad()
    loss.backward()

    if cts_atr == "cts":
        clip_grad_norm_(model.cts_ld.parameters(), 5)
    elif cts_atr == "atr":
        clip_grad_norm_(model.atr_ld.parameters(), 5)
    else:
        print("引数「cts_atr」には文字列\"cts\"か\"atr\"を入れてください")
        sys.exit(1)
    optimizer.step()


def train(model, vae_optimizer, cts_ld_optimizer, atr_ld_optimizer,
          vae_loader, cts_ld_loader, atr_ld_loader, writer, epoch, debug):
    model.train()

    loss_sum         = 0
    rec_loss_sum     = 0
    mse_atr_loss_sum = 0
    cts_kl_loss_sum  = 0
    atr_kl_loss_sum  = 0
    cts_ld_loss_sum  = 0
    atr_ld_loss_sum  = 0
    counter          = 1

    # 別々にloaderを用意しているのは, 各discriminatorとvaeを学習するときに
    # 別々のdataで学習したほうがいいと思ったから（もしかしたら一緒でもいいのかも）
    for (vae_x_t1, vae_x_t2, vae_label), (cts_ld_x_t, _, cts_ld_label), \
        (atr_ld_x_t, _, atr_ld_label)\
            in zip(vae_loader, cts_ld_loader, atr_ld_loader):

        vae_x_t1, vae_x_t2, vae_label = vae_x_t1.to(hp.device), \
                                        vae_x_t2.to(hp.device), \
                                        vae_label.to(hp.device)
        cts_ld_x_t, cts_ld_label      = cts_ld_x_t.to(hp.device), \
                                        cts_ld_label.to(hp.device)
        atr_ld_x_t, atr_ld_label      = atr_ld_x_t.to(hp.device), \
                                        atr_ld_label.to(hp.device)

        # cts_ldの学習
        calc_ld_loss(model, cts_ld_x_t, cts_ld_label, cts_ld_optimizer,
                     cts_atr="cts")

        # atr_ldの学習
        calc_ld_loss(model, atr_ld_x_t, atr_ld_label, atr_ld_optimizer,
                     cts_atr="atr")

        # vaeの学習
        total_iter = (epoch - 1) * len(vae_loader) + counter

        cts_z,  cts_mean, cts_logvar = model.cts_encode(vae_x_t1)
        atr_z1, atr_mean, atr_logvar = model.atr_encode(vae_x_t1)
        atr_z2, _,        _          = model.atr_encode(vae_x_t2)
        x_recon_t                    = model.decode(cts_z, atr_z1)

        cts_discriminator_result = model.cts_discriminate(cts_z)
        atr_discriminator_result = model.atr_discriminate(atr_z1)

        rec_loss     = 0.5 * f.mse_loss(x_recon_t, vae_x_t1)
        mse_atr_loss = 0.5 * f.mse_loss(atr_z1, atr_z2)
        cts_kl_loss  = kl_div(cts_mean, cts_logvar)
        atr_kl_loss  = kl_div(atr_mean, atr_logvar)
        cts_ld_loss  = f.cross_entropy(cts_discriminator_result, vae_label)
        cts_lambda   = get_lambda(hp.cts_ld_lambda, total_iter,
                                  hp.lambda_schedule)
        atr_ld_loss  = f.cross_entropy(atr_discriminator_result, vae_label)
        atr_lambda   = get_lambda(hp.atr_ld_lambda, total_iter,
                                  hp.lambda_schedule)

        # vaeのloss
        loss = hp.rec_lambda * rec_loss + hp.mse_atr_lambda * mse_atr_loss \
               + hp.cts_kl_lambda * cts_kl_loss \
               + hp.atr_kl_lambda * atr_kl_loss \
               - cts_lambda * cts_ld_loss + atr_lambda * atr_ld_loss

        vae_optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.vae.parameters(), 5)
        vae_optimizer.step()

        loss_sum         += loss.item()
        rec_loss_sum     += rec_loss.item()
        mse_atr_loss_sum += mse_atr_loss.item()
        cts_kl_loss_sum  += cts_kl_loss.item()
        atr_kl_loss_sum  += atr_kl_loss.item()
        cts_ld_loss_sum  += cts_ld_loss.item()
        atr_ld_loss_sum  += atr_ld_loss.item()

        counter += 1

        # debugモード時はiter数が3回を超えると強制的に次のepochへ行く
        if counter > 3 and debug:
            break

    batch_num = len(vae_loader)
    writer.add_scalar("train/model",       loss_sum / batch_num, epoch)
    writer.add_scalar("train/reconstruct", rec_loss_sum / batch_num, epoch)
    writer.add_scalar("train/mse_atr",     mse_atr_loss_sum / batch_num, epoch)
    writer.add_scalar("train/cts_kl",      cts_kl_loss_sum / batch_num, epoch)
    writer.add_scalar("train/atr_kl",      atr_kl_loss_sum / batch_num, epoch)
    writer.add_scalar("train/cts_ld",      cts_ld_loss_sum / batch_num, epoch)
    writer.add_scalar("train/atr_ld",      atr_ld_loss_sum / batch_num, epoch)


def valid(model, clf, loader, writer, epoch, debug):
    model.eval()

    loss_sum         = 0
    accu_sum         = 0
    rec_loss_sum     = 0
    mse_atr_loss_sum = 0
    cts_kl_loss_sum  = 0
    atr_kl_loss_sum  = 0
    cts_ld_loss_sum  = 0
    atr_ld_loss_sum  = 0
    counter          = 1

    with torch.no_grad():
        for x_t1, x_t2, label in loader:
            x_t1, x_t2, label = x_t1.to(hp.device), x_t2.to(hp.device), \
                                label.to(hp.device)

            cts_z,  cts_mean, cts_logvar = model.cts_encode(x_t1)
            atr_z1, atr_mean, atr_logvar = model.atr_encode(x_t1)
            atr_z2, _,        _          = model.atr_encode(x_t2)

            cts_discriminator_result = model.cts_discriminate(cts_z)
            atr_discriminator_result = model.atr_discriminate(atr_z1)
            x_recon_t = model.decode(cts_z, atr_z1)

            clf_result = clf(x_recon_t)
            estimated_label = f.softmax(clf_result, dim=1)
            accu = (estimated_label.max(1)[1] == label)
            accu = torch.mean(accu.float())

            rec_loss     = 0.5 * f.mse_loss(x_recon_t, x_t1)
            mse_atr_loss = 0.5 * f.mse_loss(atr_z1, atr_z2)
            cts_kl_loss  = kl_div(cts_mean, cts_logvar)
            atr_kl_loss  = kl_div(atr_mean, atr_logvar)
            cts_ld_loss  = f.cross_entropy(cts_discriminator_result, label)
            atr_ld_loss  = f.cross_entropy(atr_discriminator_result, label)

            # vaeのloss
            loss = hp.rec_lambda * rec_loss + hp.mse_atr_lambda * mse_atr_loss \
                   + hp.cts_kl_lambda * cts_kl_loss \
                   + hp.atr_kl_lambda * atr_kl_loss \
                   - hp.cts_ld_lambda * cts_ld_loss \
                   + hp.atr_ld_lambda * atr_ld_loss

            loss_sum         += loss.item()
            accu_sum         += accu.item()
            rec_loss_sum     += rec_loss.item()
            mse_atr_loss_sum += mse_atr_loss.item()
            cts_kl_loss_sum  += cts_kl_loss.item()
            atr_kl_loss_sum  += atr_kl_loss.item()
            cts_ld_loss_sum  += cts_ld_loss.item()
            atr_ld_loss_sum  += atr_ld_loss.item()

            counter += 1

            if counter > 3 and debug:
                break

    batch_num = len(loader)
    writer.add_scalar("valid/model",       loss_sum / batch_num,         epoch)
    writer.add_scalar("valid/accu",        accu_sum / batch_num,         epoch)
    writer.add_scalar("valid/reconstruct", rec_loss_sum / batch_num,     epoch)
    writer.add_scalar("valid/mse_atr",     mse_atr_loss_sum / batch_num, epoch)
    writer.add_scalar("valid/cts_kl",      cts_kl_loss_sum / batch_num,  epoch)
    writer.add_scalar("valid/atr_kl",      atr_kl_loss_sum / batch_num,  epoch)
    writer.add_scalar("valid/cts_ld",      cts_ld_loss_sum / batch_num,  epoch)
    writer.add_scalar("valid/atr_ld",      atr_ld_loss_sum / batch_num,  epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=hp.exp_name)
    parser.add_argument("--debug", type=bool, default=hp.debug)

    args = parser.parse_args()

    # ロガーの作成
    writer = SummaryWriter(hp.session_dir/ "log" / "spnetvc" / args.exp_name)

    # モデル
    model = SplitterVC(hp.seen_spkr_num).to(hp.device)

    clf = Classifier(hp.seen_spkr_num).to(hp.device)
    clf.load_state_dict(torch.load(hp.tng_result_dir / hp.clf_name /
                                   "CLF-latest.pth",
                                   map_location=hp.device)["model"])

    # optimizer
    vae_optimizer  = torch.optim.Adam(model.vae.parameters(),    lr=hp.lr)
    c_ld_optimizer = torch.optim.Adam(model.cts_ld.parameters(), lr=0.0002)
    a_ld_optimizer = torch.optim.Adam(model.atr_ld.parameters(), lr=0.0002)

    with open(hp.session_dir / "seen_speaker.json", 'r') as f:
        spkr_dct = json.load(f)

    with open(hp.session_dir / "mcep_statistics.json", 'r') as f:
        m_statistics_dct = json.load(f)

    # Create data loaders
    t_data = AudioDataset(hp.tng_data_dir, spkr_dct, m_statistics_dct,
                          seq_len=hp.seq_len)
    v_data = AudioDataset(hp.val_data_dir, spkr_dct, m_statistics_dct,
                          seq_len=hp.seq_len)

    vae_t_loader  = DataLoader(t_data, batch_size=hp.batch_size, shuffle=True)
    c_ld_t_loader = DataLoader(t_data, batch_size=hp.batch_size, shuffle=True)
    a_ld_t_loader = DataLoader(t_data, batch_size=hp.batch_size, shuffle=True)
    v_loader      = DataLoader(v_data, batch_size=hp.batch_size, shuffle=True)

    save_dir = hp.tng_result_dir / args.exp_name / "spnetvc"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(1, hp.spnetvc_epochs + 1)):
        train(model, vae_optimizer, c_ld_optimizer, a_ld_optimizer,
              vae_t_loader, c_ld_t_loader, a_ld_t_loader, writer, epoch,
              args.debug)
        valid(model, clf, v_loader, writer, epoch, args.debug)

        if epoch == hp.spnetvc_epochs:
            save_path = save_dir / "SPNETVC-latest.pth"
            save_checkpoint(save_path, model, vae_optimizer, epoch)
        elif epoch % hp.save_interval == 0:
            save_path = save_dir / f"SPNETVC-{epoch:03}.pth"
            save_checkpoint(save_path, model, vae_optimizer, epoch)


if __name__ == "__main__":
    main()
