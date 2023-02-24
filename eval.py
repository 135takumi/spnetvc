import argparse
import json
import os

import librosa
import numpy as np
import torch

import hyper_parameters as hp
from utils import (save_wav, speech_synthesis, pitch_conversion, mcep_normalize,
                   mcep_denormalize, delete_dump_fame, load_trained_model)


def convert(model, s_spkr_dct, t_spkr_dct, f0_dct, mcep_dct, exp_name):
    for s_spkr in s_spkr_dct:
        s_spkr_dir = hp.test_data_dir / s_spkr

        for uttr in ['A26']:
            s_uttr_dir = s_spkr_dir / uttr
            save_dir = hp.test_result_dir / exp_name / s_spkr / uttr
            os.makedirs(save_dir, exist_ok=True)

            if not os.path.isdir(s_uttr_dir):
                continue

            s_mcep = np.load(s_uttr_dir / "mcep.npy")
            s_mcep = mcep_normalize(s_mcep, s_spkr, mcep_dct)
            s_mcep = torch.from_numpy(s_mcep).float()
            s_power = np.load(s_uttr_dir / "power.npy")
            s_f0    = np.load(s_uttr_dir / "f0.npy")
            s_ap    = np.load(s_uttr_dir / "ap.npy")

            for t_spkr in t_spkr_dct:
                t_uttr_dir = hp.test_data_dir / t_spkr / uttr

                if not os.path.isdir(t_uttr_dir):
                    continue

                t_mcep = np.load(t_uttr_dir / "mcep.npy")
                t_mcep = mcep_normalize(t_mcep, s_spkr, mcep_dct)
                t_mcep = torch.from_numpy(t_mcep).float()
                t_power = np.load(t_uttr_dir / "power.npy")
                t_f0    = np.load(t_uttr_dir / "f0.npy")
                t_ap    = np.load(t_uttr_dir / "ap.npy")

                with torch.no_grad():
                    _, c_mean, _ = model.cts_encode(s_mcep.to(hp.device))
                    _, a_mean, _ = model.atr_encode(t_mcep.to(hp.device))
                    a_mean = torch.mean(a_mean, 0, keepdim=True)
                    g_mcep = model.decode(c_mean, a_mean)

                # 変換音声の生成
                g_mcep = g_mcep.cpu().numpy()
                s_mcep_frame_num = s_f0.shape[0]
                g_mcep = delete_dump_fame(g_mcep, s_mcep_frame_num)
                g_mcep = mcep_denormalize(g_mcep, s_spkr, mcep_dct)
                g_mcep = np.concatenate([s_power, g_mcep], axis=1)
                g_mcep = g_mcep.copy()
                f0_converted = pitch_conversion(s_f0, s_spkr, t_spkr, f0_dct)
                converted_wav = speech_synthesis(f0_converted, g_mcep, s_ap,
                                                 hp.sampling_rate)
                # [1.0, -1.0]の範囲を超えることがあるので正規化して0.99かけておく
                converted_wav = librosa.util.normalize(converted_wav) * 0.99
                save_path = save_dir / f"{s_spkr}_to_{t_spkr}.wav"
                save_wav(save_path, hp.sampling_rate, converted_wav)

                # 変換音声との比較のために，target音声も用意
                # 生成音声の音質低下がモデルによるものかworldによるものか判別するために
                # target音声もworldを通す
                t_mcep_frame_num = t_f0.shape[0]
                t_mcep = delete_dump_fame(t_mcep, t_mcep_frame_num)
                t_mcep = mcep_denormalize(t_mcep, s_spkr, mcep_dct)
                t_mcep = np.concatenate([t_power, t_mcep], axis=1)
                t_mcep = t_mcep.copy()
                target_wav = speech_synthesis(t_f0, t_mcep, t_ap, hp.sampling_rate)
                target_wav = librosa.util.normalize(target_wav) * 0.99
                save_path = save_dir / f"{t_spkr}.wav"
                save_wav(save_path, hp.sampling_rate, target_wav)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="SPNETVC-latest.pth")
    parser.add_argument("--exp_name", type=str, default=hp.exp_name)
    args = parser.parse_args()

    with open(hp.session_dir / "seen_test_speaker.json", 'r') as f:
        seen_test_spkr_dct = json.load(f)

    with open(hp.session_dir / "unseen_speaker.json", 'r') as f:
        unseen_spkr_dct = json.load(f)

    with open(hp.session_dir / "f0_statistics.json", 'r') as f:
        f0_statistics_dct = json.load(f)

    m_statistics_dct = {}
    with open(hp.session_dir / "mcep_statistics.json", 'r') as f:
        for k, v in json.load(f).items():
            m_statistics_dct[k] = {
                "mean": np.array(v["mean"])[None, :],
                "std": np.array(v["std"])[None, :]
            }

    model = load_trained_model(args)

    convert(model, seen_test_spkr_dct, seen_test_spkr_dct, f0_statistics_dct,
            m_statistics_dct, args.exp_name)

    convert(model, seen_test_spkr_dct, unseen_spkr_dct, f0_statistics_dct,
            m_statistics_dct, args.exp_name)


if __name__ == '__main__':
    main()
