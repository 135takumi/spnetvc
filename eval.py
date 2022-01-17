import argparse
import json
import os

import librosa
import numpy as np
import torch

import hyper_parameters as hp
from model import SplitterVC
from utils import (save_wav, speech_synthesis)


def mcep_normalize(mcep, label, mcep_dict):
    speaker_dict = mcep_dict[label]
    mean, std = speaker_dict["mean"], speaker_dict["std"]
    mcep = (mcep - mean) / std

    return mcep


def mcep_denormalize(mcep, label, mcep_dict):
    speaker_dict = mcep_dict[label]
    mean, std = speaker_dict["mean"], speaker_dict["std"]
    mcep = mcep * std + mean

    return mcep


def pitch_conversion(f0, source, target, f0_dict):
    mean_source, std_source = f0_dict[source]["mean"], f0_dict[source]["std"]
    mean_target, std_target = f0_dict[target]["mean"], f0_dict[target]["std"]

    f0_converted = np.exp((np.log(f0 + 1e-6) - mean_source) / std_source * std_target + mean_target)

    return f0_converted

def delete_dump_fame(mceps, num_mcep_flame):
    """
    生成したmelには重複しているフレームがあるのでそのフレームの削除
    """

    # ここで空のリストを作成
    # (本来のフレーム数, 32)
    reshaped_mcep = np.zeros((num_mcep_flame, mceps.shape[2]))

    for i in range(len(mceps)):
        # データ1つ分とって、変換用に増やしていた次元1つを抜く
        # (フレーム数、メルケプ次元数) = (32, 32)
        target_mcep = mceps[i][0]
        target_mcep = np.pad(target_mcep, [(i, num_mcep_flame - (i + hp.mcep_channels)),
                                           (0, 0)], 'constant')
        reshaped_mcep += target_mcep

    # 1. メルケプ次元数をフレーム数に合わせてpaddingすることで、正方行列にする
    reshaped_mcep = np.pad(reshaped_mcep, [(0, 0),
                                           (0, num_mcep_flame - hp.mcep_channels)],
                              'constant')

    # 2. 単位行列を作成し、中身を1, 1,…から1,1/2,…1/32,…1/32,1/31,…1にする
    identity_matrix = np.eye(num_mcep_flame)
    for flame in range(num_mcep_flame):
        if flame < num_mcep_flame / 2:
            identity_matrix[flame][flame] = 1 / min(flame + 1, hp.mcep_channels)
        else:
            identity_matrix[flame][flame] = 1 / min(num_mcep_flame - flame, hp.mcep_channels)

    # 2と1の行列積をとり、paddingしていた範囲を削除
    reshaped_mcep = identity_matrix @ reshaped_mcep
    reshaped_mcep = np.delete(reshaped_mcep, slice(hp.mcep_channels,
                                                   num_mcep_flame), 1)

    return reshaped_mcep


def convert(model, source_speaker_dict, target_speaker_dict, f0_dict, mcep_dict, exp_name):
    for source_speaker in source_speaker_dict:
        source_speaker_dir = hp.test_data_dir / source_speaker

        for uttr in os.listdir(source_speaker_dir):
            source_uttr_dir = source_speaker_dir / uttr
            save_dir = hp.test_result_dir / exp_name / source_speaker / uttr

            if not os.path.isdir(source_uttr_dir):
                continue

            source_mcep = np.load(source_uttr_dir / "mcep.npy")
            source_mcep_normalized = mcep_normalize(source_mcep, source_speaker, mcep_dict)
            source_mcep_normalized = torch.from_numpy(source_mcep_normalized).float()
            source_power = np.load(source_uttr_dir / "power.npy")
            source_f0 = np.load(source_uttr_dir / "f0.npy")
            source_ap = np.load(source_uttr_dir / "ap.npy")

            for target_speaker in target_speaker_dict:
                target_uttr_dir = hp.test_data_dir / target_speaker / uttr

                if not os.path.isdir(target_uttr_dir):
                    continue

                target_mcep = np.load(target_uttr_dir / "mcep.npy")
                target_mcep_normalized = mcep_normalize(target_mcep, source_speaker, mcep_dict)
                target_mcep_normalized = torch.from_numpy(target_mcep_normalized).float()
                target_power = np.load(target_uttr_dir / "power.npy")
                target_f0 = np.load(target_uttr_dir / "f0.npy")
                target_ap = np.load(target_uttr_dir / "ap.npy")

                with torch.no_grad():
                    _, cts_mean, _ = model.cts_encode(source_mcep_normalized.to(hp.device))
                    _, atr_mean, _ = model.atr_encode(target_mcep_normalized.to(hp.device))
                    atr_mean= torch.mean(atr_mean, 0, keepdim=True)
                    mcep_converted = model.decode(cts_mean, atr_mean)

                mcep_converted = mcep_converted.cpu().numpy()
                source_mcep_frame_num = source_f0.shape[0] 
                mcep_converted = delete_dump_fame(mcep_converted, source_mcep_frame_num)
                mcep_denormed = mcep_denormalize(mcep_converted, source_speaker, mcep_dict)
                mcep_denormed = np.concatenate([source_power, mcep_denormed], axis=1)
                mcep_denormed = mcep_denormed.copy(order='C')

                f0_converted = pitch_conversion(source_f0, source_speaker, target_speaker,
                                                f0_dict)

                converted_wav = speech_synthesis(f0_converted, mcep_denormed, source_ap,
                                                 hp.sampling_rate)

                target_mcep_frame_num = target_f0.shape[0]
                target_mcep = delete_dump_fame(target_mcep, target_mcep_frame_num)
                target_mcep = np.concatenate([target_power, target_mcep], axis=1)
                target_mcep = target_mcep.copy(order='C')

                target_wav = speech_synthesis(target_f0, target_mcep, target_ap, hp.sampling_rate)

                # [1.0, -1.0]の範囲を超えることがあるので正規化して0.99かけておく
                converted_wav = librosa.util.normalize(converted_wav) * 0.99

                os.makedirs(save_dir, exist_ok=True)

                save_path = save_dir / f"{source_speaker}_to_{target_speaker}.wav"
                save_wav(save_path, hp.sampling_rate, converted_wav)

                save_path = save_dir / f"{target_speaker}.wav"
                save_wav(save_path, hp.sampling_rate, target_wav)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="VAEVC-latest.pth")
    parser.add_argument("--exp_name", type=str, default=hp.exp_name)
    args = parser.parse_args()

    with open(hp.session_dir / "seen_test_speaker.json", 'r') as f:
        seen_test_speaker_dict = json.load(f)

    with open(hp.session_dir / "unseen_speaker.json", 'r') as f:
        unseen_speaker_dict = json.load(f)

    with open(hp.session_dir / "f0_statistics.json", 'r') as f:
        f0_dict = json.load(f)

    mcep_dict = {}
    with open(hp.session_dir / "mcep_statistics.json", 'r') as f:
        for k, v in json.load(f).items():
            mcep_dict[k] = {
                "mean": np.array(v["mean"])[None, :],
                "std": np.array(v["std"])[None, :]
            }

    model = SplitterVC(hp.seen_speaker_num).to(hp.device)
    model.load_state_dict(torch.load(hp.tng_result_dir / args.exp_name / "spnetvc" / args.weight,
                                     map_location=hp.device)["model"])
    model.eval()
    
    convert(model, seen_test_speaker_dict, seen_test_speaker_dict, f0_dict, mcep_dict,
            args.exp_name)
    
    convert(model, seen_test_speaker_dict, unseen_speaker_dict, f0_dict, mcep_dict,
            args.exp_name)



if __name__ == '__main__':
    main()
