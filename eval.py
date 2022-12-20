import argparse
import json
import os
import random

import hifigan
import librosa
import torch
import torch.nn.functional as func

import hyper_parameters as hp
from model import SplitterVC
from utils import (save_wav, speech_synthesis)


def melsp_normalize(melsp, label, melsp_dict):
    speaker_dict = melsp_dict[label]
    mean, std = speaker_dict["mean"], speaker_dict["std"]
    melsp = (melsp - mean) / std

    return melsp


def melsp_denormalize(melsp, label, melsp_dict):
    speaker_dict = melsp_dict[label]
    mean, std = speaker_dict["mean"], speaker_dict["std"]
    melsp = melsp * std + mean

    return melsp


def delete_dump_fame(melsps, num_melsp_flame):
    """
    生成したmelには重複しているフレームがあるのでそのフレームの削除
    """

    # ここで空のリストを作成
    # (本来のフレーム数, 80)
    reshaped_melsp = torch.zeros((num_melsp_flame, hp.melsp_channels)).to(hp.device)

    for i in range(len(melsps)):
        # データ1つ分とって、変換用に増やしていた次元1つを抜く
        # (フレーム数、メルスぺ次元数) = (32, 80)
        target_melsp = melsps[i][0]
        target_melsp = func.pad(target_melsp, (0, 0, i, num_melsp_flame - (i + hp.seq_len)))
        reshaped_melsp += target_melsp

    # 1. メルケプ次元数をフレーム数に合わせてpaddingすることで、正方行列にする
    reshaped_melsp = func.pad(reshaped_melsp, (0, num_melsp_flame - hp.melsp_channels, 0, 0))

    # 2. 単位行列を作成し、中身を1, 1,…から1,1/2,…1/32,…1/32,1/31,…1にする
    identity_matrix = torch.eye(num_melsp_flame).to(hp.device)
    for flame in range(num_melsp_flame):
        if flame < num_melsp_flame / 2:
            identity_matrix[flame][flame] = 1 / min(flame + 1, hp.seq_len)
        else:
            identity_matrix[flame][flame] = 1 / min(num_melsp_flame - flame, hp.seq_len)

    # 2と1の行列積をとり、paddingしていた範囲を削除
    reshaped_melsp = identity_matrix @ reshaped_melsp
    reshaped_melsp = reshaped_melsp[:, :hp.melsp_channels]

    return reshaped_melsp


def convert(model, vocoder, source_speaker_dict, target_speaker_dict, melsp_dict, exp_name):
    for source_speaker in source_speaker_dict:
        source_speaker_dir = hp.test_data_dir / source_speaker

        uttr_list = random.sample(os.listdir(source_speaker_dir), 1)
        for uttr in uttr_list:
            source_uttr_dir = source_speaker_dir / uttr
            save_dir = hp.test_result_dir / exp_name / source_speaker / uttr

            if not os.path.isdir(source_uttr_dir):
                continue

            source_melsp = torch.load(source_uttr_dir / "melsp.pt").to(hp.device)
            source_melsp_frame_num = source_melsp.size(0) + hp.seq_len - 1
            source_melsp_normalized = melsp_normalize(source_melsp, source_speaker, melsp_dict)

            for target_speaker in target_speaker_dict:
                target_uttr_dir = list(hp.test_data_dir.glob(target_speaker +
                                                             "/VOICEACTRESS100_???/"))[0]

                if not os.path.isdir(target_uttr_dir):
                    continue

                target_melsp = torch.load(target_uttr_dir / "melsp.pt").to(hp.device)
                target_melsp_normalized = melsp_normalize(target_melsp, source_speaker, melsp_dict)

                with torch.no_grad():
                    _, cts_mean, _ = model.cts_encode(source_melsp_normalized)
                    atr_z, _, _ = model.atr_encode(target_melsp_normalized)
                    atr_z = torch.mean(atr_z, 0, keepdim=True)
                    melsp_converted = model.decode(cts_mean, atr_z)

                melsp_converted = melsp_converted
                melsp_converted = delete_dump_fame(melsp_converted, source_melsp_frame_num)
                melsp_denormed = melsp_denormalize(melsp_converted, source_speaker, melsp_dict)
                melsp_denormed = torch.transpose(melsp_denormed, 0, 1).unsqueeze(0)

                converted_wav = speech_synthesis(vocoder, melsp_denormed)

                target_melsp_frame_num = target_melsp.size(0) + hp.seq_len - 1
                target_melsp_normalized = delete_dump_fame(target_melsp_normalized,
                                                           target_melsp_frame_num)
                target_melsp_denormed = melsp_denormalize(target_melsp_normalized, source_speaker,
                                                          melsp_dict)
                target_melsp_denormed = torch.transpose(target_melsp_denormed, 0, 1).unsqueeze(0)

                target_wav = speech_synthesis(vocoder, target_melsp_denormed)
                target_wav = librosa.util.normalize(target_wav) * 0.99

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

    melsp_dict = {}
    with open(hp.session_dir / "melsp_statistics.json", 'r') as f:
        for k, v in json.load(f).items():
            melsp_dict[k] = {
                "mean": torch.tensor(v["mean"], dtype=torch.float, device=hp.device),
                "std": torch.tensor(v["std"], dtype=torch.float, device=hp.device)
            }

    model = SplitterVC(hp.seen_speaker_num).to(hp.device)
    model.load_state_dict(torch.load(hp.tng_result_dir / args.exp_name / "spnetvc" / args.weight,
                                     map_location=hp.device)["model"])
    model.eval()
    vocoder = hifigan.get_vocoder().to(hp.device).to(hp.device)

    convert(model, vocoder, seen_test_speaker_dict, seen_test_speaker_dict, melsp_dict,
            args.exp_name)

    convert(model, vocoder, seen_test_speaker_dict, unseen_speaker_dict, melsp_dict, args.exp_name)


if __name__ == '__main__':
    main()
