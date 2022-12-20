import json
import os
import random
import shutil

import hifigan
import librosa
import numpy as np
import torch
from tqdm import tqdm

import hyper_parameters as hp


def make_exp_dir():
    """
    実験に使うディレクトリの作成
    """

    os.makedirs(hp.wav_dir / 'male', exist_ok=True)
    os.makedirs(hp.wav_dir / 'female', exist_ok=True)
    os.makedirs(hp.tng_data_dir, exist_ok=True)
    os.makedirs(hp.val_data_dir, exist_ok=True)
    os.makedirs(hp.test_data_dir, exist_ok=True)
    os.makedirs(hp.tng_result_dir, exist_ok=True)
    os.makedirs(hp.test_result_dir, exist_ok=True)

def melsp_statistics(melsp):
    melsp = torch.cat(melsp, dim=0)

    mean = list(torch.mean(melsp, dim=0).numpy().astype(np.float64))
    std = list(torch.std(melsp, dim=0).numpy().astype(np.float64))

    return mean, std


def extract_feature(audio2mel, wav_path, speaker_dir, test_train):
    melsp_list = []

    # 読み込み 正規化
    wav, _ = librosa.load(wav_path, sr=hp.sampling_rate)
    wav = librosa.util.normalize(wav)

    # 前後の無音を除去 top dbでどれぐらい厳しく削除するか決める
    wav, _ = librosa.effects.trim(wav, top_db=60)
    wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float()
    melsp = audio2mel(wav)
    melsp = torch.transpose(melsp, 2, 1)

    # HiFi-GANを利用して特徴量を取得
    melsp_len = melsp.size()[1]

    # 長さが短いものを除く
    if melsp_len < hp.seq_len:
        print(f'{wav_path} is too short')
        return None

    start_max = melsp_len - hp.seq_len + 1

    # test用データはmelを1フレームずつずらして32フレームごとに区切る
    if test_train == 'train':
        cut_interval = 5
    # train用データは5フレームずつずらす
    else:
        cut_interval = 1

    for start in range(0, start_max, cut_interval):
        cutted_melsp = melsp[:,start:start+hp.seq_len,:].unsqueeze(0)
        melsp_list.append(cutted_melsp)

    cutted_melsp = torch.cat(melsp_list, 0)

    save_dir = speaker_dir / wav_path.name.split(".")[0]

    melsp_path = save_dir / 'melsp.pt'

    os.makedirs(save_dir, exist_ok=True)

    torch.save(cutted_melsp, melsp_path)

    return melsp.squeeze(0)


def download_wav(speaker_list, gender):
    for speaker in speaker_list:
        jvs_wav_dir = hp.dir_path_jvs / speaker / 'parallel100' / 'wav24kHz16bit'
        save_dir = hp.wav_dir / gender / speaker
        os.makedirs(save_dir, exist_ok=True)
        shutil.copytree(jvs_wav_dir, save_dir)


def prepare_wav():
    with open(hp.dir_path_jvs / 'gender_f0range.txt') as f:
        speaker_info_list = f.readlines()
    male_speaker_list = [si[0:6] for si in speaker_info_list if si[7] == 'M']
    female_speaker_list = [si[0:6] for si in speaker_info_list if si[7] == 'F']

    sampled_male_speaker_list = random.sample(male_speaker_list, int(hp.seen_speaker_num / 2))
    sampled_female_speaker_list = random.sample(female_speaker_list, int(hp.seen_speaker_num / 2))

    download_wav(sampled_male_speaker_list, 'male')
    download_wav(sampled_female_speaker_list, 'female')


def make_seen_dataset(audio2mel, seen_speaker_lst, seen_test_speaker_lst, melsp_dct):
    """
    学習用データの作成(train用話者の時と，test用話者の時で分岐あり)
    """
    melsp_lst = []
    seen_speaker_dct = {}
    seen_test_speaker_dct = {}

    for speaker_index, speaker in enumerate(tqdm(seen_speaker_lst)):
        seen_speaker_dct[speaker] = speaker_index

        wav_lst = list(hp.wav_dir.glob('*/'+speaker+'/VOICEACTRESS100_???.wav'))
        wav_lst = random.sample(sorted(wav_lst), len(wav_lst))
        train_wav_lst = wav_lst[:hp.train_wav_num]
        valid_wav_lst = wav_lst[hp.train_wav_num:hp.train_wav_num+hp.valid_wav_num]

        if speaker in seen_test_speaker_lst:
            seen_test_speaker_dct[speaker] = speaker_index
            test_wav_lst = wav_lst[-hp.test_wav_num:]

        for train_wav in train_wav_lst:
            melsp = extract_feature(audio2mel, train_wav, hp.tng_data_dir / speaker, 'train')
            melsp_lst.append(melsp)

        for valid_wav in valid_wav_lst:
            _ = extract_feature(audio2mel, valid_wav, hp.val_data_dir / speaker, 'trian')

        for test_wav in test_wav_lst:
            _ = extract_feature(audio2mel, test_wav, hp.test_data_dir / speaker, 'test')

    for speaker in seen_speaker_lst:
        melsp_mean, melsp_std = melsp_statistics(melsp_lst)
        melsp_dct[speaker] = {"mean": melsp_mean, "std": melsp_std}

    return seen_speaker_dct, seen_test_speaker_dct


def make_unseen_dataset(audio2mel, test_speaker_lst):
    """
    テスト用データの作成(seen話者とunseen話者で分岐あり)
    """
    unseen_speaker_dct={}

    for speaker_index, speaker in enumerate(tqdm(test_speaker_lst)):
        wav_lst = list(hp.wav_dir.glob('*/' + speaker + '/VOICEACTRESS100_???.wav'))
        wav_lst = random.sample(sorted(wav_lst), len(wav_lst))
        test_wav_lst = wav_lst[:hp.test_wav_num]

        unseen_speaker_dct[speaker] = speaker_index

        for test_wav in test_wav_lst:
             _ = extract_feature(audio2mel, test_wav, hp.test_data_dir / speaker, 'test')

    return unseen_speaker_dct


def make_dataset():
    melsp_dct = {}

    male_speaker_lst = [f for f in os.listdir(hp.wav_dir / 'male') if not f.startswith('.')]
    male_speaker_lst = random.sample(sorted(male_speaker_lst), len(male_speaker_lst))
    female_speaker_lst = [f for f in os.listdir(hp.wav_dir / 'female') if not f.startswith('.')]
    female_speaker_lst = random.sample(sorted(female_speaker_lst), len(female_speaker_lst))

    seen_speaker_lst = male_speaker_lst[:int(hp.seen_speaker_num / 2)] + \
                       female_speaker_lst[:int(hp.seen_speaker_num / 2)]
    seen_test_speaker_lst = male_speaker_lst[:int(hp.seen_test_speaker_num / 2)] + \
                            female_speaker_lst[:int(hp.seen_test_speaker_num / 2)]
    unseen_speaker_lst = male_speaker_lst[-int(hp.unseen_speaker_num / 2):] + \
                         female_speaker_lst[-int(hp.unseen_speaker_num / 2):]

    audio2mel = hifigan.get_feature_extractor()

    seen_speaker_dct, seen_test_speaker_dct = make_seen_dataset(audio2mel, seen_speaker_lst,
                                                                seen_test_speaker_lst,
                                                                melsp_dct)

    unseen_speaker_dct = make_unseen_dataset(audio2mel, unseen_speaker_lst)

    with open(hp.session_dir / 'melsp_statistics.json', 'w') as f:
        json.dump(melsp_dct, f, indent=2)
    with open(hp.session_dir / 'seen_speaker.json', 'w') as f:
        json.dump(seen_speaker_dct, f, indent=2)
    with open(hp.session_dir / 'seen_test_speaker.json', 'w') as f:
        json.dump(seen_test_speaker_dct, f, indent=2)
    with open(hp.session_dir / 'unseen_speaker.json', 'w') as f:
        json.dump(unseen_speaker_dct, f, indent=2)


def main():
    make_exp_dir()
    # prepare_wav()
    make_dataset()

if __name__ == '__main__':
    main()