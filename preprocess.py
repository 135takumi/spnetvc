import json
import os
import random
import subprocess


import  librosa
import numpy as np
from tqdm import tqdm


import hyper_parameters as hp
from utils import world_decompose, world_encode_spectral_envelop


def pitch_statistics(f0_lst):
    f0 = np.concatenate(f0_lst)
    log_f0 = np.log(f0)
    mean = log_f0.mean()
    std = log_f0.std()

    return mean, std


def mcep_statistics(mcep_lst):
    mcep = np.concatenate(mcep_lst, axis=0)
    mean = list(np.mean(mcep, axis=0, keepdims=True).squeeze())
    std = list(np.std(mcep, axis=0, keepdims=True).squeeze())

    return mean, std


def make_exp_dir():
    """
    実験に使うディレクトリの作成
    """

    os.makedirs(hp.wav_dir / "male", exist_ok=True)
    os.makedirs(hp.wav_dir / "female", exist_ok=True)
    os.makedirs(hp.tng_data_dir, exist_ok=True)
    os.makedirs(hp.val_data_dir, exist_ok=True)
    os.makedirs(hp.test_data_dir, exist_ok=True)
    os.makedirs(hp.tng_result_dir, exist_ok=True)
    os.makedirs(hp.test_result_dir, exist_ok=True)


def make_seen_dataset(seen_speaker_lst, seen_test_speaker_lst, mcep_dct, f0_dct):
    """
    学習用データの作成(train用話者の時と，test用話者の時で分岐あり)
    """
    mcep_lst = []
    seen_speaker_dct = {}
    seen_test_speaker_dct = {}

    for speaker_index, speaker in enumerate(tqdm(seen_speaker_lst)):
        f0_lst = []
        if speaker[3] == "0":
            speaker_dir = hp.wav_dir /"male"/ speaker
        else:
            speaker_dir = hp.wav_dir / "female" / speaker

        if not os.path.isdir(speaker_dir):
            continue

        seen_speaker_dct[speaker] = speaker_index

        if speaker in seen_test_speaker_lst:
            seen_test_speaker_dct[speaker] = speaker_index

        train_wav_lst = list(speaker_dir.glob("*/[B-J]??.wav"))
        valid_wav_lst = list(speaker_dir.glob("*/[A]??.wav"))
        valid_wav_lst = sorted(valid_wav_lst)[25:]

        for train_wav in train_wav_lst:
            f0, mcep = extract_acoustic_features(train_wav, hp.tng_data_dir / speaker,
                                                 test_train="train")
            f0 = [f for f in f0 if f > 0.0]
            mcep_lst.append(mcep)
            f0_lst.append(f0)

        for valid_wav in valid_wav_lst:
            _, _ = extract_acoustic_features(valid_wav, hp.val_data_dir / speaker,
                                                 test_train="train")

        f0_mean, f0_std = pitch_statistics(f0_lst)
        f0_dct[speaker] = {"mean": f0_mean, "std": f0_std}

    for speaker in seen_speaker_lst:
        mcep_mean, mcep_std = mcep_statistics(mcep_lst)
        mcep_dct[speaker] = {"mean": mcep_mean, "std": mcep_std}

    return seen_speaker_dct, seen_test_speaker_dct


def make_test_dataset(test_speaker_lst, f0_dct=None, unseen=False):
    """
    テスト用データの作成(seen話者とunseen話者で分岐あり)
    """
    unseen_speaker_dct={}

    for speaker_index, speaker in enumerate(tqdm(test_speaker_lst)):
        f0_lst = []
        if speaker[3] == "0":
            speaker_dir = hp.wav_dir / "male" / speaker
        else:
            speaker_dir = hp.wav_dir / "female" / speaker

        unseen_speaker_dct[speaker] = speaker_index

        test_wav_lst = list(speaker_dir.glob("*/[A]??.wav"))
        test_wav_lst = sorted(test_wav_lst)[:25]

        for test_wav in test_wav_lst:
            f0, mcep = extract_acoustic_features(test_wav, hp.test_data_dir / speaker,
                                                 test_train="test")
            f0 = [f for f in f0 if f > 0.0]
            f0_lst.append(f0)

        if unseen:
            f0_mean, f0_std = pitch_statistics(f0_lst)
            f0_dct[speaker] = {"mean": f0_mean, "std": f0_std}

    return unseen_speaker_dct


def extract_acoustic_features(wav_path, speaker_dir, test_train):
    mcep_list = []

    wav, _ = librosa.core.load(wav_path, sr=hp.sampling_rate)
    wav = librosa.util.normalize(wav)
    wav, _ = librosa.effects.trim(wav, top_db=60)

    f0, _, sp, ap = world_decompose(wav, hp.sampling_rate)
    mcep = world_encode_spectral_envelop(sp, hp.sampling_rate, hp.mcep_channels+1)
    power = mcep[:,0:1]
    mcep = mcep[:, 1:]

    mcep_len = mcep.shape[0]

    if mcep_len < hp.seq_len:
        print(f"{wav_path} is too short")
        return None

    start_max = mcep_len - hp.mcep_channels + 1

    # test用データはmelを1フレームずつずらして32フレームごとに区切る
    if test_train == "train":
        cut_interval = 5
    # train用データは5フレームずつずらす
    else:
        cut_interval = 1

    for start in range(0, start_max, cut_interval):
        cutted_mcep = mcep[np.newaxis, np.newaxis, start:start + hp.mcep_channels, :]
        mcep_list.append(cutted_mcep)

    cutted_mcep = np.concatenate(mcep_list, 0)

    save_dir = speaker_dir / wav_path.name.split(".")[0]

    f0_path = save_dir / "f0.npy"
    mcep_path = save_dir / "mcep.npy"
    power_path = save_dir / "power.npy"
    ap_path = save_dir / "ap.npy"

    os.makedirs(save_dir, exist_ok=True)

    np.save(f0_path, f0, allow_pickle=False)
    np.save(mcep_path, cutted_mcep, allow_pickle=False)
    np.save(power_path, power, allow_pickle=False)
    np.save(ap_path, ap, allow_pickle=False)

    return f0, mcep


def convert_raw_to_wav():
    """
    ASJデータセットに含まれるrawデータをsoxをもちいてwavに変換する．
    """

    # subprocessでsoxを動かすためのリスト
    list_sox_comm_and_params = ["sox", "-r", "16000", "-b", "16", "-c", "1",
                   "-e", "signed-integer", "-t", "raw", "-x"]

    # loadという名前の付いたpathは，asjコーパスのdir以下にあるrawのpath
    # saveという名前の付いたpathは，読み込んだrawをwavにしたものを保存するpath
    for vol_num in range(1, hp.vols+1):
        file_paths_load_raw = hp.dir_path_asj.glob("vol" + str(vol_num) +"/DAT/???????/[A-J]/*.AD")
        for file_path_load_raw in file_paths_load_raw:
            # ASJの話者名はCAN0001やCAN1001という7文字の英数字になっている
            # 4桁目の数字は話者性別を表し，0は男性，1は女性
            file_name_load_raw = str(file_path_load_raw).split("/DAT/")[1]
            if file_name_load_raw[3] == "0":
                gender = "male"
            else:
                gender = "female"

            dir_name_load_raw = file_name_load_raw.split(".")[0]
            speaker_name_load_raw = dir_name_load_raw.split("/")[0]
            utter_set_name_load_raw = dir_name_load_raw.split("/")[1]
            dir_path_save_wav = hp.wav_dir / gender / speaker_name_load_raw  / \
                                      utter_set_name_load_raw

            os.makedirs(dir_path_save_wav, exist_ok=True)

            file_path_save_wav = hp.wav_dir / gender / (dir_name_load_raw + ".wav")
            list_sox_comm_and_params.append(str(file_path_load_raw))
            list_sox_comm_and_params.append(str(file_path_save_wav))
            run_sox = subprocess.run(list_sox_comm_and_params)
            if run_sox.returncode != 0:
                print("sox failed")

            # loadするデータのpathとsave用pathを初期化
            del list_sox_comm_and_params[-1]
            del list_sox_comm_and_params[-1]


def make_dataset():
    # randomのseed固定
    random.seed(0)

    # attrのidxがどの話者かを知るためのラベル
    mcep_dct = {}
    f0_dct = {}

    male_speaker_lst = os.listdir(hp.wav_dir / "male")
    male_speaker_lst = random.sample(male_speaker_lst, len(male_speaker_lst))
    female_speaker_lst = os.listdir(hp.wav_dir / "female")
    female_speaker_lst = random.sample(female_speaker_lst, len(female_speaker_lst))

    seen_speaker_lst = male_speaker_lst[:int(hp.seen_speaker_num/2)] + \
                              female_speaker_lst[:int(hp.seen_speaker_num/2)]
    seen_test_speaker_lst = male_speaker_lst[:int(hp.seen_speaker_num/2)] + \
                              female_speaker_lst[:int(hp.seen_speaker_num/2)]
    unseen_speaker_lst = male_speaker_lst[-int(hp.unseen_speaker_num/2):] + \
                              female_speaker_lst[-int(hp.unseen_speaker_num/2):]

    #  学習話者のそれぞれの発話をtrain_dataset用に加工してpickleで保存
    seen_speaker_dct, seen_test_speaker_dct = make_seen_dataset(seen_speaker_lst,
                                                                seen_test_speaker_lst,
                                                                mcep_dct, f0_dct)
    _ = make_test_dataset(seen_test_speaker_lst)
    unseen_speaker_dct = make_test_dataset(unseen_speaker_lst, f0_dct=f0_dct, unseen=True)

    with open(hp.session_dir / "f0_statistics.json", 'w') as f:
        json.dump(f0_dct, f, indent=2)
    with open(hp.session_dir / "mcep_statistics.json", 'w') as f:
        json.dump(mcep_dct, f, indent=2)
    with open(hp.session_dir / "seen_speaker.json", 'w') as f:
        json.dump(seen_speaker_dct, f, indent=2)
    with open(hp.session_dir / "seen_test_speaker.json", 'w') as f:
        json.dump(seen_test_speaker_dct, f, indent=2)
    with open(hp.session_dir / "unseen_speaker.json", 'w') as f:
        json.dump(unseen_speaker_dct, f, indent=2)

def main():
    make_exp_dir()
    #convert_raw_to_wav()
    make_dataset()

if __name__ == '__main__':
    main()

