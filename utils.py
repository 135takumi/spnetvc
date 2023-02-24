import numpy as np
import pyworld
import scipy.io.wavfile
import torch


import hyper_parameters as hp
from model import SplitterVC

def save_wav(file_path, sampling_rate, audio):
    audio = (audio * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)


def world_decompose(wav, fs, frame_period=5.0):
    wav = wav.astype(np.float64)
    f0, _time = pyworld.dio(wav, fs, frame_period=frame_period, f0_floor=71.0,
                            f0_ceil=800.0)
    f0 = pyworld.stonemask(wav, f0, _time, fs)
    sp = pyworld.cheaptrick(wav, f0, _time, fs)
    ap = pyworld.d4c(wav, f0, _time, fs)

    return f0, _time, sp, ap


def world_encode_spectral_envelop(sp, fs, dim=25):
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)

    return coded_sp


def world_decode_spectral_envelop(coded_sp, fs):
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)

    return decoded_sp


def speech_synthesis(f0, coded_sp, ap, fs, frame_period=5.0):
    decoded_sp = world_decode_spectral_envelop(coded_sp, fs)
    wav_signal = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    wav_signal = wav_signal.astype(np.float32)

    return wav_signal

def mcep_normalize(mcep, label, mcep_dct):
    speaker_dct = mcep_dct[label]
    mean, std = speaker_dct["mean"], speaker_dct["std"]
    mcep = (mcep - mean) / std

    return mcep


def mcep_denormalize(mcep, label, mcep_dct):
    speaker_dct = mcep_dct[label]
    mean, std = speaker_dct["mean"], speaker_dct["std"]
    mcep = mcep * std + mean

    return mcep

def save_checkpoint(filepath, model, optimizer, epoch):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(state, filepath)


def load_trained_model(args):
    model = SplitterVC(hp.seen_spkr_num).to(hp.device)
    model.load_state_dict(torch.load(hp.tng_result_dir / args.exp_name /
                                     "spnetvc" / args.weight,
                                     map_location=hp.device)["model"])
    model.eval()

    return model


def pitch_conversion(f0, source, target, f0_dct):
    mean_source, std_source = f0_dct[source]["mean"], f0_dct[source]["std"]
    mean_target, std_target = f0_dct[target]["mean"], f0_dct[target]["std"]

    f0_converted = np.exp((np.log(f0 + 1e-6) - mean_source) /
                          std_source * std_target + mean_target)

    return f0_converted


def delete_dump_fame(mceps, mcep_f_num):
    """
    生成したmelには重複しているフレームがあるのでそのフレームの削除
    """

    # ここで空のリストを作成
    # (本来のフレーム数, 32)
    r_mcep = np.zeros((mcep_f_num, mceps.shape[2]))

    for i in range(len(mceps)):
        # データ1つ分とって、変換用に増やしていた次元1つを抜く
        # (フレーム数、メルケプ次元数) = (32, 32)
        t_mcep = mceps[i][0]
        t_mcep = np.pad(t_mcep, [(i, mcep_f_num - (i + hp.mcep_dim)),
                                 (0, 0)])
        r_mcep += t_mcep

    # 1. メルケプ次元数をフレーム数に合わせてpaddingすることで、正方行列にする
    r_mcep = np.pad(r_mcep, [(0, 0), (0, mcep_f_num - hp.mcep_dim)])

    # 2. 単位行列を作成し、中身を1, 1,…から1,1/2,…1/32,…1/32,1/31,…1にする
    i_matrix = np.eye(mcep_f_num)
    for flame in range(mcep_f_num):
        if flame < mcep_f_num / 2:
            i_matrix[flame][flame] = 1 / min(flame + 1, hp.mcep_dim)
        else:
            i_matrix[flame][flame] = 1 / min(mcep_f_num - flame,
                                             hp.mcep_dim)

    # 2と1の行列積をとり、paddingしていた範囲を削除
    r_mcep = i_matrix @ r_mcep
    r_mcep = np.delete(r_mcep, slice(hp.mcep_dim, mcep_f_num), 1)

    return r_mcep

