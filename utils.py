import numpy as np
import pyworld
import scipy.io.wavfile


def save_wav(file_path, sampling_rate, audio):
    audio = (audio * 32768).astype("int16")
    scipy.io.wavfile.write(file_path, sampling_rate, audio)


def world_decompose(wav, fs, frame_period=5.0):
    wav = wav.astype(np.float64)
    f0, _time = pyworld.dio(wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)
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


def speech_synthesis(vocoder, coded_sp):
    decoded_sp = vocoder(coded_sp)
    wav_signal = decoded_sp.squeeze().cpu().detach().numpy()

    return wav_signal
