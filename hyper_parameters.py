from pathlib import Path


import torch


# ASJセットで使用するvolの数
vols = 1

# feature config
sampling_rate = 16000
flame_interval = 5
mcep_channels = 32
seq_len = 32
seen_speaker_num = 4
unseen_speaker_num = 4

# training config
batch_size = 128
lr = 1e-4
spnetvc_epochs = 20000
clf_epochs = 10000

rec_lambda  = 1
cts_kl_lambda = 0.05
atr_kl_lambda = 1
cts_ld_lambda = 1
lambda_schedule = 80000

debug = False

# save comfig
save_interval = 1000

# 以降は実行に必要なファイルパスの指定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 実験名
clf_name = 'clf1'
exp_name = 'no_atr_ld_and_atr_mse'

# 以降は実行に必要なファイルパスの指定
#base_dir = Path('/mnt/d/brood/M1/projects/spnetvc')
base_dir = Path('/home/isako/M1/projects/spnetvc')
dir_path_asj = Path('/data/corpus/ASJ')
session_dir = base_dir / 'sessions'
wav_dir = session_dir / 'wav_data'
tng_data_dir = session_dir / 'train_data'
val_data_dir = session_dir / 'valid_data'
test_data_dir = session_dir / 'test_data'
tng_result_dir = session_dir / 'training'
test_result_dir = session_dir / 'test_wav'
log_dir = session_dir / 'log'
