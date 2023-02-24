from pathlib import Path


import torch


# ASJセットで使用するvolの数
vols = 3

# feature config
sampling_rate = 16000
flame_interval = 5
mcep_dim = 32
seq_len = 32
seen_spkr_num = 50
seen_test_spkr_num = 4
unseen_spkr_num = 4
train_wav_num = 25
vaild_wav_num = 5
test_wav_num = 25

# training config
batch_size = 128
lr = 1e-4
spnetvc_epochs = 2
clf_epochs = 1

rec_lambda = 1
mse_atr_lambda = 1
cts_kl_lambda = 0.05
atr_kl_lambda = 1
cts_ld_lambda = 1
atr_ld_lambda = 0.5
lambda_schedule = 80000

debug = True

# save comfig
save_interval = 1000

# 以降は実行に必要なファイルパスの指定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 実験名
clf_name = 'clf_hikitugi'
exp_name = 'model_hikitugi'

# 以降は実行に必要なファイルパスの指定
base_dir = Path('/mnt/d/brood/M1/projects/spnetvc')
# base_dir = Path('/home/isako/M1/projects/spnetvc')

# ASJコーパスの保存先
dir_path_asj = Path('/data/corpus/ASJ')

# session_dir以下に実験のデータが保存される（sessionsディレクトリは自分で作る必要あり）
session_dir = base_dir / 'sessions'
wav_dir = session_dir / 'wav_data'
tng_data_dir = session_dir / 'train_data'
val_data_dir = session_dir / 'valid_data'
test_data_dir = session_dir / 'test_data'
tng_result_dir = session_dir / 'training'
test_result_dir = session_dir / 'test_wav'
log_dir = session_dir / 'log'
