# VAE-VC

- model.py

モデルが定義されている

- train.py

学習スクリプト

- utils.py

音声の読み込みなど

- data.py

データローダーの定義用

- eval.py

推論スクリプト

## 動かし方

```
- [data_root]
  - wav
    - speaker01
      - XXXXX.wav
      - YYYYY.wav
    - speaker02
    ...
```

適当な場所に`data_root`を作成してその中に上のように音声ファイルを配置

### 前処理

```
python preprocess [data_root]
```

### 学習

```
python train.py [data_root]
```

使う GPU を指定 & バックグラウンドでログアウトしても続くように

```
CUDA_VISIBLE_DEVICES=0 nohup python train.py [data_root] [結果の保存先] > out.log &
```
