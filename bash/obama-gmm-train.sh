#! bin/bash
cfg_file="config/gmm-tts.yaml"
base_root="/home/work_nfs3/xswang/data/TTS/obama2/clip/train"
model_name='obama_gmm'

python main.py \
    --cfg_file=${cfg_file} \
    --txt_dir=${base_root}/texts_seq \
    --mel_dir=${base_root}/acoustic_features/mels \
    --file_dir=${base_root}/train_val \
    --save_root=logdir/${model_name}/logdir \
    --train 
