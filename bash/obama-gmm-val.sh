#! bin/bash
cfg_file="config/gmm-tts.yaml"
base_root="/home/work_nfs3/xswang/data/TTS/obama2/clip/test"
model_name='obama_gmm'

python main.py \
    --cfg_file=${cfg_file} \
    --txt_dir=${base_root}/texts_seq \
    --mel_dir=${base_root}/acoustic_features/mels \
    --file_dir=${base_root}/filenames \
    --save_root=logdir/${model_name}/logdir \
    --wav_path /home/work_nfs3/xswang/code/TTS/Tacotron-pytorch/logdir/obama_gmm/logdir/wav_gta \
    --mel_path /home/work_nfs3/xswang/code/TTS/Tacotron-pytorch/logdir/obama_gmm/logdir/mel_gta \
    --ckpt_pth /home/work_nfs3/xswang/code/TTS/Tacotron-pytorch/logdir/obama_gmm/logdir/ckpt/ckpt_200000 
