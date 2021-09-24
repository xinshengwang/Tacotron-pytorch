#! bin/bash

base_root="/home/work_nfs3/xswang/data/TTS/obama2/clip/test"

python text_processing.py \
    --txt_dir=${base_root}/texts \
    --save_dir=${base_root}/texts_seq
