# transfer text sequence to .npy (only for English)
# Author       : Xinsheng Wang
# Email        : w.xinshawn@gmail.com
# Filename     : train.py
# Created time : 2021-07-13 10:00
# Last modified: 
# ==============================================================================
import os
import argparse
import future
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from text import text_to_sequence

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-td', '--txt_dir', type = str, default = r'F:\dataset\TTS_TEST\texts',
						help = 'directory to text path')
	parser.add_argument('-sd', '--save_dir', type = str, default = r'F:\dataset\TTS_TEST\texts_seq',
						help = 'directory to save npy')
	args = parser.parse_args()
	return args

def get_text(input_path,save_path):
    with open(input_path,'r') as f:
        text = f.readlines()[0].split('\t')[-1]
        seq = text_to_sequence(text,['english_cleaners'])
        np.save(save_path,seq)

if __name__ == '__main__':
    args = parse_args()
    names =  os.listdir(args.txt_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    Executor = ProcessPoolExecutor(max_workers=4)
    futures = []
    for name in tqdm(names):
        input_path = os.path.join(args.txt_dir,name)
        save_path = os.path.join(args.save_dir,os.path.splitext(name)[0] + '.npy')
        get_text(input_path,save_path)
    #     futures.append(Executor.submit(partial(get_text,input_path,save_path)))
    # for future in tqdm(futures):
    #     future.result()
