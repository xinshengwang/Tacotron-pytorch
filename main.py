# Tacotron pytorch implement
# Author       : Xinsheng Wang
# Email        : w.xinshawn@gmail.com
# Filename     : train.py
# Created time : 2021-06-23 10:00
# Last modified: 
# ==============================================================================
"""Main entrance for training"""
import os
import pdb
from pickle import TRUE
import torch
import random
import pprint
import argparse
import numpy as np
from steps import trainer
from utils import util
from utils.config import cfg_from_file,cfg

def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)   
	torch.cuda.manual_seed_all(seed)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-cfg','--cfg_file',default='config/dca-tts.yaml',
                        help='yaml files for configurations.')
	parser.add_argument('-wd', '--txt_dir', type = str, default = r'F:\dataset\TTS_TEST\texts_seq',
						help = 'directory to wav path')
	parser.add_argument('-md', '--mel_dir', type = str, default = r'F:\dataset\TTS_TEST\acoustic_features\mels',
						help = 'directory to mel path')
	parser.add_argument('-sd','--file_dir',type=str,default=r'F:\dataset\TTS_TEST\train_val',
						help='dir to save the split (train / val) filenames')
	parser.add_argument('-sr','--save_root',type = str, default='logdir')
	parser.add_argument('-l', '--log_dir', type = str, default = '',
						help = 'directory to save tensorboard logs')
	parser.add_argument('-ll','--train_info',type=str,default='')
	parser.add_argument('-cd', '--ckpt_dir', type = str, default = '',
						help = 'directory to save checkpoints')
	parser.add_argument('-cp', '--ckpt_pth', type = str, default = '',
						help = 'path to load checkpoints')
	parser.add_argument('-wp', '--wav_path', type = str, default = '',
						help = 'path to load checkpoints')
	parser.add_argument('--random_seed',type=int,default=1234)
	parser.add_argument('--train',default=True) #
	args = parser.parse_args()
	return args

def make_dirs(args):
	args.log_dir = os.path.join(args.save_root,'event')
	args.train_info = os.path.join(args.save_root,'train_info.log')
	args.ckpt_dir = os.path.join(args.save_root,'ckpt')
	args.wav_path = os.path.join(args.save_root,'wavs')
	for path in [args.log_dir,args.ckpt_dir,args.wav_path]:
		if not os.path.exists(path):
			os.makedirs(path)
		
if __name__ == '__main__':
	args = parse_args()
	set_seed(args.random_seed) 
	if args.cfg_file is not None:
		cfg_from_file(args.cfg_file)
	cfg.file_dir = args.file_dir
	print('Using config:')
	pp = pprint.PrettyPrinter(width=41, compact=True)
	pp.pprint(cfg)
	make_dirs(args)
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = False # faster due to dynamic input shape
	torch.backends.cudnn.deterministic = True
	# pdb.set_trace()
	if args.train:
		trainer.train(args)
	else:
		trainer.infer(args)