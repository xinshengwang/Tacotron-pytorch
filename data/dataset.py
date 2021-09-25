import os
import json
import torch
import random
from glob import glob
import numpy as np
from utils.config import cfg 
from torch.utils.data import Dataset
import pdb

def create_test_filenames(args):
	txt_names =[os.path.split(name)[-1].split('.')[0] for name in glob(os.path.join(args.txt_dir,'*.npy'))]
	mel_names = [os.path.split(name)[-1].split('.')[0] for name in glob(os.path.join(args.mel_dir,'*.npy'))]
	names = list(set(txt_names) & set(mel_names))
	names = txt_names
	test_path = os.path.join(args.file_dir,'filenames_test.json')
	with open(test_path,'w') as f:
		json.dump(names,f)

def create_train_val_filenames(args):
	print('selecting val files and creating training filenmaes')
	txt_names =[os.path.split(name)[-1].split('.')[0] for name in glob(os.path.join(args.txt_dir,'*.npy'))]
	mel_names = [os.path.split(name)[-1].split('.')[0] for name in glob(os.path.join(args.mel_dir,'*.npy'))]
	names = list(set(txt_names) & set(mel_names))
	# pdb.set_trace()
	val_name = random.sample(names,cfg.val_num)
	train_name = list(set(names) - set(val_name))
	train_path = os.path.join(args.file_dir,'filenames_train.json')
	val_path = os.path.join(args.file_dir,'filenames_val.json')
	with open(train_path,'w') as f:
		json.dump(train_name,f)
	with open(val_path,'w') as f:
		json.dump(val_name,f)

class Basedataset(Dataset):
	def __init__(self,args,split):
		self.split = split
		self.args = args
		if split == 'train' or split == 'val':
			if not os.path.exists(args.file_dir) or len(os.listdir(args.file_dir))==0:
				os.makedirs(args.file_dir,exist_ok=True)
				create_train_val_filenames(args)
			self.filenames = self.load_filenames(args,split)
		else:
			if not os.path.exists(args.file_dir):
				self.filenames = create_test_filenames(args)
			else:
				self.filenames = self.load_filenames(args,split)
	def load_filenames(self,args,split):
		if split == 'train':
			path = os.path.join(args.file_dir,'filenames_train.json')
		elif split == 'val':
			path = os.path.join(args.file_dir,'filenames_val.json')
		else:
			path = os.path.join(args.file_dir,'filenames_test.json')
		with open(path,'r') as f:
			names = json.load(f)
		return names
		
	def get_text_mel_pair(self,index):
		name  = self.filenames[index]
		mel_path = os.path.join(self.args.mel_dir,name + '.npy')
		txt_path = os.path.join(self.args.txt_dir,name + '.npy')
		mel = np.load(mel_path)
		txt = np.load(txt_path)
		if mel.shape[-1] == cfg.acoustic_dim:
			mel = mel.T
		return torch.IntTensor(txt),torch.Tensor(mel)

	def __len__(self):
		return len(self.filenames)

class Tacodataset(Basedataset):
	def __init__(self,args,split='train'):
		Basedataset.__init__(self,args,split)
	
	def __getitem__(self, index):
		txt,mel = self.get_text_mel_pair(index)
		if cfg.multi_speaker_training:
			spk = self.load_spk(self,index)
			return txt,mel,spk,index
		else:
			return txt,mel,index
	def load_spk(self,index):
		name = self.filenames[index]
		path = os.path.join(self.data_root,'spks',name+'.npy')
		spk = np.load(path)
		return spk

class Tacocollate():
	def __init__(self):
		self.n_frames_per_step = cfg.n_frames_per_step
	def __call__(self,batch):
		# Right zero-pad all one-hot text sequences to max input length
		input_lengths, ids_sorted_decreasing = torch.sort(
			torch.LongTensor([len(x[0]) for x in batch]),
			dim=0, descending=True)
		max_input_len = input_lengths[0]

		text_padded = torch.LongTensor(len(batch), max_input_len)
		text_padded.zero_()		

		# Right zero-pad mel-spec
		num_mels = batch[0][1].size(0)
		max_target_len = max([x[1].size(1) for x in batch])
		if max_target_len % self.n_frames_per_step != 0:
			max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
			assert max_target_len % self.n_frames_per_step == 0

		# include mel padded and gate padded
		mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
		mel_padded.zero_()
		gate_padded = torch.FloatTensor(len(batch), max_target_len)
		gate_padded.zero_()
		output_lengths = torch.LongTensor(len(batch))
		output_lengths.zero_()
		indexs = torch.LongTensor(len(batch))
		indexs.zero_()
		for i in range(len(ids_sorted_decreasing)):
			text = batch[ids_sorted_decreasing[i]][0]
			text_padded[i, :text.shape[0]] = text
			mel = batch[ids_sorted_decreasing[i]][1]
			mel_padded[i, :, :mel.size(1)] = mel
			gate_padded[i, mel.size(1)-1:] = 1
			output_lengths[i] = mel.size(1)
			indexs[i] = batch[ids_sorted_decreasing[i]][-1]
		
		if cfg.multi_speaker_training:
			spks = torch.FloatTensor(len(batch),batch[0][-1].shape[0])
			for i in range(len(ids_sorted_decreasing)):
				spk = batch[ids_sorted_decreasing[i]][-1]
				spks[i] = spk

			return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, spks, indexs
		else:
			return text_padded, input_lengths, mel_padded, gate_padded, output_lengths,indexs