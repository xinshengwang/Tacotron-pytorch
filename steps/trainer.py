import os
import time
import json
import torch
import random
import numpy as np
from data.dataset import Tacodataset, Tacocollate
from utils.config import cfg
from utils.util import mode, to_arr
from data.dataset import create_test_filenames
from torch.utils.data import DataLoader
from utils.logger import Tacotron2Logger
from model.model import Tacotron2, Tacotron2Loss
from utils.audio import save_wav, inv_melspectrogram


def validate(model, criterion, iteration, valdata_loader,logger):
	"""Evaluate on validation set, get validation loss and printing
	"""
	model.eval()
	with torch.no_grad():
		val_loss = 0.0
		for i, batch in enumerate(valdata_loader):
			inputs, targets,_ = model.parse_data_batch(batch)
			predicts = model(inputs)

			# Loss
			loss = criterion(predicts, targets)

			val_loss += loss.item()
		val_loss = val_loss / (i + 1)

	model.train()
	print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
	# logger.sample_training(val_loss, model, targets, predicts, iteration)
	logger.log_validation(val_loss, model, targets, predicts, iteration)


def infer(args):
    	# build model
	file_path = os.path.join(args.file_dir,'filenames_test.json')
	if not os.path.exists(file_path):
		create_test_filenames(args)
	with open(file_path,'r') as f:
		filenames = json.load(f)
	model = Tacotron2(n_vocab=cfg.n_symbols,
                          embed_dim=cfg.symbols_embedding_dim,
                          mel_dim=cfg.acoustic_dim,
                          max_decoder_steps=cfg.max_decoder_steps,
                          stop_threshold=cfg.gate_threshold,
                          r=cfg.n_frames_per_step
                          )
	mode(model, True)

	model, _, _ = load_checkpoint(args.ckpt_pth, model)

	val_loader = prepare_dataloaders(args,'test')
	model.eval()
	# ================ MAIN TRAINNIG LOOP! ===================
	for batch in val_loader:
		x, _, indexs = model.parse_data_batch(batch)
		if not cfg.groundtruth_alignment:
			x = x[0]
		y_pred = model.inference(x)
		for i in range(len(y_pred)):
			length = (y_pred[2][i]<0.5).sum()
			predict_mel = y_pred[1][i][:length,:]
			name = os.path.basename(filenames[i]).split('.')[0]
			save_path = os.path.join(args.wav_path,name + '.wav')
			wav_postnet = inv_melspectrogram(to_arr(predict_mel))
			#save_wav(wav, pth+'.wav')
			save_wav(wav_postnet, save_path)

def infer_train(args,model):
    	# build model
	filepath = os.path.join(args.file_dir,'filenames_val.json')
	with open(filepath,'r') as f:
		filenames = json.load(f)
	model.eval()
	val_loader = prepare_dataloaders(args,'val')
	# ================ MAIN TRAINNIG LOOP! ===================
	for batch in val_loader:
		x, _, indexs = model.parse_data_batch(batch)
		y_pred = model.inference(x[0])
		for i in range(len(y_pred)):
			length = (y_pred[2][i]<0.5).sum()
			predict_mel = y_pred[1][i][:length,:]
			name = os.path.basename(filenames[i]).split('.')[0]
			save_path = os.path.join(args.wav_path,name + '.wav')
			wav_postnet = inv_melspectrogram(to_arr(predict_mel))
			#save_wav(wav, pth+'.wav')
			save_wav(wav_postnet, save_path)


def train(args):
	# build model
	model = Tacotron2(n_vocab=cfg.n_symbols,
                          embed_dim=cfg.symbols_embedding_dim,
                          mel_dim=cfg.acoustic_dim,
                          max_decoder_steps=cfg.max_decoder_steps,
                          stop_threshold=cfg.gate_threshold,
                          r=cfg.n_frames_per_step
                          )
	mode(model, True)
	optimizer = torch.optim.Adam(model.parameters(), lr = cfg.initial_learning_rate,
								betas = cfg.betas, eps = cfg.eps,
								weight_decay = cfg.weight_decay)
	criterion = Tacotron2Loss()
	# load checkpoint
	iteration = 1
	if args.ckpt_pth != '':
		model, optimizer, iteration = load_checkpoint(args.ckpt_pth, model, optimizer)
		iteration += 1 # next iteration is iteration+1

	# get scheduler
	if cfg.sch:
		min_rate = cfg.final_learning_rate / cfg.initial_learning_rate
		lr_lambda = lambda step: max(cfg.sch_step**0.5*min((step+1)*cfg.sch_step**-1.5, (step+1)**-0.5), min_rate)
		if args.ckpt_pth != '':
			scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch = iteration)
		else:
			scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

	# make dataset
	train_loader = prepare_dataloaders(args,'train')
	val_loader = prepare_dataloaders(args,'val')

	# get logger ready
	if args.log_dir != '':
		if not os.path.isdir(args.log_dir):
			os.makedirs(args.log_dir)
		logger = Tacotron2Logger(args.log_dir)

	# get ckpt_dir ready
	if args.ckpt_dir != '' and not os.path.isdir(args.ckpt_dir):
		os.makedirs(args.ckpt_dir)	
	model.train()

	# ================ MAIN TRAINNIG LOOP! ===================
	while iteration <= cfg.max_iter:
		for batch in train_loader:
			if iteration > cfg.max_iter:
				break
			start = time.perf_counter()
			x, y, indexs = model.parse_data_batch(batch)
			y_pred = model(x)

			# loss
			loss = criterion(y_pred, y)
			
			# zero grad
			model.zero_grad()
			
			# backward, grad_norm, and update
			loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_thresh)
			optimizer.step()
			if cfg.sch:
				scheduler.step()
			
			# info
			dur = time.perf_counter()-start
			info = 'Iter: {} Loss: {:.2e} Grad Norm: {:.2e} {:.1f}s/it \n'.format(
				iteration, loss.item(), grad_norm, dur)
			print(info)
			if iteration % 100 == 0:
				with open(args.train_info,'a') as f:
					f.writelines(info)
			# log
			if args.log_dir != '' and (iteration % cfg.save_training_summary_steps == 0):
				learning_rate = optimizer.param_groups[0]['lr']
				logger.log_training(loss.item(), grad_norm, learning_rate, iteration)
			
			# val log
			if args.log_dir != '' and (iteration % cfg.save_val_summary_steps == 0):
				validate(model, criterion, iteration, val_loader, logger)

			# sample 
			if iteration % cfg.save_smaple_steps == 0:
				args.wav_path = os.path.join(args.save_root,'wavs',str(iteration))
				if not os.path.exists(args.wav_path):
					os.makedirs(args.wav_path)
				infer_train(args,model)
				model.train()
				
				
			# save ckpt
			if args.ckpt_dir != '' and (iteration % cfg.save_checkpoints_steps == 0):
				ckpt_pth = os.path.join(args.ckpt_dir, 'ckpt_{}'.format(iteration))
				save_checkpoint(model, optimizer, iteration, ckpt_pth)

			iteration += 1
	if args.log_dir != '':
		logger.close()


def save_checkpoint(model, optimizer, iteration, ckpt_pth):
    torch.save({'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iteration': iteration}, ckpt_pth)

def prepare_dataloaders(args,split):  
	trainset = Tacodataset(args,split)
	collate_fn = Tacocollate()
	if split == 'train':
		data_loader = DataLoader(trainset, num_workers = cfg.n_workers, shuffle = True,
									batch_size = cfg.batch_size, pin_memory = cfg.pin_mem,
									drop_last = True, collate_fn = collate_fn)
	else:
		data_loader = DataLoader(trainset, num_workers = cfg.n_workers, shuffle = True,
									batch_size = cfg.valid_batch_size, pin_memory = cfg.pin_mem,
									drop_last = False, collate_fn = collate_fn)
	return data_loader

def load_checkpoint(ckpt_pth, model, optimizer=None):
	ckpt_dict = torch.load(ckpt_pth)
	model.load_state_dict(ckpt_dict['model'])
	if optimizer !=None:
		optimizer.load_state_dict(ckpt_dict['optimizer'])
	iteration = ckpt_dict['iteration']
	return model, optimizer, iteration

def get_eval_text(args):
	path = os.path.join(args.file_dir,'filenames_val.json')
	with open(path,'r') as f:
		names = json.load(f)
	name = random.sample(names,1)[0]
	txt_path = os.path.join(args.txt_dir,name + '.npy')
	txt = np.load(txt_path)
	return torch.IntTensor(txt)