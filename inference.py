import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
from text import text_to_sequence
from model.model import Tacotron2
from utils.config import cfg_from_file,cfg
from utils.util import mode, to_arr
from utils.audio import save_wav, inv_melspectrogram


def load_model(ckpt_pth):
	ckpt_dict = torch.load(ckpt_pth)
	model = Tacotron2(n_vocab=cfg.n_symbols,
                          embed_dim=cfg.symbols_embedding_dim,
                          mel_dim=cfg.acoustic_dim,
                          max_decoder_steps=cfg.max_decoder_steps,
                          stop_threshold=cfg.gate_threshold,
                          r=cfg.n_frames_per_step
                          )
	model.load_state_dict(ckpt_dict['model'])
	model = mode(model, True).eval()
	return model


def infer(text, model):
	sequence = text_to_sequence(text,['english_cleaners'])
	sequence = mode(torch.IntTensor(sequence)[None, :]).long()
	mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
	return (mel_outputs, mel_outputs_postnet, alignments)


def plot_data(data, figsize = (16, 4)):
	fig, axes = plt.subplots(2, 1, figsize = figsize)
	for i in range(len(data)):
		axes[i].imshow(data[i], aspect = 'auto', origin = 'lower')


def plot(output, pth):
	mel_outputs, mel_outputs_postnet, alignments = output
	plot_data((to_arr(mel_outputs_postnet[0].T),
				to_arr(torch.clamp(alignments[0].T,0,0.8,out=None))))
	plt.savefig(pth+'.png')


def audio(output, pth):
	mel_outputs, mel_outputs_postnet, _ = output
	#wav = inv_melspectrogram(to_arr(mel_outputs[0]))
	wav_postnet = inv_melspectrogram(to_arr(mel_outputs_postnet[0]))
	#save_wav(wav, pth+'.wav')
	save_wav(wav_postnet, pth+'.wav')


def save_mel(output, pth):
	mel_outputs, mel_outputs_postnet, _ = output
	#np.save(pth+'.npy', to_arr(mel_outputs[0]).T)
	np.save(pth+'.npy', to_arr(mel_outputs[0]).T)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-cfg','--cfg_file',default='config/lsa-tts.yaml',
                        help='yaml files for configurations.')
	parser.add_argument('-c', '--ckpt_pth', type = str, default = '/home/work_nfs3/xswang/code/TTS/Tacotron-pytorch/logdir/obama-lsa/logdir/ckpt/ckpt_200000', help = 'path to load checkpoints')
	parser.add_argument('-i', '--img_pth', type = str, default = '/home/work_nfs3/xswang/code/TTS/Tacotron-pytorch/logdir/obama-lsa/logdir/infer/plot_lsa20w',
						help = 'path to save images')
	parser.add_argument('-w', '--wav_pth', type = str, default = '/home/work_nfs3/xswang/code/TTS/Tacotron-pytorch/logdir/obama-lsa/logdir/infer/audio_lsa20w',
						help = 'path to save wavs')
	parser.add_argument('-n', '--npy_pth', type = str, default = '/home/work_nfs3/xswang/code/TTS/Tacotron-pytorch/logdir/obama-lsa/logdir/infer/mel_lsa20w',
						help = 'path to save mels')
	parser.add_argument('-t', '--text', type = str, default = 'It gives me great pleasure to present you the next issue of our academic journal which includes open faculty and research positions in engineering and technology.',
						help = 'text to synthesize')

	args = parser.parse_args()
	"""
	cfg_from_file(args.cfg_file)
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = False
	model = load_model(args.ckpt_pth)
	output = infer(args.text, model)
	if args.img_pth != '':
		plot(output, args.img_pth)
	if args.wav_pth != '':
		audio(output, args.wav_pth)
	if args.npy_pth != '':
		save_mel(output, args.npy_pth)
	"""
	out = np.load('/home/work_nfs3/xswang/data/TTS/obama2/clip/test/tmp/mel/00191_007.npy')
	save_path = '/home/work_nfs3/xswang/data/TTS/obama2/clip/test/tmp/gl_wav/00191_007.wav'
	wav_postnet = inv_melspectrogram(out)
	#save_wav(wav, pth+'.wav')
	save_wav(wav_postnet, save_path)

	