import scipy
import librosa
import librosa.filters
import numpy as np
from scipy.io import wavfile
from utils.config import cfg


def load_wav(path):
	sr, wav = wavfile.read(path)
	wav = wav.astype(np.float32)
	wav = wav/np.max(np.abs(wav))
	try:
		assert sr == cfg.sample_rate
	except:
		print('Error:', path, 'has wrong sample rate.')
	return wav


def save_wav(wav, path):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	wavfile.write(path, cfg.sample_rate, wav.astype(np.int16))


def preemphasis(x):
	return scipy.signal.lfilter([1, -cfg.preemphasis], [1], x)


def inv_preemphasis(x):
	return scipy.signal.lfilter([1], [1, -cfg.preemphasis], x)


def spectrogram(y):
	D = _stft(preemphasis(y))
	S = _amp_to_db(np.abs(D)) - cfg.ref_level_db
	return _normalize(S)


def inv_spectrogram(spectrogram):
	'''Converts spectrogram to waveform using librosa'''
	S = _db_to_amp(_denormalize(spectrogram) + cfg.ref_level_db)	# Convert back to linear
	return inv_preemphasis(_griffin_lim(S ** cfg.power))			# Reconstruct phase

def melspectrogram(y):
	D = _stft(preemphasis(y))
	S = _amp_to_db(_linear_to_mel(np.abs(D))) - cfg.ref_level_db
	return _normalize(S)


def inv_melspectrogram(spectrogram):
	mel = _db_to_amp(_denormalize(spectrogram) + cfg.ref_level_db)
	S = _mel_to_linear(mel)
	return inv_preemphasis(_griffin_lim(S ** cfg.power))


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
	window_length = int(cfg.sample_rate * min_silence_sec)
	hop_length = int(window_length / 4)
	threshold = _db_to_amp(threshold_db)
	for x in range(hop_length, len(wav) - window_length, hop_length):
		if np.max(wav[x:x+window_length]) < threshold:
			return x + hop_length
	return len(wav)


def _griffin_lim(S):
	'''librosa implementation of Griffin-Lim
	Based on https://github.com/librosa/librosa/issues/434
	'''
	angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
	S_complex = np.abs(S).astype(np.complex)
	y = _istft(S_complex * angles)
	for i in range(cfg.griffin_lim_iters):
		angles = np.exp(1j * np.angle(_stft(y)))
		y = _istft(S_complex * angles)
	return y


def _stft(y):
	n_fft, hop_length, win_length = _stft_parameters()
	return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
	_, hop_length, win_length = _stft_parameters()
	return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
	return (cfg.num_freq - 1) * 2, cfg.hop_size, cfg.win_size


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis()
	return np.dot(_mel_basis, spectrogram)
	

def _mel_to_linear(spectrogram):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis()
	inv_mel_basis = np.linalg.pinv(_mel_basis)
	if spectrogram.shape[0] != cfg.acoustic_dim:
		spectrogram = spectrogram.transpose(1,0)
	inverse = np.dot(inv_mel_basis, spectrogram)
	inverse = np.maximum(1e-10, inverse)
	return inverse


def _build_mel_basis():
	n_fft = (cfg.num_freq - 1) * 2
	return librosa.filters.mel(cfg.sample_rate, n_fft, n_mels=cfg.acoustic_dim, fmin = cfg.fmin, fmax = cfg.fmax)

def _amp_to_db(x):
	return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
	return np.power(10.0, x * 0.05)

def _normalize(S):
	return np.clip((S - cfg.min_level_db) / -cfg.min_level_db, 0, 1)

def _denormalize(S):
	return ((np.clip(S, -4, 4) + 4) * -cfg.min_level_db / (2 * 4)) + cfg.min_level_db