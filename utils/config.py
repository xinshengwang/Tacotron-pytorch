import yaml
import numpy as np
from easydict import EasyDict as edict
from text import symbols
__C = edict()
cfg = __C



# data setting
__C.data_root = ''
__C.file_dir = ''
__C.n_symbols = len(symbols)
__C.symbols_embedding_dim = 512


# Multi_spk_setting
__C.multi_speaker_training = False
__C.spk_embedding_dim = 256

# Data_settings 
__C.min_level_db = -115 # minimum level of dbs when set lower bound in amp_to_db
__C.ref_level_db = 20 # shift db before signal normalization
__C.fmin = 0 # To test depending on dataset
__C.fmax = 8000 # To test depending on dataset, this value is for 16khz wav
__C.sample_rate = 16000  # Default sample rate is 16khz
__C.auto_resample = False # Whether auto re-sample original wavs
__C.rescale = False  # Whether to do waveform energy normalization
__C.rescaling_max = 0.999  # Waveform energy normalization scale
__C.preemphasis = 0.97
__C.trim_silence = False # Whether to clip wav silence at beginning and end
__C.num_silent_frames = 0  # Whether to pad 4 frames on both sides of waveforms
__C.max_acoustic_length = 2000  # Ignore data whose number of frame is more than max_acoustic_length
__C.signal_normalization = True  # Set as true for signal normalization
__C.allow_clipping_in_normalization = True # Spectrograms normalization/scaling and clipping
__C.symmetric_acoustic = True  # Whether to scale the data to be symmetric around 0
# max absolute value of data. If symmetric data will be [-max, max] else [0, max]
__C.max_abs_value = 4. # Absolute value of acoustic-spectrograms
__C.trim_fft_size = 512 # fft size used in trimming
__C.trim_hop_size = 128 # hop size used in trimming
__C.trim_top_db = 30 # top db used in trimming sensitive to each dataset
__C.min_db = -115 # minimum db values which is used in acoustic' normalize
__C.outputs_per_step = 2 # Number of frames at each decoding step
__C.chunk_size = 20000 # chunk size for a single record file
__C.prefetch_batch_num = 10 # Pre-fetch batch number

# Mel_settings
__C.acoustic_dim = 80  # Dimension of acousitc spectrograms
__C.num_freq = 1025  # Only used when adding linear spectrograms post processing network
__C.fft_size = 2048  # FFT size for fourier transforming
__C.n_fft = 2048  # Extra window size is filled with 0 paddings to match this parameter
__C.hop_size = 200 # hop_size between acoustic feature and waveforms
__C.win_size = 800 # For 22050Hz 1100 ~ =  50 ms (If None win_size  =  n_fft)
__C.frame_shift_ms = None # if hop_size is not set can be calculated by this 

# Input_embedding
__C.embedding_dim = 512  # Dimension of embedding space
__C.phone_embedding_dim = 448 # Dimension of phone embedding space
__C.tone_embedding_dim = 64 # Dimension of tone embedding space
__C.seg_tag_embedding_dim = 32 # Dimension of word segment embedding space
__C.prsd_embedding_dim = 32 # Dimension of prosody embedding space

# Attention_mechanism

__C.attention_type = 'GMM' # choice = ['GMM','LSA','DCA','SMA']

# GMM attention
__C.gmm_version = '2' # choice = ['0','1','2']
__C.gmm_num_mixture = 8
__C.gmm_mlp_hidden_dim = 128

# LSA attention
__C.attention_dim = 128
__C.attention_rnn_dim = 1024

__C.lsa_smoothing = False  # Whether to smooth the attention normalization function
__C.lsa_attention_dim = 128  # Dimension of attention space
__C.lsa_attention_filters = 32  # Number of attention convolution filters
__C.lsa_attention_kernel = 31  # Kernel size of attention convolution
__C.lsa_cumulate_weights = True # Whether to cumulate (sum) all previous attention weights

# Stepwise attention
__C.stepwise_attention_depth = 128
__C.stepwise_mode = 'parallel' # Mode of stepwise attention = parallel, hard, sampling


# Encoder parameters
__C.encoder_kernel_size = 5
__C.encoder_n_convolutions = 3
__C.encoder_embedding_dim = 512

# Decoder parameters
__C.n_frames_per_step = 3
__C.decoder_rnn_dim = 1024
__C.prenet_dim = 256
__C.max_decoder_steps = 1000
__C.gate_threshold = 0.5
__C.p_attention_dropout = 0.1
__C.p_decoder_dropout = 0.1

__C.postnet_embedding_dim = 512
__C.postnet_kernel_size = 5
__C.postnet_n_convolutions = 5




# Training_settings
__C.is_cuda = True
__C.pin_mem = True
__C.n_workers = 4
__C.val_num = 16
__C.batch_size = 48  # Number of training samples on each training steps
__C.valid_batch_size = 16 # randomly selected sampling from the training set for validation during training
__C.initial_learning_rate = 2e-3 
__C.betas = (0.9, 0.999)
__C.eps = 1e-6
__C.sch = True # Whether use scheduler to adjust the learning rate True
__C.sch_step = 4000  # iteration for warming up
__C.final_learning_rate = 2e-6  # Minimal learning rate
__C.max_iter = 200000 
__C.weight_decay = 1e-6
__C.grad_clip_thresh = 1.0
__C.mask_padding = True 
__C.p = 1 # mel spec loss penalty 10

__C.reg_weight = 1e-6  # regularization weight (for l2 regularization) -7
__C.use_regularization = True
__C.start_decay = 25000  # Step at which learning decay starts
__C.decay_steps = 25000 # Decay steps
__C.decay_rate = 0.5  # Learning rate decay rate
__C.final_learning_rate = 2e-6  # Minimal learning rate
__C.adam_beta1 = 0.9  # AdamOptimizer beta1 parameter
__C.adam_beta2 = 0.999  # AdamOptimizer beta2 parameter
__C.adam_epsilon = 1e-6  # AdamOptimizer beta3 parameter
__C.gradclip_value = 1.0 # Gradient clipped values
__C.natural_eval = False # Whether to use 100% natural eval (to evaluate Curriculum Learning performance) or with same teacher-forcing ratio as in training (just for overfit)
__C.tacotron_teacher_forcing_mode = 'constant'
__C.tacotron_teacher_forcing_ratio = 1. # Value from [0. 1.] 0. = 0%

# 1. = 100% determines the % of times we force next decoder inputs Only
# relevant if mode = 'constant'
__C.tacotron_teacher_forcing_init_ratio = 1. # initial teacher forcing ratio.

# Relevant if mode = 'scheduled'
__C.tacotron_teacher_forcing_final_ratio = 0. # final teacher forcing ratio.

# Relevant if mode = 'scheduled'
__C.tacotron_teacher_forcing_start_decay = 10000 # starting point of teacher forcing ratio decay. Relevant if mode = 'scheduled'
__C.tacotron_teacher_forcing_decay_steps = 280000 # Default = 280000 Determines

# the teacher forcing ratio decay slope. Relevant if mode = 'scheduled'
__C.tacotron_teacher_forcing_decay_alpha = 0. # Teacher forcing ratio decay rate. Relevant if mode = 'scheduled'
__C.save_training_summary_steps = 500  # Steps between running summary ops
__C.save_val_summary_steps = 1000
__C.save_smaple_steps = 1000 # steps between syntheizing samples
__C.save_checkpoints_steps = 5000  # Steps between writing checkpoints
__C.keep_checkpoint_max = 20  # Maximum keeped model


# inference
__C.groundtruth_alignment = False

__C.power = 1.5
__C.griffin_lim_iters = 60
def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, _ in a.items():
        for k, v in a[k].items():
            # a must specify keys that are in b
            if k not in b:
                raise KeyError('{} is not a valid config key'.format(k))

            # the types must match, too
            old_type = type(b[k])
            if old_type is not type(v):
                if isinstance(b[k], np.ndarray):
                    v = np.array(v, dtype=b[k].dtype)
                else:
                    raise ValueError(('Type mismatch ({} vs. {}) '
                                    'for config key: {}').format(type(b[k]),
                                                                type(v), k))

            # recursively merge dicts
            if type(v) is edict:
                try:
                    _merge_a_into_b(a[k], b[k])
                except:
                    print('Error under config key: {}'.format(k))
                    raise
            else:
                b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""  
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)



