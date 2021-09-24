from attention.attention_base import *
from attention.dca import DynamicConvolutionAttention
from attention.gmm import GMMAttention
from attention.sma import StepwiseMonotonicAttention

def attention_mechanism(att):
    if att == 'GMM':
        return GMMAttention
    elif att == 'LSA':
        return LocationSensitiveAttention
    elif att == 'DCA':
        return DynamicConvolutionAttention
    elif att == 'SMA':
        return StepwiseMonotonicAttention
    else:
        raise('wrong attention mechanism type')