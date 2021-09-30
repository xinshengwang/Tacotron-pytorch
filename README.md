# Tacotron2-PyTorch

PyTorch implementation of Tacotron2 (https://arxiv.org/pdf/1712.05884.pdf) with various variants.

## Supported attention mechanisms

* GMM Attention (GMM)
* Location Sensitive Attention (LAS)
* Dynamic Convolution Attention (DCA)
* StepwiseMonotonicAttention (SMA)

## Supported Tacotron Variants

* Tacotron2
* GST-Tacotron 
* VAE-Tacotron 

## Dataset

This code is validated on a found data, which is extracted from Obama's talking videos, including around 11 hours. Currently, only English is supported.

## Setup

* Text processing

    Refer to ```bash/text-to-seq``` to preprocess the input transcriptions. In this script, ```save_dir ``` will be the text input dir in the training script. 
    ```
    python text_processing.py \
        --txt_dir=${the path to the text dir} \
        --save_dir=${dir to save texts_seq}
    ``` 
* Training

    This code supports kinds of attention mechanisms and reference embedding methods. You can define the specific model within a hyperparameter config file. In ```/bash```, there are already several set hyperparameter files.  You can follow the script ```bash/gst-train.sh``` to start the training.
    ```
    python main.py \
        --cfg_file= config/gst-tts.yaml \
        --txt_dir=/path to/texts_seq \
        --mel_dir=/path to/mels \
        --file_dir=/path to filename dir \
        --save_root=/logdir \
        --train 
    ```
    Note that "--file_dir" is a directory to save/load filenames for training and test. You don't have to manually create these files. If not filename file exists, this code will create them.  

## Results

Here are alignment results achieved by this code based on LSA attention mechanism  and GMM attention mechanism respectively:

* LSA
    
    ![lsa](/pic/lsa.jpg)

* GMM
    
    ![gmm](/pic/gmm.jpg)

## wave reconstruction

* Griffin-Lim was supported

## References
This project is highly based on the work below.
- [Tacotron by thuhcsi](https://github.com/thuhcsi/tacotron)
- [Tacotron2 by BogiHsu](https://github.com/BogiHsu/Tacotron2-PyTorch)
- [Attentions-in-Tacotron by LeoniusChen](https://github.com/LeoniusChen/Attentions-in-Tacotron)
