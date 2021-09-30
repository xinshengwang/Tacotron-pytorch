""" adapted from https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py """

import torch
from torch import nn
from model.layers import ReferenceEncoder
from utils.config import cfg

class GST(nn.Module):
    """
    GlobalStyleToken (GST)

    GST is described in:
        Y. Wang, D. Stanton, Y. Zhang, R.J. Shkerry-Ryan, E. Battenberg, J. Shor, Y. Xiao, F. Ren, Y. Jia, R.A. Saurous,
        "Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis,"
        in Proceedings of the 35th International Conference on Machine Learning (PMLR), 80:5180-5189, 2018.
        https://arxiv.org/abs/1803.09017

    See:
        https://github.com/KinglittleQ/GST-Tacotron/blob/master/GST.py
        https://github.com/NVIDIA/mellotron/blob/master/modules.py
    """

    def __init__(self, mel_dim, gru_units=cfg.gst_token_embed_dim,
                 conv_channels=[32, 32, 64, 64, 128, 128], kernel_size=3, stride=2, padding=1,
                 num_tokens=10, token_embed_dim=256, num_heads=8):
        super().__init__()

        self.encoder = ReferenceEncoder(mel_dim, gru_units, conv_channels, kernel_size, stride, padding)
        self.stl = StyleTokenLayer(gru_units, num_tokens, token_embed_dim, num_heads)

    def forward(self, inputs, input_lengths=None):
        """
        input:
            inputs --- [B, T, mel_dim]
            input_lengths --- [B]
        output:
            style_embed --- [B, 1, token_embed_dim]
        """

        ref_embed = self.encoder(inputs, input_lengths=input_lengths)  # [B, gru_units]
        style_embed = self.stl(ref_embed.unsqueeze(1))  # [B, 1, gru_units] -> [B, 1, token_embed_dim]

        return style_embed

class StyleTokenLayer(nn.Module):
    """
    StyleTokenLayer (STL)
        - A bank of style token embeddings
        - An attention module
    """

    def __init__(self, query_dim, num_tokens=10, token_embed_dim=256, num_heads=8):
        super(StyleTokenLayer, self).__init__()

        # style token embeddings
        self.embeddings = nn.Parameter(torch.FloatTensor(num_tokens, token_embed_dim // num_heads))
        nn.init.normal_(self.embeddings, mean=0, std=0.5)

        # multi-head attention
        d_q = query_dim
        d_k = token_embed_dim // num_heads
        self.attention = MultiHeadAttention(d_q, d_k, d_k, token_embed_dim, num_heads)

    def forward(self, inputs):
        """
        input:
            inputs --- [B, 1, query_dim]
        output:
            style_embed --- [B, 1, token_embed_dim]
        """

        B = inputs.size(0)
        query = inputs  # [B, 1, query_dim]
        keys = torch.tanh(self.embeddings).unsqueeze(0).expand(B, -1, -1)  # [B, num_tokens, token_embed_dim // num_heads]
        style_embed = self.attention(query, keys, keys)  # [B, 1, token_embed_dim]

        return style_embed

    def from_token(self, token_scores):
        """
        Get style embedding by specifying token_scores

        input:
            token_scores --- [B, 1, num_tokens]
        output:
            style_embed --- [B, 1, token_embed_dim]
        """

        B = token_scores.size(0)
        tokens = torch.tanh(self.embeddings).unsqueeze(0).expand(B, -1, -1)  # [B, num_tokens, token_embed_dim // num_heads]
        tokens = self.attention.W_value(tokens)  # [B, num_tokens, token_embed_dim]
        style_embed = torch.matmul(token_scores, tokens)  # [B, 1, token_embed_dim]

        return style_embed


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention
    """

    def __init__(self, query_dim, key_dim, val_dim, num_units, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.split_size = num_units // num_heads
        self.scale_factor = key_dim ** 0.5

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=val_dim, out_features=num_units, bias=False)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, query, key, value):
        """
        input:
            query --- [B, T_q, query_dim]
            key   --- [B, T_k, key_dim]
            value --- [B, T_k, val_dim]
        output:
            out --- [B, T_q, num_units]
        """

        querys = self.W_query(query)  # [B, T_q, num_units]
        keys   = self.W_key(key)      # [B, T_k, num_units]
        values = self.W_value(value)  # [B, T_k, num_units]

        querys = torch.stack(torch.split(querys, self.split_size, dim=2), dim=0)  # [h, B, T_q, num_units/h]
        keys   = torch.stack(torch.split(keys,   self.split_size, dim=2), dim=0)  # [h, B, T_k, num_units/h]
        values = torch.stack(torch.split(values, self.split_size, dim=2), dim=0)  # [h, B, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3)) / self.scale_factor  # [h, B, T_q, T_k]
        scores = self.softmax(scores)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, B, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [B, T_q, num_units]

        return out
