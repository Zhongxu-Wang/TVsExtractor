import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from conformer.conformer.encoder import ConformerBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Permute(nn.Module):
    def __init__(self, dim1, dim2):
        super(Permute, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        x = x.transpose(self.dim1,self.dim2)
        return x

class EMA_Predictor(nn.Module):
    def __init__(self):
        super(EMA_Predictor, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(82,256),
            Permute(1,2),
            nn.BatchNorm1d(256),
            Permute(1,2),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.block2 = nn.ModuleList([ConformerBlock(
            encoder_dim=256,
            num_attention_heads=4,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            feed_forward_dropout_p=0.05,
            attention_dropout_p=0.05,
            conv_dropout_p=0.05,
            conv_kernel_size=31,
            half_step_residual=True,
        ) for _ in range(5)])

        self.pool = weight_norm(nn.ConvTranspose1d(256, 256, kernel_size=3, stride=2, groups=256, padding=1, output_padding=1))

        self.block3 = nn.LSTM(input_size=256,hidden_size=256,num_layers=1,dropout=0,bidirectional =True)

        self.block4 = nn.Sequential(
            nn.Linear(512,128),
            Permute(1,2),
            nn.BatchNorm1d(128),
            Permute(1,2),
            nn.ReLU(),
            nn.Linear(128,10),
        )

    def get_mask_from_lengths(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
        return mask

    def log_norm(self, x, mean=-4, std=4, dim=2):
        """
        normalized log mel -> mel -> norm -> log(norm)
        """
        x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
        return x

    def forward(self, mels, mel_lens, max_mel_len, EMA, F0, synthesize = False):
        energy = self.log_norm(mels).unsqueeze(2)

        mel_masks = (
            self.get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        features = torch.cat((F0, energy, mels),2)

        dec_output = self.block1(features)

        de_masks = mel_masks.unsqueeze(2).expand_as(dec_output)

        for layer in self.block2:
            dec_output = layer(dec_output)
            dec_output = dec_output.masked_fill(de_masks, 0)
        dec_output, (_, _) = self.block3(dec_output)
        outputs = self.block4(dec_output)

        out_masks = mel_masks.unsqueeze(2).expand_as(outputs)
        if synthesize:
            return [outputs, mel_lens, F0, energy]
        return [outputs, out_masks]
