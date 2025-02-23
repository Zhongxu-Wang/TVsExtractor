from scipy.io import wavfile
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import os
import json

matplotlib.use("Agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(data, device):
    (ids, mels, mel_lens, max_mel_len, EMA, pitches) = data

    mels = torch.from_numpy(mels).float().to(device, non_blocking=True)
    mel_lens = torch.from_numpy(mel_lens).to(device, non_blocking=True)
    if EMA is not None:
        EMA = torch.from_numpy(EMA).float().to(device, non_blocking=True)
    pitches = torch.from_numpy(pitches).float().to(device, non_blocking=True).unsqueeze(-1)
    return (ids, mels, mel_lens, max_mel_len, EMA, pitches)

def log(logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/relative_error ", losses[1], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)

def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded
    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])
    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")
        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]
    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])
    return output

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask
