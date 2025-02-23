import os
import json
import math
import torch
import pickle
import random
import librosa
import torchaudio
import numpy as np
from torch import nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from tools import pad_1D, pad_2D

to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

_pad = " "
_letters = 'abcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
symbols = [_pad] + list(_letters) + list(_letters_ipa)
dicts = {s:i for i,s in enumerate(symbols)}

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            indexes.append(self.word_index_dictionary[char])
        return indexes

text_cleaner = TextCleaner()


class DatasetEMA(Dataset):
    def __init__(self, train_config, sort=False, drop_last=False, file_list=None):
        self.root_dir = train_config["path"]["preprocessed_path"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.sort = sort
        self.drop_last = drop_last
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(os.path.join(self.root_dir, self.file_list[idx]), 'rb') as f:
            data = pickle.loads(f.read())
        EMA = torch.from_numpy(data["EMA"])
        Mel = data["Mel"]
        F0 = torch.from_numpy(data["f0"])

        crop_length = int((EMA.shape[1]-20)*(random.random()**0.5))+20
        random_start = np.random.randint(0, EMA.shape[1] - crop_length)
        EMA = EMA[:, random_start:random_start+crop_length]
        Mel = Mel[:, random_start:random_start+crop_length]
        F0 = F0[:, random_start:random_start+crop_length]

        mask = torch.isnan(F0)
        F0[mask] = 0
        F0 = F0.squeeze()

        sample = {"id": self.file_list[idx], "F0": F0, "Mel": Mel.transpose(1,0), "EMA": EMA.transpose(1,0),}
        return sample

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        mels = [data[idx]["Mel"] for idx in idxs]
        EMA = [data[idx]["EMA"] for idx in idxs]
        F0 = [data[idx]["F0"] for idx in idxs]
        mel_lens = np.array([mel.shape[0] for mel in mels])
        mels = pad_2D(mels)
        EMA = pad_2D(EMA)
        F0 = pad_1D(F0)
        return (ids,mels,mel_lens,max(mel_lens),EMA,F0)

    def collate_fn(self, data):
        data_size = len(data)
        if self.sort:
            len_arr = np.array([d["Mel"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)
        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]
        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))
        return output


class Datasetsys(Dataset):
    def __init__(
        self, batch_size, sort=False, drop_last=False, file_list = False):
        self.sort = sort
        self.drop_last = drop_last
        self.file_list = file_list
        self.batch_size = batch_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load("../"+self.file_list[idx])
        if wav.shape[0] == 2:
            wav = wav[0,:].squeeze()
        else:
            wav = wav.squeeze()
        if sr != 24000:
            wav = librosa.resample(wav.numpy(), orig_sr=sr, target_sr=24000)

        F0, _, _ = librosa.pyin(wav.numpy(), fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), win_length = 1200, hop_length = 300)
        Mel = to_mel(wav)
        Mel = ((torch.log(1e-5 + Mel.unsqueeze(0)) - mean) / std).squeeze()
        F0 = torch.from_numpy(F0)

        mask = torch.isnan(F0)
        F0[mask] = 0
        F0 = F0.squeeze()

        sample = {"id": self.file_list[idx], "F0": F0, "Mel": Mel.transpose(1,0)}
        return sample

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        mels = [data[idx]["Mel"] for idx in idxs]
        F0 = [data[idx]["F0"] for idx in idxs]
        mel_lens = np.array([mel.shape[0] for mel in mels])
        mels = pad_2D(mels)
        EMA = None
        F0 = pad_1D(F0)
        return (ids,mels,mel_lens,max(mel_lens),EMA,F0)

    def collate_fn(self, data):
        data_size = len(data)
        if self.sort:
            len_arr = np.array([d["Mel"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)
        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]
        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))
        return output
