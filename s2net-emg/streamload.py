import torch
from torch.utils.data import Dataset
import numpy as np
if torch.cuda.is_available() == True:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
import pandas as pd
import torchvision
import librosa
import scipy
from scipy import signal
from scipy import stats

class EMGStream():

    def __init__(self, fn, bat_size, buff_size, transform,
                 shuffle=False, seed=0):
        if buff_size % bat_size != 0:
            raise Exception("buff_size must be evenly div by bat_size")

        self.bat_size = bat_size
        self.buff_size = buff_size
        self.transform = transform
        self.shuffle = shuffle

        self.rnd = np.random.RandomState(seed)

        self.ptr = 0  # points into x_data and y_data
        self.fin = open(fn, "r")  # line-based text file

        self.the_buffer = []  # list of numpy vectors
        self.xy_mat = None  # NumPy 2-D version of buffer
        self.x_data = None  # predictors as Tensors
        self.y_data = None  # targets as Tensors
        self.reload_buffer()

    def reload_buffer(self):
        self.the_buffer = []
        self.ptr = 0
        ct = 0  # number of lines read
        while ct < self.buff_size:
            line = self.fin.readline()
            if line == "":
                self.fin.seek(0)
                return -1  # reached EOF
            else:
                line = line.strip()  # remove trailing newline
                np_vec = np.fromstring(line, sep="\t")
                self.the_buffer.append(np_vec)
                ct += 1

        if len(self.the_buffer) != self.buff_size:
            return -2

        if self.shuffle == True:
            self.rnd.shuffle(self.the_buffer)  # in-place

        return 0  # buffer successfully loaded

    def __iter__(self):
        return self

    def __next__(self):  # next batch as a tuple
        res = 0
        if self.transform is not None:

            self.the_buffer = np.stack(self.the_buffer)
            self.y_data = int(stats.mode(self.the_buffer[:,-1])[0][0])
            self.x_data = self.transform(self.the_buffer[:,1:9])

        if self.ptr + self.bat_size > self.buff_size:  # reload
            res = self.reload_buffer()
            # 0 = success, -1 = hit eof, -2 = not fully loaded
        if res == 0:
            x = torch.tensor(np.expand_dims(self.x_data, 0))
            y = torch.tensor(self.y_data)

            self.ptr += self.bat_size
            return (x, y)
        # reached end-of-epoch (EOF), so signal no more
        self.reload_buffer()  # prepare for next epoch
        raise StopIteration


class PSD:

    def __init__(self, sr, n_fft):
        self.sr = sr
        self.n_fft = n_fft

    def __call__(self, sig):

        sig = sig.T

        fr, psd_full = signal.welch(sig[0], fs=self.sr, nperseg=75, nfft=self.n_fft)
        psd_full = np.expand_dims(psd_full.T, 0)

        for i in range(1, 8):
            fr, psd_new = signal.welch(sig[i], fs=self.sr, nperseg=75, nfft=self.n_fft)
            psd_new = np.expand_dims(psd_new.T, 0)
            psd_full = np.vstack((psd_full, psd_new))

        feat = psd_full
        feat_list = [feat.T]
        feat_list.append(librosa.feature.delta(feat, order=1).T)
        feats = np.stack(feat_list)
        feats = np.resize(feats, (2, 76, 8))
        return feats


class STFT:

    def __init__(self, n_fft, hop_length):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, sig):
        sig = sig.T

        S = librosa.stft(y=sig,
                         n_fft=self.n_fft,
                         hop_length=self.hop_length,
                         )
        feats = np.dstack((S.real, S.imag))
        feats = np.resize(feats, (8, 76, 52))
        return feats

class PSDNoDelta:

    def __init__(self, sr, n_fft):
        self.sr = sr
        self.n_fft = n_fft

    def __call__(self, sig):

        sig = sig.T

        fr, psd_full = signal.welch(sig[0], fs=self.sr, nfft=self.n_fft)
        psd_full = np.expand_dims(psd_full.T, 0)

        for i in range(1, 8):
            fr, psd_new = signal.welch(sig[i], fs=self.sr, nfft=self.n_fft)
            psd_new = np.expand_dims(psd_new.T, 0)
            psd_full = np.vstack((psd_full, psd_new))

        feat = psd_full
        feat_list = [feat.T]
        feat_list = np.expand_dims(feat_list, 0)
        feats = np.resize(feat_list, (1, 76, 8))
        return feats

class NoFeatureExtraction:

    def __call__(self, sig):
        sig = np.resize(sig, (500, 8))
        return np.expand_dims(sig, 0)

class Rescale:
    def __call__(self, input):
        return librosa.util.normalize(input, axis=2)

