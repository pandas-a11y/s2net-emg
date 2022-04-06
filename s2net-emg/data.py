import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import txt2list
import librosa
import pandas as pd
import scipy
from scipy import signal
from scipy import stats

class EMGDataset(Dataset):
    def __init__(self, data_root, label_dct, mode, transform=None, max_nb_per_class=None):
        
        assert mode in ["train", "test"], 'mode should be "train" or "test"'
        
        self.filenames = []
        self.labels = []
        self.mode = mode
        self.transform = transform
        
        if self.mode == "train":
            testing_list = txt2list(os.path.join(data_root, "testing_list.txt"))
        else:
            testing_list = []

        
        for root, dirs, files in os.walk(data_root):
            for filename in files:
                command = root.split("/")[-1]
                label = label_dct.get(command)
                if label is None:
                    break
                partial_path = '/'.join([command, filename])
                
                testing_file = (partial_path in testing_list)
                training_file = not testing_file
                
                if (self.mode == "test") or (self.mode == "train" and training_file):
                    full_name = os.path.join(root, filename)
                    self.filenames.append(full_name)
                    self.labels.append(label)
                
        if max_nb_per_class is not None:
            
            selected_idx = []
            for label in np.unique(self.labels):
                label_idx = [i for i,x in enumerate(self.labels) if x==label]
                if len(label_idx) < max_nb_per_class:
                    selected_idx += label_idx
                else:
                    selected_idx += list(np.random.choice(label_idx, max_nb_per_class))
            
            self.filenames = [self.filenames[idx] for idx in selected_idx]
            self.labels = [self.labels[idx] for idx in selected_idx]
        
                
        if self.mode == "train":
            label_weights = 1./np.unique(self.labels, return_counts=True)[1]
            label_weights /=  np.sum(label_weights)
            self.weights = torch.DoubleTensor([label_weights[label] for label in self.labels])
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        filename = self.filenames[idx]
        df = pd.read_csv(filename)
        item = np.transpose(np.array(df.loc[:, ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']]))

        
        if self.transform is not None:
            item = self.transform(item)

        label = self.labels[idx]
        
        return item, label

    
class PSD:
    
    def __init__(self, sr, n_fft):
        
        self.sr = sr
        self.n_fft = n_fft
        
        
    def __call__(self, sig):

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

        fr, psd_full = signal.welch(sig[0], fs=self.sr, nperseg=75,  nfft=self.n_fft)
        psd_full = np.expand_dims(psd_full.T, 0)

        for i in range(1, 8):
            fr, psd_new = signal.welch(sig[i], fs=self.sr, nperseg=75, nfft=self.n_fft)
            psd_new = np.expand_dims(psd_new.T, 0)
            psd_full = np.vstack((psd_full, psd_new))

        feat = psd_full
        feat_list = [feat.T]
        feat_list = np.expand_dims(feat_list, 0)
        feats = np.resize(feat_list, (1, 76, 8))
        return feats


class NoFeatureExtraction:
    
    def __call__(self, sig):
        sig = sig.T

        sig = np.resize(sig, (500, 8))
        sig = np.expand_dims(sig, 0)
        return sig

    
class Rescale:
    
    def __call__(self, input):
        
        return librosa.util.normalize(input, axis=2)
