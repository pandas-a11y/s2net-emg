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

    
class FMD: # 0

    def __call__(self, sig):
        feats_fmd = np.vstack([np.expand_dims(signal.periodogram(i)[1]*1/2, 0) for i in sig])

        feats = feats_fmd.T
        feats = np.expand_dims(feats, 0)
        return feats

class MMDF: # 1

    def __call__(self, sig):
        feats_fmd = np.vstack([np.expand_dims(np.sqrt(signal.periodogram(i)[1])*np.sqrt(1/len(i)/2)*1/2, 0) for i in sig])

        feats = feats_fmd.T
        feats = np.expand_dims(feats, 0)
        return feats

class MAV: # 2

    def __call__(self, sig):
        feats_fmd = np.vstack(np.expand_dims([np.abs(i)*1/len(i) for i in sig], 0))

        feats = feats_fmd.T
        feats = np.expand_dims(feats, 0)
        return feats

class RMS: # 3

    def __init__(self, window):
        self.window = window

    def __call__(self, sig):
        sig2 = np.power(sig, 2)
        window = np.ones(self.window) / float(self.window)
        #feats = np.vstack([np.expand_dims(np.convolve(i, window, 'valid'), 0) for i in sig2])
        #feats = np.expand_dims(feats, 2) # -1 = 8:141:1, 0 = 1,8,141
        feats = np.vstack([np.expand_dims(np.convolve(i, window, 'valid'), 0) for i in sig])
        feats = feats.T
        feats = np.expand_dims(feats, 0)

        return feats
    
class PSD: # 4

    def __init__(self, sr, n_fft, window):
        self.sr = sr
        self.n_fft = n_fft
        self.nperseg = window

    def __call__(self, sig):
        feats = np.vstack([np.expand_dims(signal.welch(i, fs=self.sr, nperseg=self.nperseg,  nfft=self.n_fft)[1], 0) for i in sig])
        feats = feats.T
        feats = np.expand_dims(feats, 0)
        return feats




class Rescale:
    
    def __call__(self, input):
        
        return librosa.util.normalize(input, axis=2)
