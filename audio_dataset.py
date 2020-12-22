import pandas as pd
import numpy as np

import soundfile as sf
import librosa

import torch
from torch.utils.data import Dataset

def denormalize(x:torch.tensor):
    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225])
    return x * std[...,None,None] + mean[...,None,None]

class AudioDataset(Dataset):
    def __init__(self,df,path,period=30,tfms=None,stats=None,eps=1e-6):
        self.df = df
        self.tfms = tfms
        self.period = period
        self.path = path
        self.stats = stats
        if self.stats:
            self.mean,self.std = stats
        self.eps = eps
        
    def __len__(self):
        return self.df.shape[0]
        
    def __getitem__(self,idx):
        item_p = self.df.iloc[idx].filename
        item_p = self.path + '/' + item_p
        x,sr = sf.read(item_p)
        
        if self.tfms:
            x = self.tfms(x)
        
        else:
            len_x = len(x)
            effective_len = sr * self.period
            if len_x < effective_len:
                new_x = np.zeros(effective_len,dtype=x.dtype)
                start = np.random.randint(effective_len - len_x)
                new_x[start:start + len_x] = x
                x = new_x.astype(np.float32)

            elif len_x > effective_len:
                start = np.random.randint(len_x - effective_len)
                x = x[start:start+effective_len].astype(np.float32)

            else:
                x = x.astype(np.float32)
        
        y = self.df.iloc[idx].tsh,self.df.iloc[idx].t3,self.df.iloc[idx].t4
        if self.stats:
            x = self.normalize(x)
        return (x,(y,item_p))
    
    def normalize(self,x):
        return (x - self.mean) / (self.std + self.eps)
    
    
class MelSpectrogramDataset(Dataset):
    def __init__(self,df,path,period=30,wave_tfms=None,mel_tfms=None,img_tfms=None,stats=None,eps=1e-6):
        self.df = df
        self.path = path
        self.period = period
        self.wave_tfms,self.mel_tfms,self.img_tfms = wave_tfms,mel_tfms,img_tfms
        self.stats = stats
        if self.stats:
            self.mean, self.std = self.stats
        self.eps = eps
    
    def __getitem__(self,idx):
        item_p = self.df.iloc[idx].filename
        item_p = self.path + '/' + item_p
        x,sr = sf.read(item_p)
        
        if self.wave_tfms:
            x = self.wave_tfms(x)
        
        else:
            len_x = len(x)
            effective_len = sr * self.period
            if len_x < effective_len:
                new_x = np.zeros(effective_len,dtype=x.dtype)
                start = np.random.randint(effective_len - len_x)
                new_x[start:start + len_x] = x
                x = new_x.astype(np.float32)

            elif len_x > effective_len:
                start = np.random.randint(len_x - effective_len)
                x = x[start:start+effective_len].astype(np.float32)

            else:
                x = x.astype(np.float32)
        
        melspec = librosa.feature.melspectrogram(x,sr=sr,n_mels=128)
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        
        if self.mel_tfms:
            melspec = self.mel_tfms(melspec)
        
        image = self.bw_to_color(melspec)
        if self.img_tfms:
            aug = self.img_tfms(
                image = image,
            )
            image = aug['image']
        y = (
            self.df.iloc[idx].tsh.astype(np.float32),
            self.df.iloc[idx].t3.astype(np.float32),
            self.df.iloc[idx].t4.astype(np.float32)
        )
#         if self.stats:
#             x = self.normalize(x)
        return (image,(y,item_p))
    
    def bw_to_color(self,x):
        x = np.stack([x,x,x],axis=-1)
        mean = x.mean()
        xstd = x.std()
        x = (x - mean) / (xstd + self.eps)
        _min, _max = x.min(),x.max()
        if (_max - _min) > self.eps:
            x[x < _min] = _min
            x[x > _max] = _max
            x = 255 * (x - _min) / (_max - _min)
            x = x.astype(np.uint8)
        else:
            x = np.zeros_like(x,dtype=np.uint8)
        return x
        
    
    def __len__(self):
        return self.df.shape[0]
    
    
class MelSpectrogram_Classification_Dataset(Dataset):
    def __init__(self,df,path,period=30,wave_tfms=None,mel_tfms=None,img_tfms=None,stats=None,eps=1e-6):
        self.df = df
        self.path = path
        self.period = period
        self.wave_tfms,self.mel_tfms,self.img_tfms = wave_tfms,mel_tfms,img_tfms
        self.stats = stats
        if self.stats:
            self.mean, self.std = self.stats
        self.eps = eps
    
    def __getitem__(self,idx):
        item_p = self.df.iloc[idx].filename
        item_p = self.path + '/' + item_p
        x,sr = sf.read(item_p)
        
        if self.wave_tfms:
            x = self.wave_tfms(x)
        
        else:
            len_x = len(x)
            effective_len = sr * self.period
            if len_x < effective_len:
                new_x = np.zeros(effective_len,dtype=x.dtype)
                start = np.random.randint(effective_len - len_x)
                new_x[start:start + len_x] = x
                x = new_x.astype(np.float32)

            elif len_x > effective_len:
                start = np.random.randint(len_x - effective_len)
                x = x[start:start+effective_len].astype(np.float32)

            else:
                x = x.astype(np.float32)
        
        melspec = librosa.feature.melspectrogram(x,sr=sr,n_mels=128)
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        
        if self.mel_tfms:
            melspec = self.mel_tfms(melspec)
        
        image = self.bw_to_color(melspec)
        if self.img_tfms:
            aug = self.img_tfms(
                image = image,
            )
            image = aug['image']
        y = (
            self.df.iloc[idx].tsh_label,
            self.df.iloc[idx].t3_label,
            self.df.iloc[idx].t4_label
        )
#         y = (
#             self._onehot(2,self.df.iloc[idx].tsh_label),
#             self._onehot(2,self.df.iloc[idx].t3_label),
#             self._onehot(2,self.df.iloc[idx].t4_label)
#         )
        #y = self.df.iloc[idx].tsh,self.df.iloc[idx].t3,self.df.iloc[idx].t4
#         if self.stats:
#             x = self.normalize(x)
        return (image,(y,item_p))
    
    def bw_to_color(self,x):
        x = np.stack([x,x,x],axis=-1)
        mean = x.mean()
        xstd = x.std()
        x = (x - mean) / (xstd + self.eps)
        _min, _max = x.min(),x.max()
        if (_max - _min) > self.eps:
            x[x < _min] = _min
            x[x > _max] = _max
            x = 255 * (x - _min) / (_max - _min)
            x = x.astype(np.uint8)
        else:
            x = np.zeros_like(x,dtype=np.uint8)
        return x
        
    def _onehot(self,size,target):
        vec = torch.zeros(size,dtype=torch.float32)
        vec[target] = 1.
        return vec
    
    def __len__(self):
        return self.df.shape[0]