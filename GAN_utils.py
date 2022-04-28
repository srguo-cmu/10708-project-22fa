import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
import PIL
import time
import math
import sys
import json
import pickle
import os

import torch
import torch.nn as nn
# import torchaudio

# import torchvision
# import torchvision.utils as vutils
# import torchvision.datasets as dset
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchsummary import summary


from IPython.display import display, Audio, HTML
import glob
import time
import random

import librosa
import soundfile as sf


def create_chunks(chunk_len=4):
    for i in range(19):
        path = "piano/" + str(i).rjust(2, "0") + ".wav"
        x, sr = librosa.load(path , sr=2**14)

    for j in range(int(len(x)/ (chunk_len*2**14) )):
        chunk = x[j*chunk_len*2**14:(j+1)*chunk_len*2**14]
        sf.write("piano/chunks/{}_{}.wav".format( i, j)  , chunk, 2**14)



'''
Call this function to listen to sounds, either raw or generated
x is the numpy array holding the raw waveform
'''
def sound( x, rate, label=''):
    display( HTML( 
    '<style> table, th, td {border: 0px; }</style> <table><tr><td>' + label + 
    '</td><td>' + Audio( x, rate=rate)._repr_html_()[3:] + '</td></tr></table>'
    ))
    
'''
Create custom dataset/loaders for the drum data
'''

class Piano_DS(Dataset):
    def __init__(self, files, sr=2**14, output_len=2**14,
                 subsample_ratio=1, inject_noise_var=0): # sampling rate and waveform length at 4096 or 16384 
        '''
        files: a list of the file names (NO DIRECTORY PREFIXES!) in the dataset 
          # ASSUMES THE WORKING DIRECTORY IS THE '.../10708 Project' folder
          # Make sure the cells at the beginning reflect this
        sr: the resampling rate for librosa to use; 
        output_len: common length for the data samples; should be a power of 2 
        We pad/truncate each raw sample appropriately
        '''
        self.d = output_len
        self.sr = sr


        # self.files = list(os.listdir("spoken_mnist") )
        if subsample_ratio<1:
            self.files = random.sample(files, int(subsample_ratio*len(files)) ) # sample without replacement
        else:
            self.files = files
            
        
        self.n = len(self.files)
        
        self.inject_noise_var = inject_noise_var
        


    def __len__(self):
        return(self.n)
  
    def __getitem__(self, i):
    
        path = self.files[i]
        x, _ = librosa.load(path,sr=self.sr)

        if len(x)>self.d: # truncate if too long
            x=x[:self.d]
        elif len(x)<self.d: # pad if too short
            x = np.pad(x, (0, self.d-len(x)) )

        x /= np.max(x)
        # Architectures take shape (N, C, d)
        # N is number of samples, C is channels (input has 1)
        x=x.reshape(1, -1) # shape (1, d)

        if self.inject_noise_var>0:
            x += np.random.normal(0, np.sqrt(self.inject_noise_var) , size= x.shape )
            x = np.clip(x, -1, 1)
        
        # label doesnt matter here
        label = -1

        return x, label
    
    
def grad_penalty(D, X_real, X_fake, device):
    # Reference : https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead
    # Used to understand the gradient penalty and its implementation.
    # Slightly modified

    # Assumes X_real and X_fake both have the same shape: (m, 1, 4096) or (m, 1, 16384)

    m = X_real.size(0)
    # Sample Epsilon from uniform distribution
    eps = torch.rand(m, 1, 1).to(device)
    eps = eps.expand_as(X_real)

    # Interpolation between real data and fake data.
    interpolation = eps*X_real + (1-eps)*X_fake
    interpolation.requires_grad=True

    # get logits for interpolated data
    interp_logits = D(interpolation)
    grad_outputs = torch.ones_like(interp_logits)

    # Compute Gradients
    gradients = torch.autograd.grad(
      outputs=interp_logits,
      inputs=interpolation,
      grad_outputs=grad_outputs,
      create_graph=True,
      retain_graph=True,
    )[0]

    # Compute and return Gradient Norm
    gradients = gradients.view(m, -1)
    grad_norm = gradients.norm(2, 1)
    return torch.mean((grad_norm - 1) ** 2)



if __name__ == "__main__":
    create_chunks()
