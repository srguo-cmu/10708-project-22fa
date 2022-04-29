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
import glob
import time


from GAN_utils import sound, Piano_DS, grad_penalty
from GANs import Discriminator, Generator, Discriminator_small, Generator_small, Discriminator_new, Generator_new

import sys
import argparse

def train_GAN(epochs, lr, batch_size, z_size, device, G_d, D_d, 
              G_type="old", D_type="old", 
              k=2, l = 10, 
              subsample_ratio=1, inject_noise_var=0,
              save_results=True, pretrained=None):
    
    '''
    epochs: number of epochs to train for
    lr: learning rate
    batch_size: batch size
    z_size: size of the random vector input to generator
    G_d: d parameter of generator
    D_d: d parameter of discriminator
    k: number of times discrim gets trained per epoch
    inject_noise_sd: variance of the random noise gaussian we inject into the data
    
    '''
    lambda_ = l
    assert (G_type=="old" or G_type=="new") and (D_type=="old" or D_type=="new")
    if G_type=="old":
        G = Generator(input_size=z_size, d=G_d, device=device).to(device)
    else:
        G = Generator_new(input_size=z_size, d=G_d, device=device).to(device)
        
    if D_type=="old":
        D = Discriminator(d=D_d, device=device).to(device)
    else:
        D = Discriminator_new(d=D_d, device=device).to(device)
        
#     G = Generator_new(input_size=z_size, d=G_d, device=device).to(device)
#     D = Discriminator(d=D_d, device=device).to(device)
#     G = Generator(input_size=z_size, d=G_d, device=device).to(device)
#     D = Discriminator(d=D_d, device=device).to(device)

    G_losses=[]
    D_losses=[]
    W_losses=[] # wasserstein loss (part of D_loss)
    
    pretrained_epochs = 0
    
    if pretrained is not None:
        G.load_state_dict(pretrained["G_state_dict"])
        D.load_state_dict(pretrained["D_state_dict"])
        G_losses = pretrained["G_losses"]
        D_losses = pretrained["D_losses"]
        W_losses = pretrained["W_losses"]
        pretrained_epochs = pretrained["epochs"]
    
    G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9)) # CHANGE THIS?
    D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    dataset = Piano_DS(files=glob.glob('piano/chunks/*.wav', recursive=True), 
                       sr=4096, output_len=4*4096,
                       subsample_ratio=subsample_ratio, inject_noise_var=inject_noise_var)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cumulated_epochs = pretrained_epochs + epochs
    
    direc = \
        os.path.join("GAN_results_piano", 
                     "{}{}model_4sec_Gd={}_Dd={}_e={}_lr={}_b={}_z={}_k={}_subsamp={}_noisevar={}".format(G_type, D_type, G_d, D_d, 
                                                                                                cumulated_epochs, 
                                                                                                lr, batch_size, z_size, k,
                                                                                                subsample_ratio, inject_noise_var)  )
    
    for epoch in range(pretrained_epochs, cumulated_epochs ):
        start = time.time()

        batch_G_losses = []
        batch_D_losses = []
        batch_W_losses = [] # wasserstein loss (part of D_loss)
        
        for i, batch in enumerate(dataloader):
            X_real = batch[0].to(device)
            m = X_real.size(dim=0)

            # Discriminator update:
            # maximize log(D(x)) + log(1-D(G(z)))
            # equivalent to minimize BCELoss, with real and fake samples
            for _ in range(k):
                z = torch.rand( size=(m ,z_size) ,device=device)*2 - 1 # unif(-1, 1)
                with torch.no_grad(): # make sure the generator is detached
                    X_fake = G(z)
#                     print("FAKE SIZE ", X_fake.size())
                y_pred_real = D(X_real)
                y_pred_fake = D(X_fake)

                D_opt.zero_grad()
                # D_loss = (criterion(y_pred_real, y_real ) + criterion(y_pred_fake, y_fake )) /2 # vanilla GAN loss
                # D_loss = (torch.mean( (y_pred_real-1)**2 ) + torch.mean( (y_pred_fake)**2 ) ) / 2 # LSGAN loss
                D_loss = torch.mean( y_pred_fake - y_pred_real ) + lambda_*grad_penalty(D, X_real, X_fake, device) # WGAN-GP loss
                D_loss.backward()
                D_opt.step()

                # record losses so we can take a mean over the entire epoch
                D_loss_sum = m*D_loss.detach().to("cpu")
                batch_D_losses.append(D_loss_sum)

                with torch.no_grad():
                    W_loss_sum = torch.sum(y_pred_fake - y_pred_real).to("cpu")
                batch_W_losses.append(W_loss_sum)
            # Generator update:
            # minimize log(1-D(G(z)))
            # equivalent to minimize -BCELoss with only fake samples

            z = torch.rand( size=(m ,z_size) ,device=device)*2 - 1 # unif(-1, 1)
            X_fake = G(z)
            y_pred_fake = D(X_fake)

            G_opt.zero_grad()
            # neg_G_loss = -criterion(y_pred_fake, y_fake )
            # G_loss = torch.mean( (y_pred_fake-1)**2 ) # LSGAN loss
            G_loss = -torch.mean( y_pred_fake ) # WGAN-GP loss
            G_loss.backward()
            G_opt.step()

            # record losses so we can take a mean over the entire epoch
            G_loss_sum = m*(G_loss.detach().to("cpu")) 
            batch_G_losses.append(G_loss_sum)    

        G_losses.append( np.sum(batch_G_losses) / len(dataloader.dataset) )
        D_losses.append( np.sum(batch_D_losses) / len(dataloader.dataset) /k )
        W_losses.append( np.sum(batch_W_losses) / len(dataloader.dataset) /k )
        
        print("Finished epoch {}, took {} min".format(epoch, (time.time()-start)/60 ))
        print("G/D/W Losses: {}, {}, {}".format(G_losses[-1], D_losses[-1], W_losses[-1]))

#         if epoch%50 == 49:
#             try:
#                 
#                 gen_sound = G( torch.rand(size=(1,z_size)).to(device)*2-1 ).detach().to("cpu").numpy()[0][0]
#                 plt.figure()
#                 plt.plot(X_fake[0][0])
#                 sound( gen_sound,rate=2**12)
#             except Exception as e:
#                 print(e)
        # saving intermediate results at increments of 100 epochs
        if save_results and epoch%250 == 249:
#             direc = os.path.join(  "GAN_results_piano", "Gd={}_Dd={}_e={}_lr={}_b={}_z={}_k={}".format(G_d, D_d, cumulated_epochs, lr, batch_size, z_size, k)  )
            if not os.path.exists(direc):
                os.makedirs(direc)
            try:
                torch.save(G.state_dict() , os.path.join(direc, "G_{}.pt").format(epoch+1))
                torch.save(D.state_dict() , os.path.join(direc, "D_{}.pt").format(epoch+1))

                loss_dict = {"G": G_losses, "D": D_losses, "W": W_losses}
                with open(os.path.join(direc, "loss_trajecs_{}.pkl".format(epoch+1)), "wb") as f:
                    pickle.dump(loss_dict,f)

            except Exception as e:
                print("Problem saving")
                print(e)

        # save the final results
        if save_results:
#             direc = os.path.join(  "GAN_results_piano", "Gd={}_Dd={}_e={}_lr={}_b={}_z={}_k={}".format(G_d, D_d, cumulated_epochs, lr, batch_size, z_size, k)  )
            if not os.path.exists(direc):
                os.makedirs(direc)

            try:
                torch.save(G.state_dict() , os.path.join(direc, "G_{}.pt".format(cumulated_epochs) ))
                torch.save(D.state_dict() , os.path.join(direc, "D_{}.pt".format(cumulated_epochs) ))

                loss_dict = {"G": G_losses, "D": D_losses, "W": W_losses}
                with open(os.path.join(direc, "loss_trajecs_{}.pkl".format(cumulated_epochs) ), "wb") as f:
                    pickle.dump(loss_dict,f)

            except Exception as e:
                print("Problem saving")
                print(e)

    return G,D, G_losses, D_losses, W_losses



if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="number of epochs", default=500)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.0001)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=64)
    parser.add_argument("--z_size", type=int, help="random vector size in generator", default=400)
    parser.add_argument("--G_d", type=int, help="generator dimensionality parameter", default=64)
    parser.add_argument("--D_d", type=int, help="discriminator dimensionality parameter", default=64)
    parser.add_argument("--G_type", type=str, help="generator type, old vs new", default="old")
    parser.add_argument("--D_type", type=str, help="discriminator type, old vs new", default="old")
    parser.add_argument("--k", type=int, help="number of discriminator sub-epochs per generator epoch", default=5)
    parser.add_argument("--l", type=float, help="lambda weighting on gradient penalty", default=10)
    parser.add_argument("--subsample_ratio", type=float, help="subsampling ratio of data", default=1)
    parser.add_argument("--inject_noise_var", type=float, help="variance of injected gaussian noise to data", default=0)
    parser.add_argument("--save_results", type=bool, help="save the results?", default=True)
#     parser.add_argument("--pretrained_path", type=str, help="number of epochs", default=)
    
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device) # this should be 'cuda'
    assert device==torch.device("cuda")
    
    epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    z_size = args.z_size
    G_d = args.G_d
    D_d = args.D_d
    G_type = args.G_type
    D_type = args.D_type
    k = args.k
    l = args.l
    subsample_ratio = args.subsample_ratio
    inject_noise_var = args.inject_noise_var
    save_results = args.save_results
    
    train_GAN(epochs, lr, batch_size, z_size, device, G_d, D_d, 
              G_type=G_type, D_type=D_type, 
              k=k, l = l, 
              subsample_ratio=subsample_ratio, inject_noise_var=inject_noise_var,
              save_results=save_results, pretrained=None)
    
    