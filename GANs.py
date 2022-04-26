import numpy as np
import time
import math

import torch
import torch.nn as nn
# import torchaudio

# import torchvision
# import torchvision.utils as vutils
# import torchvision.datasets as dset
import torch.nn.functional as F


'''
Modified wavegan
https://arxiv.org/pdf/1802.04208.pdf, page 15
'''
class PhaseShuffle(nn.Module):
    def __init__(self, n, device):
        super(PhaseShuffle, self).__init__()
        self.n = n
        self.device=device

    def forward(self, x):
        # x is (N,C,L)
        # We need to pad by self.n on each side, and shift the last dimension
        n_channels = x.shape[1]
        length = x.shape[2]

        phase_offsets = torch.randint(-self.n, self.n+1, (x.shape[0], n_channels, 1) ) # (N, C, 1)
        phase_offsets = phase_offsets + torch.arange(self.n, self.n+length).view(1, 1, length).type(torch.int64) # sum with broadcasting, (N, C, L)
        phase_offsets = phase_offsets.to(self.device)
        # phase_offsets = phase_offsets.type(torch.int64)
        # reflection pad and shift by offsets
        out = nn.functional.pad(x, (self.n, self.n), mode="reflect")
        out = torch.gather(out, dim=2 , index=phase_offsets)

        return out


class Generator(nn.Module):
    def __init__(self, input_size, d, device ):
        '''
        d is a hyperparam to adjust the size of the model
        '''
        super(Generator, self).__init__()

        self.d = d
        # Input of (N, 100)
        self.dense = nn.Linear(input_size, 256*d) # (N, 256d)
        # in between, reshape to (N, 16d,16)
        self.relu0 = nn.ReLU()
        self.tconv1 = nn.ConvTranspose1d(in_channels=16*d, 
                                         out_channels=8*d, 
                                         kernel_size=24, 
                                         stride=4, 
                                         padding=10)  # (N, 8d, 64)
        self.relu1 = nn.ReLU()

        self.tconv2 = nn.ConvTranspose1d(in_channels=8*d, 
                                         out_channels=4*d, 
                                         kernel_size=24, 
                                         stride=4, 
                                         padding=10)  # (N, 4d, 256)
        self.relu2 = nn.ReLU()

        self.tconv3 = nn.ConvTranspose1d(in_channels=4*d, 
                                         out_channels=2*d, 
                                         kernel_size=24, 
                                         stride=4, 
                                         padding=10)  # (N, 2d, 1024)
        self.relu3 = nn.ReLU()

        self.tconv4 = nn.ConvTranspose1d(in_channels=2*d, 
                                         out_channels=d, 
                                         kernel_size=24, 
                                         stride=4, 
                                         padding=10)  # (N, d, 4096)
        self.relu4 = nn.ReLU()

        self.tconv5 = nn.ConvTranspose1d(in_channels=d, 
                                         out_channels=1, 
                                         kernel_size=24, 
                                         stride=4, 
                                         padding=10)  # (N, 1, 16384)
        # map output values to (-1, +1)
        self.tanh_output = nn.Tanh()
    
    def forward(self,x):
        out = self.relu0(self.dense(x).view(-1, 16*self.d, 16))
        out = self.relu1(self.tconv1(out))
        out = self.relu2(self.tconv2(out))
        out = self.relu3(self.tconv3(out))
        out = self.relu4(self.tconv4(out))
        out = self.tanh_output(self.tconv5(out))

        return out
    
'''
New discriminator/generators with architectural modifications
'''

class D_StackDilateBlock(nn.Module):
    def __init__(self, in_c, out_c_per, kernel_size, stride, paddings, dilations, device):
        super(D_StackDilateBlock, self).__init__()
        
        assert len(paddings)==len(dilations)
        out_c_per = int(out_c_per)
        self.paddings= paddings
        
        self.layers = \
            [ nn.Conv1d(in_channels=in_c, 
                        out_channels=out_c_per, 
                        kernel_size=kernel_size, 
                        stride=stride,
                        padding=0, # pad one side in forward pass 
                        dilation=dilations[i]).to(device)
                for i in range(len(dilations)) ]
        
    def forward(self, x):
        outs = [conv( F.pad(x, pad=(self.paddings[i],0)) ) for \
                i,conv in enumerate(self.layers)] # each has shape (N, out_c_per, M)
#         for o in outs: print(o.shape)
        out = torch.cat(outs, dim=1)
        
        return out

def calc_pad(lin, lout, stride, k, d):
    return math.ceil( ((lout -1)*stride + 1 - lin + d*(k-1)) ) 

class Discriminator_new(nn.Module): # technically a critic; real data should get higher score than fake data
    def __init__(self, d, device):
        super(Discriminator_new, self).__init__()
        # input (N, 1, 16384)
        assert d%4 == 0
        self.d = d

#         self.conv1 = nn.Conv1d(in_channels=1, 
#                                out_channels=d, 
#                                kernel_size=24, 
#                                stride=4,
#                                padding=10)
        dilations = [1,3,7,15]
        paddings = [calc_pad(lin=2**14, lout=2**12, stride=4, k=24, d=dil) for dil in dilations]
        self.dsdb1 = D_StackDilateBlock(in_c=1, 
                                        out_c_per=d/4, 
                                        kernel_size=24, 
                                        stride=4, 
                                        paddings=paddings, 
                                        dilations=dilations, 
                                        device=device).to(device)
        
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.ps1 = PhaseShuffle(n=4, device=device)

#         self.conv2 = nn.Conv1d(in_channels=d, 
#                                out_channels=2*d, 
#                                kernel_size=24, 
#                                stride=4,
#                                padding=10)
        dilations = [1,3,7,15]
        paddings = [calc_pad(lin=2**12, lout=2**10, stride=4, k=24, d=dil) for dil in dilations]
        self.dsdb2 = D_StackDilateBlock(in_c=d, 
                                        out_c_per=2*d/4, 
                                        kernel_size=24, 
                                        stride=4, 
                                        paddings=paddings, 
                                        dilations=dilations, 
                                        device=device).to(device)

        self.lrelu2 = nn.LeakyReLU(0.2)
        self.ps2 = PhaseShuffle(n=4, device=device)

#         self.conv3 = nn.Conv1d(in_channels=2*d, 
#                                out_channels=4*d, 
#                                kernel_size=24, 
#                                stride=4,
#                                padding=10)

        dilations = [1,3,7,15]
        paddings = [calc_pad(lin=2**10, lout=2**8, stride=4, k=24, d=dil) for dil in dilations]
        self.dsdb3 = D_StackDilateBlock(in_c=2*d, 
                                        out_c_per=4*d/4, 
                                        kernel_size=24, 
                                        stride=4, 
                                        paddings=paddings, 
                                        dilations=dilations, 
                                        device=device).to(device)

        self.lrelu3 = nn.LeakyReLU(0.2)
        self.ps3 = PhaseShuffle(n=4, device=device)

#         self.conv4 = nn.Conv1d(in_channels=4*d, 
#                                out_channels=8*d, 
#                                kernel_size=24, 
#                                stride=4,
#                                padding=10)

        dilations = [1,3,7,15]
        paddings = [calc_pad(lin=2**8, lout=2**6, stride=4, k=24, d=dil) for dil in dilations]
        self.dsdb4 = D_StackDilateBlock(in_c=4*d, 
                                        out_c_per=8*d/4, 
                                        kernel_size=24, 
                                        stride=4, 
                                        paddings=paddings, 
                                        dilations=dilations, 
                                        device=device).to(device)

        self.lrelu4 = nn.LeakyReLU(0.2)
        self.ps4 = PhaseShuffle(n=4, device=device)


#         self.conv5 = nn.Conv1d(in_channels=8*d, 
#                                out_channels=16*d, 
#                                kernel_size=24, 
#                                stride=4,
#                                padding=10)
        dilations = [1,3]
        paddings = [calc_pad(lin=2**6, lout=2**4, stride=4, k=24, d=dil) for dil in dilations]
        self.dsdb5 = D_StackDilateBlock(in_c=8*d, 
                                        out_c_per=16*d/2, 
                                        kernel_size=24, 
                                        stride=4, 
                                        paddings=paddings, 
                                        dilations=dilations, 
                                        device=device).to(device)

        self.lrelu5 = nn.LeakyReLU(0.2)
        # reshape to (N, 256d) in between
        self.flat = nn.Flatten()
        self.dense = nn.Linear(256*d, 1)

    def forward(self, x):
        out = self.ps1( self.lrelu1( self.dsdb1(x) ) )
        out = self.ps2( self.lrelu2( self.dsdb2(out) ) )
        out = self.ps3( self.lrelu3( self.dsdb3(out) ) )
        out = self.ps4( self.lrelu4( self.dsdb4(out) ) )
        out = self.lrelu5( self.dsdb5(out) ) 

        out = self.dense( self.flat(out) )

        return out

def calc_negpad_transpose(lin, lout, stride, k, d):
    return (lin-1)*stride -lout + d*(k-1)+1

class G_StackDilateBlock(nn.Module):
    def __init__(self, in_c, out_c_per, kernel_size, stride, neg_paddings, dilations, device):
        super(G_StackDilateBlock, self).__init__()
        
        assert len(neg_paddings)==len(dilations)
        out_c_per = int(out_c_per)
        self.neg_paddings= neg_paddings
        
        self.layers = \
            [ nn.ConvTranspose1d(in_channels=in_c, 
                        out_channels=out_c_per, 
                        kernel_size=kernel_size, 
                        stride=stride,
                        padding=0, # negatively pad one side in forward pass 
                        dilation=dilations[i]).to(device)
                for i in range(len(dilations)) ]
        
    def forward(self, x):
        outs = [conv_t( x )[:,:, int(self.neg_paddings[i]): ] for \
                i,conv_t in enumerate(self.layers)] # each has shape (N, out_c_per, M)
#         for o in outs: print(o.shape)
        out = torch.cat(outs, dim=1)
        
        return out    
    
class Generator_new(nn.Module):
    def __init__(self, input_size, d, device ):
        '''
        d is a hyperparam to adjust the size of the model
        '''
        super(Generator_new, self).__init__()

        self.d = d
        # Input of (N, 100)
        self.dense = nn.Linear(input_size, 256*d) # (N, 256d)
        # in between, reshape to (N, 16d,16)
        self.relu0 = nn.ReLU()
#         self.tconv1 = nn.ConvTranspose1d(in_channels=16*d, 
#                                          out_channels=8*d, 
#                                          kernel_size=24, 
#                                          stride=4, 
#                                          padding=10)  # (N, 8d, 64)
        dilations = [1,3,7,15]
        neg_paddings = [calc_negpad_transpose(lin=2**4, lout=2**6, stride=4, k=24, d=dil) for dil in dilations]
        self.gsdb1 = G_StackDilateBlock(in_c=16*d, 
                                        out_c_per=8*d/2, 
                                        kernel_size=24, 
                                        stride=4, 
                                        neg_paddings=neg_paddings, 
                                        dilations=dilations, 
                                        device=device).to(device)

        self.relu1 = nn.ReLU()

#         self.tconv2 = nn.ConvTranspose1d(in_channels=8*d, 
#                                          out_channels=4*d, 
#                                          kernel_size=24, 
#                                          stride=4, 
#                                          padding=10)  # (N, 4d, 256)

        dilations = [1,3,7,15]
        neg_paddings = [calc_negpad_transpose(lin=2**6, lout=2**8, stride=4, k=24, d=dil) for dil in dilations]
        self.gsdb2 = G_StackDilateBlock(in_c=8*d, 
                                        out_c_per=4*d/4, 
                                        kernel_size=24, 
                                        stride=4, 
                                        neg_paddings=neg_paddings, 
                                        dilations=dilations, 
                                        device=device).to(device)

        self.relu2 = nn.ReLU()

#         self.tconv3 = nn.ConvTranspose1d(in_channels=4*d, 
#                                          out_channels=2*d, 
#                                          kernel_size=24, 
#                                          stride=4, 
#                                          padding=10)  # (N, 2d, 1024)

        dilations = [1,3,7,15]
        neg_paddings = [calc_negpad_transpose(lin=2**8, lout=2**10, stride=4, k=24, d=dil) for dil in dilations]
        self.gsdb3 = G_StackDilateBlock(in_c=4*d, 
                                        out_c_per=2*d/4, 
                                        kernel_size=24, 
                                        stride=4, 
                                        neg_paddings=neg_paddings, 
                                        dilations=dilations, 
                                        device=device).to(device)

        self.relu3 = nn.ReLU()

#         self.tconv4 = nn.ConvTranspose1d(in_channels=2*d, 
#                                          out_channels=d, 
#                                          kernel_size=24, 
#                                          stride=4, 
#                                          padding=10)  # (N, d, 4096)

        dilations = [1,3,7,15]
        neg_paddings = [calc_negpad_transpose(lin=2**10, lout=2**12, stride=4, k=24, d=dil) for dil in dilations]
        self.gsdb4 = G_StackDilateBlock(in_c=2*d, 
                                        out_c_per=d/4, 
                                        kernel_size=24, 
                                        stride=4, 
                                        neg_paddings=neg_paddings, 
                                        dilations=dilations, 
                                        device=device).to(device)

        self.relu4 = nn.ReLU()

#         self.tconv5 = nn.ConvTranspose1d(in_channels=d, 
#                                          out_channels=1, 
#                                          kernel_size=24, 
#                                          stride=4, 
#                                          padding=10)  # (N, 1, 16384)

        dilations = [1,3]
        neg_paddings = [calc_negpad_transpose(lin=2**12, lout=2**14, stride=4, k=24, d=dil) for dil in dilations]
        self.gsdb5 = G_StackDilateBlock(in_c=d, 
                                        out_c_per=d/4, 
                                        kernel_size=24, 
                                        stride=4, 
                                        neg_paddings=neg_paddings, 
                                        dilations=dilations, 
                                        device=device).to(device)
        
        self.last_conv = nn.Conv1d(in_channels=d,
                                   out_channels=1,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)
        
        # map output values to (-1, +1)
        self.tanh_output = nn.Tanh()
    
    def forward(self,x):
        out = self.relu0(self.dense(x).view(-1, 16*self.d, 16))
        out = self.relu1(self.gsdb1(out))
        out = self.relu2(self.gsdb2(out))
        out = self.relu3(self.gsdb3(out))
        out = self.relu4(self.gsdb4(out))
        out = self.tanh_output(self.last_conv(self.gsdb5(out)))

        return out

class Discriminator(nn.Module): # technically a critic; real data should get higher score than fake data
    def __init__(self, d, device):
        super(Discriminator, self).__init__()
        # input (N, 1, 16384)
        self.d = d

        self.conv1 = nn.Conv1d(in_channels=1, 
                               out_channels=d, 
                               kernel_size=24, 
                               stride=4,
                               padding=10)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.ps1 = PhaseShuffle(n=4, device=device)

        self.conv2 = nn.Conv1d(in_channels=d, 
                               out_channels=2*d, 
                               kernel_size=24, 
                               stride=4,
                               padding=10)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.ps2 = PhaseShuffle(n=4, device=device)

        self.conv3 = nn.Conv1d(in_channels=2*d, 
                               out_channels=4*d, 
                               kernel_size=24, 
                               stride=4,
                               padding=10)
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.ps3 = PhaseShuffle(n=4, device=device)

        self.conv4 = nn.Conv1d(in_channels=4*d, 
                               out_channels=8*d, 
                               kernel_size=24, 
                               stride=4,
                               padding=10)
        self.lrelu4 = nn.LeakyReLU(0.2)
        self.ps4 = PhaseShuffle(n=4, device=device)


        self.conv5 = nn.Conv1d(in_channels=8*d, 
                               out_channels=16*d, 
                               kernel_size=24, 
                               stride=4,
                               padding=10)
        self.lrelu5 = nn.LeakyReLU(0.2)
        # reshape to (N, 256d) in between
        self.flat = nn.Flatten()
        self.dense = nn.Linear(256*d, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.ps1( self.lrelu1( self.conv1(x) ) )
        out = self.ps2( self.lrelu2( self.conv2(out) ) )
        out = self.ps3( self.lrelu3( self.conv3(out) ) )
        out = self.ps4( self.lrelu4( self.conv4(out) ) )
        out = self.lrelu5( self.conv5(out) ) 
        # out = self.sigmoid( self.dense( self.flat(out) ) )
        out = self.dense( self.flat(out) )

        return out

    
    
    
'''
"Small" versions of G and D
Audio length and sampling rate of 4096 instead of 16384
'''
class Generator_small(nn.Module): 
    def __init__(self, input_size, d, device ):
        '''
        d is a hyperparam to adjust the size of the model
        '''
        super(Generator_small, self).__init__()

        self.d = d
        # Input of (N, 100)
        self.dense = nn.Linear(input_size, 256*d) # (N, 256d)
        # in between, reshape to (N, 16d,16)
        self.relu0 = nn.ReLU()
        self.tconv1 = nn.ConvTranspose1d(in_channels=16*d, 
                                         out_channels=8*d, 
                                         kernel_size=24, 
                                         stride=4, 
                                         padding=10)  # (N, 8d, 64)
        self.relu1 = nn.ReLU()

        self.tconv2 = nn.ConvTranspose1d(in_channels=8*d, 
                                         out_channels=4*d, 
                                         kernel_size=24, 
                                         stride=4, 
                                         padding=10)  # (N, 4d, 256)
        self.relu2 = nn.ReLU()

        self.tconv3 = nn.ConvTranspose1d(in_channels=4*d, 
                                         out_channels=2*d, 
                                         kernel_size=24, 
                                         stride=4, 
                                         padding=10)  # (N, 2d, 1024)
        self.relu3 = nn.ReLU()

        self.tconv4 = nn.ConvTranspose1d(in_channels=2*d, 
                                         out_channels=1, 
                                         kernel_size=24, 
                                         stride=4, 
                                         padding=10)  # (N, 1, 4096)
        # self.relu4 = nn.ReLU()

        # self.tconv5 = nn.ConvTranspose1d(in_channels=d, 
        #                                  out_channels=1, 
        #                                  kernel_size=24, 
        #                                  stride=4, 
        #                                  padding=10)  # (N, 1, 16384)
        # map output values to (-1, +1)
        self.tanh_output = nn.Tanh()
    
    def forward(self,x):
        out = self.relu0(self.dense(x).view(-1, 16*self.d, 16))
        out = self.relu1(self.tconv1(out))
        out = self.relu2(self.tconv2(out))
        out = self.relu3(self.tconv3(out))
        out = self.tanh_output(self.tconv4(out))

        return out


class Discriminator_small(nn.Module): # technically a critic
    def __init__(self, d, device):
        super(Discriminator_small, self).__init__()
        # input (N, 1, 4096)
        self.d = d

        self.conv1 = nn.Conv1d(in_channels=1, 
                               out_channels=d, 
                               kernel_size=24, 
                               stride=4,
                               padding=10)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.ps1 = PhaseShuffle(n=2, device=device)

        self.conv2 = nn.Conv1d(in_channels=d, 
                               out_channels=2*d, 
                               kernel_size=24, 
                               stride=4,
                               padding=10)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.ps2 = PhaseShuffle(n=2, device=device)

        self.conv3 = nn.Conv1d(in_channels=2*d, 
                               out_channels=4*d, 
                               kernel_size=24, 
                               stride=4,
                               padding=10)
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.ps3 = PhaseShuffle(n=2, device=device)

        self.conv4 = nn.Conv1d(in_channels=4*d, 
                               out_channels=8*d, 
                               kernel_size=24, 
                               stride=4,
                               padding=10)
        self.lrelu4 = nn.LeakyReLU(0.2)
        # self.ps4 = PhaseShuffle(n=2, device=device)


        # self.conv5 = nn.Conv1d(in_channels=8*d, 
        #                        out_channels=16*d, 
        #                        kernel_size=24, 
        #                        stride=4,
        #                        padding=10)
        # self.lrelu5 = nn.LeakyReLU(0.2)
        # reshape to (N, 128d) in between
        self.flat = nn.Flatten()
        self.dense = nn.Linear(128*d, 1) # sigmoid score outputs
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.ps1( self.lrelu1( self.conv1(x) ) )
        out = self.ps2( self.lrelu2( self.conv2(out) ) )
        out = self.ps3( self.lrelu3( self.conv3(out) ) )
        # out = self.ps4( self.lrelu4( self.conv4(out) ) )
        # out = self.lrelu5( self.conv5(out) ) 
        out = self.lrelu4( self.conv4(out) )
        # out = self.sigmoid( self.dense( self.flat(out) ) )
        out = self.dense( self.flat(out) )

        return out