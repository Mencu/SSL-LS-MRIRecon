import torch
import numpy as np
import data_consistency as dcp
import torch.nn as nn
import torch.nn.functional as F

from svd import CustomSVD
from merlinth.complex import complex2real, real2complex
from PIL import Image
from torch import nn
from fft import NUFFT
import medutils.visualization as mvs

class CNNLayer(nn.Module):
    def __init__(self, in_ch=2, n_f=32):
        super(CNNLayer, self).__init__()
        layers = [
            nn.Conv3d(in_channels=in_ch, out_channels=n_f, kernel_size=3, padding='same', bias=False),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=n_f, out_channels=n_f, kernel_size=3, padding='same', bias=False),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=n_f, out_channels=2, kernel_size=3, padding='same', bias=False),
        ]
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        input2c = complex2real(x)
        out = self.seq(input2c)
        out = real2complex(out)
        return out

class UnrolledNet(nn.Module):
    def __init__(self, in_ch, nb_unroll_blocks, nb_blocks, op, op_adj, out_ch=2, config=None):
        super(UnrolledNet, self).__init__()
        
        if config is None:
            raise Exception(f'The model requires a NUFFT configuration!')

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nb_unroll_blocks = nb_unroll_blocks
        self.nb_blocks = nb_blocks
        self.op = op
        self.op_adj = op_adj

        self.sconvs = nn.ModuleList([CNNLayer(in_ch=4, n_f=32)]) #real imag, M, L

        self.mu_penalty = nn.Parameter(torch.FloatTensor([.05]), requires_grad=True)

        self.thres_coefs = nn.ParameterList([torch.nn.Parameter(torch.tensor(-2, dtype=torch.float32), requires_grad=True) for _ in range(self.nb_unroll_blocks)])
        self.gammas = nn.ParameterList([nn.Parameter(torch.tensor(.5, dtype=torch.float32), requires_grad=True) for _ in range(self.nb_unroll_blocks)])

        # Choose DC layer CG or Wenqi Huang's grad step
        self.dc = dcp.dc_block()
        # self.dc = dcp.DC_wenqi()

    def forward(self, input_x, sens_maps, ktraj, dcomp, trn_mask, loss_mask, init_kspace):

        input_x = input_x.permute(1,0,2,3).contiguous() # (1,30,dim,dim)
        x = input_x

        nb, nt, nx, ny = self.nb, self.nt, self.nx, self.ny =  x.shape

        L = torch.reshape(x, [nb, nt, nx*ny])
        S = torch.zeros_like(x)
        M = L.clone()

        for i in range(self.nb_unroll_blocks):

            # Do denoiser step
            L = self.lowrank(M, self.thres_coefs[i])
            S = self.sparse(M,L, self.sconvs[0])

            # Do data consistency 

            ##########################
            # Wenqi's data consistency
            ##########################
            # rhs = (L+S).reshape([nb,nt,nx,ny])
            # rhs = rhs.permute(1,0,2,3)      # back to 30,1,img_s, img_s

            # dc = self.dc(rhs, sens_maps, ktraj, dcomp, trn_mask, self.mu_penalty, self.op, self.op_adj, init_kspace) # 30,1,img_s, img_s

            # # Prepare DC for update step
            # dc_reshape = dc.permute(1,0,2,3)       # to 1,30,img_s, img_s
            # dc_reshape = dc_reshape.reshape([nb, nt, -1])

            # if i != self.nb_unroll_blocks - 1:
            #     gamma = F.relu(self.gammas[i])
            # else:
            #     gamma = 1.0
            # M = (L+S) - gamma * dc_reshape

            ##########################
            # CG data consistency
            ##########################
            rhs = input_x + self.mu_penalty * (L+S).reshape([nb,nt,nx,ny])
            rhs = rhs.reshape(nb*nt, 1, nx, ny)      # back to 30,1,img_s, img_s

            dc = self.dc(rhs, sens_maps, ktraj, dcomp, trn_mask, self.mu_penalty, self.op, self.op_adj) # input 30,1,img_s, img_s

            # Prepare DC for update step
            dc_reshape = dc.reshape([nb, nt, -1])
            M = dc_reshape

            # DEBUG Print out value ranges
            # print(f'iter {i}: rhs {rhs.abs().min():.3f}-{rhs.abs().max():.3f} dc: {dc_reshape.abs().min():.3f}-{dc_reshape.abs().max():.3f} M: {M.abs().min():.3f}-{M.abs().max():.3f}')

        x = M.reshape(nb, nt, nx, ny).permute(1,0,2,3).contiguous()       #for DC_wenqi into shape (30,1,256,256)

        # IMPORTANT NOTE for supervised: lacking the 'full' kdata mask, because we operate on radial binned data. 
        # Useless for supervised, use only image output and do loss on image domain

        # Works for both supervised and self-supervised
        nw_kspace_output = dcp.SSDU_kspace_transform(x, sens_maps, ktraj, dcomp, loss_mask, self.op, self.op_adj)

        return x, nw_kspace_output, self.mu_penalty, self.thres_coefs

    def lowrank(self, L, thresh):
        Lpre = L
        Ut, St, VtH = CustomSVD.apply(Lpre)
        thres = torch.sigmoid(thresh) * St[:, 0]
        thres = thres.unsqueeze(-1)

        St = F.relu(St - thres)

        St = torch.torch.diag_embed(St)
        St = torch.complex(St, torch.zeros_like(St))

        L = torch.matmul(Ut, St)
        L = torch.matmul(L, VtH)

        return L
    
    def sparse(self, Mpre, L, sconv):
        M_L = [
            torch.reshape(L, [self.nb, self.nt, self.nx, self.ny]),
            torch.reshape(Mpre, [self.nb, self.nt, self.nx, self.ny])
        ]
        M_L = torch.stack(M_L, dim=1)
        
        S = sconv(M_L)      # (batch, 2, time, img_dim, img_dim)
        S = torch.reshape(S, [self.nb, self.nt, self.nx*self.ny])
        
        return S
