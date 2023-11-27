import torch
import torch.nn as nn
import torch
import torchkbnufft as tkbn
from PIL import Image
import numpy as np

class NUFFTBase(nn.Module):
    def __init__(self, configs=None):
        super(NUFFTBase, self).__init__()
        from warnings import filterwarnings
        filterwarnings("ignore") # ignores floor divide warnings

        device = torch.device('cuda')

        im_size = [configs['img_dim'], configs['img_dim']]
        grid_size = [int(sz * configs['osf']) for sz in im_size]
        self.im_size = im_size
        self.nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size, n_shift=[i//2 for i in im_size]).to(device)
        self.adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size, n_shift=[i//2 for i in im_size]).to(device)
        

class NUFFT_v3(NUFFTBase):
    def forward(self, input, smaps, ktraj, dcomp=None):
        if not torch.is_tensor(dcomp):
            input = torch.from_numpy(input).to('cuda')
            ktraj = torch.from_numpy(ktraj).to('cuda')
            smaps = torch.from_numpy(smaps).to('cuda')
            dcomp = torch.from_numpy(dcomp).to('cuda')
        out = self.nufft_ob(input, ktraj, smaps=smaps, norm='ortho') * torch.sqrt(dcomp)# * torch.sqrt(dcomp)
        return out


class NUFFTAdj_v3(NUFFTBase):
    def forward(self, input, smaps, ktraj, dcomp):
        if not torch.is_tensor(dcomp):
            input = torch.from_numpy(input).to('cuda')
            ktraj = torch.from_numpy(ktraj).to('cuda')
            smaps = torch.from_numpy(smaps).to('cuda')
            dcomp = torch.from_numpy(dcomp).to('cuda')
        out = self.adjnufft_ob(input * torch.sqrt(dcomp), ktraj, smaps=smaps, norm='ortho')
        return out

class NUFFTAdj_coil(NUFFTBase):
    def forward(self, input, smaps, ktraj, dcomp):
        out = self.adjnufft_ob(input * torch.sqrt(dcomp), ktraj, norm='ortho')
        return out

class NUFFT_coil(NUFFTBase):
    def forward(self, input, ktraj):
        out = self.nufft_ob(input, ktraj, norm='ortho')
        return out

class TempFFTOperator(object):
    def forward(self, x):
        return torch.fft.fftshift(torch.fft.fft(x, dim=0, norm='ortho'), dim=0)

    def adjoint(self, x):
        return torch.fft.ifft(torch.fft.ifftshift(x, dim=0), dim=0, norm='ortho')

def ifft2c_mri(k):
    x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, (-2,-1)), norm='ortho'), (-2,-1))
    return x

def fft2c_mri(img):
    k = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(k, (-2,-1)), norm='ortho'), (-2,-1))
    return k

def compute_dcf(ktraj, smap, kweights=None, im_size=None, op=None, op_adj=None, device=None):
    nt, nSpokes = kweights[0], kweights[1]
    nc, nx, ny = smap.shape
    # ktraj = torch.reshape(ktraj, (2, -1))
    # print(f'ktrajjj: {ktraj.shape}')
    dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj, im_size=im_size)
    im_size = [nt, 1, *im_size]
    img = torch.ones(im_size, dtype=torch.complex64).to(device)

    kdata = op(img, smap, ktraj, dcomp).reshape(nt, nc, nSpokes, -1).to(device) # * kweights will lead to a intensity change problem
    img_recon = op_adj(kdata.reshape(nt, nc, -1), smap, ktraj, dcomp)
    ratio = torch.mean(torch.abs(img_recon))
    dcomp = (dcomp / ratio).detach()

    return dcomp
