import torch
import torch.nn as nn
import torch
import torchkbnufft as tkbn
from PIL import Image
import numpy as np


class NUFFT(nn.Module):
    def __init__(self, nuffttype='optox', configs=None):
        super(NUFFT, self).__init__()
        self.nuffttype = nuffttype
        if nuffttype == 'optox':
            self.op = optn.GpuNufft(**configs)
        elif nuffttype == 'torchkbnufft':
            from warnings import filterwarnings
            filterwarnings("ignore") # ignore floor divide warnings

            im_size = [configs['img_dim'], configs['img_dim']]
            grid_size = [int(sz * configs['osf']) for sz in im_size]
            self.nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)#.to(device)
            self.adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size)#.to(device)
        
    def forward(self, input, smaps, ktraj, dcomp, inv=False):
        if self.nuffttype == 'optox':
            if not inv:
                out = self.op.forward(input, smaps, ktraj, dcomp)
            else:
                out = self.op.backward(input, smaps, ktraj, dcomp)
        elif self.nuffttype == 'torchkbnufft':
            if not inv:
                input = input.unsqueeze(1)  # dim0: use phases as 'batch' of gpunufft; dim1: add dim for 'coil'

                # img: nt(for batch), nc(1), nx, ny
                # ktraj: nt(for batch), 2, nSpokes*nFE
                # smaps: nc, nx, ny
                ktraj = torch.roll(ktraj, shifts=1, dims=1)
                out = self.nufft_ob(input, ktraj*np.pi*2, smaps=smaps, norm='ortho') * torch.sqrt(dcomp)

            else:
                ktraj = torch.roll(ktraj, shifts=1, dims=1)
                out = self.adjnufft_ob(input*torch.sqrt(dcomp), ktraj*np.pi*2, smaps=smaps, norm='ortho')
                out = out.squeeze(1)    #squeeze 'coil' dim
        return out

class NUFFTBase(nn.Module):
    def __init__(self, configs=None):
        super(NUFFTBase, self).__init__()
        from warnings import filterwarnings
        filterwarnings("ignore") # ignores floor divide warnings

        device = torch.device('cuda')

        im_size = [configs['img_dim'], configs['img_dim']] # 160x160
        grid_size = [int(sz * configs['osf']) for sz in im_size] # 256x256
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
        # print(f'input: {input.shape}\tdcomp: {dcomp.shape}\tktraj: {ktraj.shape}\tsmaps {smaps.shape}')
        out = self.adjnufft_ob(input * torch.sqrt(dcomp), ktraj, smaps=smaps, norm='ortho')
        return out