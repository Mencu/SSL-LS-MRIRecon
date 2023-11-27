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

def save_gif(img, filename, intensity_factor=1, duration=100, loop=0):
    r"""
    Save tensor or ndarray as gif.
    Args: 
        img: tensor or ndarray, (nt, nx, ny)
        filename: string
        intensity_factor: float, intensity factor for normalizing the data
        duration: int, milliseconds between frames
        loop: int, number of loops, 0 means infinite
    """
    if type(img) == torch.Tensor:
        img = img.cpu().numpy()
    if len(img.shape) != 3:
        img = img.squeeze(1)
    assert len(img.shape) == 3
    img = np.abs(img)/np.abs(img).max() * 255 * intensity_factor
    img = [img[i] for i in range(img.shape[0])]
    img = [Image.fromarray(np.clip(im, 0, 255)) for im in img]
    # duration is the number of milliseconds between frames
    img[0].save(filename, save_all=True, append_images=img[1:], duration=100, loop=0)


## Important piece of code
def compute_dcf(ktraj, smap, kweights=None, im_size=None, op=None, op_adj=None, device=None):
    nt, nSpokes = kweights[0], kweights[1]
    nc, nx, ny = smap.shape
    dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj, im_size=im_size)
    im_size = [nt, 1, *im_size]
    img = torch.ones(im_size, dtype=torch.complex64).to(device)

    kdata = op(img, smap, ktraj, dcomp).reshape(nt, nc, nSpokes, -1).to(device) # * kweights will lead to a intensity change problem
    img_recon = op_adj(kdata.reshape(nt, nc, -1), smap, ktraj, dcomp)
    ratio = torch.mean(torch.abs(img_recon))
    dcomp = (dcomp / ratio).detach()            # NOTE is this normalization necessary? 

    return dcomp
