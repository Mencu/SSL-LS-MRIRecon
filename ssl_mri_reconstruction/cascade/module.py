from cgitb import reset
from re import X
import torch
import torchvision
import pytorch_lightning as pl
import numpy as np

from torch import nn
from torch.autograd import Variable
from model import CascadeNet, DnCn3D
from fft import NUFFT_v3, NUFFTAdj_v3
from medutils.measures import psnr, ssim, nmse
from merlinth.complex import complex2real, real2complex
from PIL import Image

GIF_ROOT = '<your_path_here>'
GIF_ROOT_PREDS = '<your_path_here>'

class CascadeModule(pl.LightningModule):
    def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5, op=None, op_adj=None, lr=1e-3):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super().__init__()

        self.n_ch = n_ch
        self.nf = nf
        self.ks = ks
        self.nc = nc
        self.nd = nd
        self.learning_rate = lr

        if op is None or op_adj is None:
            nufft_config = {
                'img_dim': 160,     # image dimension
                'osf': 1.6,         # oversampling factor
                'kernel_width': 3,
                'sector_width': 8,
            }

            op = NUFFT_v3(nufft_config)
            op_adj = NUFFTAdj_v3(configs=nufft_config)

        self.op = op
        self.op_adj = op_adj

        self.model = DnCn3D(
            n_channels=self.n_ch, 
            nf=self.nf, 
            ks=self.ks, 
            nc=self.nc, 
            nd=self.nd, 
            op=self.op,
            op_adj=self.op_adj)

        # self.model = CascadeNet(
        #     n_ch=self.n_ch, 
        #     nf=self.nf, 
        #     ks=self.ks, 
        #     nc=self.nc, 
        #     nd=self.nd, 
        #     op=self.op, 
        #     op_adj=self.op_adj)

        self.criterion = nn.MSELoss()
        self.crop = torchvision.transforms.CenterCrop(160)
'

        self.save_hyperparameters()

                # To save train and validation variables/metrics
        self.trianing_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, img, kdata, ktraj, smap, dcomp, mask):
        res = self.model(img, kdata, ktraj, smap, dcomp, mask)
        return res

    def training_step(self, batch, batch_idx):
        caseid, target_img, img, kdata, ktraj, smap, dcomp, mask = batch
        # must remove batch dimension from data consistency params
        kdata, ktraj, smap, dcomp, mask =  kdata.squeeze(0), ktraj.squeeze(0), smap.squeeze(0), dcomp.squeeze(0), mask.squeeze(0)
        img = self.op_adj(kdata*mask, smap, ktraj, dcomp)
        img = complex2real(img)

        y = self.model(img, kdata, ktraj, smap, dcomp, mask)
        y = self.crop(y)

        if isinstance(self.model,DnCn3D):
            y_complex = real2complex(y) 

        else:
            # I really don't like this, but it is what it is...
            y_complex = torch.view_as_complex(y.permute(0,2,3,4,1).contiguous()).permute(0,3,1,2).contiguous().squeeze(0)

        # TODO make this faster with Kerstin's functions
        if isinstance(self.model,DnCn3D):
            target_img_real = torch.view_as_real(target_img).permute(0,1,4,2,3).squeeze().contiguous()
        else:
            target_img_real = torch.view_as_real(target_img).permute(0,4,2,3,1).contiguous()

        # both need dims (1,2,160,160,30)
        train_loss = self.criterion(y, target_img_real)

        self.log("step/train_loss", train_loss)
        self.trianing_step_outputs.append({'loss': train_loss, 'caseid': caseid})

        return {'loss': train_loss, 'img': y_complex, 'target': target_img.squeeze(0), 'caseid': caseid}

    def validation_step(self, batch, batch_idx):
        caseid, target_img, img, kdata, ktraj, smap, dcomp, mask = batch

        # must remove batch dimension from data consistency params
        kdata, ktraj, smap, dcomp, mask =  kdata.squeeze(0), ktraj.squeeze(0), smap.squeeze(0), dcomp.squeeze(0), mask.squeeze(0)
        if isinstance(self.model,DnCn3D):
            img = img.squeeze()
            img = img.permute(3,0,1,2).contiguous() # make it (30,2,160,160)

        y = self.model(img, kdata, ktraj, smap, dcomp, mask)
        y = self.crop(y)

        if isinstance(self.model,DnCn3D):
            y_complex = real2complex(y) 
        else:
            y_complex = torch.view_as_complex(y.permute(0,2,3,4,1).contiguous()).permute(0,3,1,2).contiguous().squeeze(0)
        
        if self.current_epoch % 5 == 0:
            save_gif(y_complex, GIF_ROOT + f'pred_collapsed_fix_0.5overlap_1osf_{self.current_epoch}.gif')
        
        if isinstance(self.model,DnCn3D):
            target_img_real = torch.view_as_real(target_img).permute(0,1,4,2,3).squeeze().contiguous()
        else:
            target_img_real = torch.view_as_real(target_img).permute(0,4,2,3,1).contiguous()

        val_loss = self.criterion(y, target_img_real)

        self.log("step/val_loss", val_loss)
        self.validation_step_outputs.append({'val_loss': val_loss, 'caseid': caseid})

        return {'val_loss': val_loss, 'img': y_complex, 'target': target_img.squeeze(0), 'caseid': caseid}

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        '''
        Evaluate on complex number tensors the following measures (from medutils) NMSE, SSIM, PSNR
        '''
        # TODO with NRMSE, PSNR and SSIM
        caseid, target_img, img, kdata, ktraj, smap, dcomp, mask = batch
        kdata, ktraj, smap, dcomp =  kdata.squeeze(0), ktraj.squeeze(0), smap.squeeze(0), dcomp.squeeze(0)
        if isinstance(self.model,DnCn3D):
            img = img.squeeze()
            img = img.permute(3,0,1,2).contiguous() # make it (30,2,160,160)

        y = self.model(img, kdata, ktraj, smap, dcomp, mask) 
        y = self.crop(y)

        if isinstance(self.model,DnCn3D):
            y_complex = real2complex(y) #torch.view_as_complex(y.permute(0,2,3,1).contiguous()).squeeze(-1)
        else:
            y_complex = torch.view_as_complex(y.permute(0,2,3,4,1).contiguous()).permute(0,3,1,2).contiguous().squeeze(0)

        save_gif(y_complex, GIF_ROOT_PREDS + f'pred_epoch56_{caseid.detach().cpu().numpy()[0]}.gif')

        target_complex = target_img.squeeze(0)          
        pred, tar = y_complex.detach().cpu().numpy().squeeze(1), target_complex.detach().cpu().numpy() # (tphases, 160, 160)
        # check dims
        print(f'target_complex: {tar.shape}\tpred: {pred.shape}')

        # Adjust intensities of pred to match those of tar, least square minimization (scale * pred + bias)
        # Check if same range first

        nmse_score = nmse(pred, tar, axes = (1,2))
        ssim_score = ssim(pred, tar, axes = (1,2))
        psnr_score = psnr(pred, tar, axes = (1,2))

        self.test_step_outputs.append({'nmse': nmse_score, 'ssim': ssim_score, 'psnr': psnr_score})

        return {'nmse': nmse_score, 'ssim': ssim_score, 'psnr': psnr_score}

    def configure_optimizers(self):
        params = list(self.named_parameters())
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "epoch/val_loss",
            }

        return [optimizer], scheduler

    def on_train_epoch_end(self):
        batch_losses = [x["loss"]for x in self.trianing_step_outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        self.log("epoch/train_loss", epoch_loss)
        self.trianing_step_outputs.clear()

    def on_validation_epoch_end(self):

        batch_losses = [x["val_loss"]for x in self.validation_step_outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        self.log("epoch/val_loss", epoch_loss)

        self.validation_step_outputs.clear()

        return {'val_loss': epoch_loss.item()}

    def on_test_epoch_end(self):
        batch_losses_nmse = [x["nmse"]for x in self.test_step_outputs]         # normalized mean squared error
        batch_losses_ssim = [x["ssim"]for x in self.test_step_outputs]         # structural similarity index  
        batch_losses_psnr = [x["psnr"]for x in self.test_step_outputs]         # peak signal to noise ratio

        epoch_nmse = torch.FloatTensor(batch_losses_nmse).mean()
        epoch_ssim = torch.FloatTensor(batch_losses_ssim).mean()
        epoch_psnr = torch.FloatTensor(batch_losses_psnr).mean()

        print(f'Mean values for:\nepoch_nmse: {epoch_nmse}\nepoch_ssim: {epoch_ssim}\nepoch_psnr: {epoch_psnr}')

        self.test_step_outputs.clear()

        return {'epoch_nmse': epoch_nmse.item(), 'epoch_ssim' : epoch_ssim.item(), 'epoch_psnr' : epoch_psnr.item()}


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
        img = img.detach().cpu().numpy()
    if len(img.shape) != 3:
        img = img.squeeze(1)
    assert len(img.shape) == 3
    img = np.abs(img)/np.abs(img).max() * 255 * intensity_factor
    img = [img[i] for i in range(img.shape[0])]
    img = [Image.fromarray(np.clip(im, 0, 255)) for im in img]
    # duration is the number of milliseconds between frames
    img[0].save(filename, save_all=True, append_images=img[1:], duration=100, loop=0)