import torch
import pytorch_lightning as pl
import numpy as np
import torchvision
import pandas as pd
import pickle as pkl

from prodigyopt import Prodigy
from torch import nn
from torchvision.transforms import CenterCrop
from model import UnrolledNet
from fft import NUFFT_v3, NUFFTAdj_v3
from medutils.measures import psnr, ssim, nmse
from PIL import Image

from merlinth.complex import complex2real, real2complex

class SSLModule(pl.LightningModule):
    def __init__(self, n_ch=2, nc=5, nd=5, op=None, op_adj=None, lr=1e-3, config=None, self_supervised=True,
        gif_root=None, gif_root_test=None, pred_path=None, pd_path=None):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super().__init__()

        self.n_ch = n_ch
        self.nc = nc
        self.nd = nd
        self.learning_rate = lr
        self.self_supervised = self_supervised
        self.gif_root = gif_root  
        self.gif_root_test = gif_root_test
        self.pred_path = pred_path 
        self.pd_path = pd_path

        if op is None or op_adj is None:
            config = {
                'img_dim': 256,     # image dimension
                'osf': 1,         # oversampling factor
                'kernel_width': 3,
                'sector_width': 8,
            }

            op = NUFFT_v3(config)
            op_adj = NUFFTAdj_v3(configs=config)

        self.op = op
        self.op_adj = op_adj

        self.model = UnrolledNet(
            in_ch=self.n_ch, 
            nb_unroll_blocks=self.nc, 
            nb_blocks=self.nd, 
            op=self.op, 
            op_adj=self.op_adj,
            out_ch=2,
            config=config
        )

        self.crop = CenterCrop(160)
        self.save_hyperparameters()
        if not self.self_supervised:
            self.loss = nn.MSELoss()

        # To save train and validation variables/metrics
        self.trianing_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.predict_step_outputs = []

    def forward(self, img, kdata, ktraj, smap, dcomp, mask):
        res = self.model(img, kdata, ktraj, smap, dcomp, mask)
        return res

    def training_step(self, batch, batch_idx):
        caseid, casename, target_img, kdata, ktraj, smap, dcomp, trn_mask, loss_mask = batch

        # must remove batch dimension from data consistency params
        target_img, kdata, ktraj, smap, dcomp = target_img.squeeze(0), kdata.squeeze(0), ktraj.squeeze(0), smap.squeeze(0), dcomp.squeeze(0)
        trn_mask, loss_mask = trn_mask.squeeze(0), loss_mask.squeeze(0)

        input_kdata = kdata * trn_mask
        ref_kspace_tensor = kdata * loss_mask
        img = self.op_adj(input_kdata, smap, ktraj, dcomp)

        out_img, nw_kspace_output, mu_penalty, thresh = self.model(img, smap, ktraj, dcomp, trn_mask, loss_mask, kdata) # _ params are rhs and penalties

        out_img_complex = self.crop(out_img)

        if self.self_supervised:
            # Calculate criterion on the kspace
            train_loss = 0.5 * (torch.norm(ref_kspace_tensor - nw_kspace_output) / torch.norm(ref_kspace_tensor)) + \
                0.5 * torch.norm(ref_kspace_tensor - nw_kspace_output, p=1) / torch.norm(ref_kspace_tensor, p=1)
        else:
            # Calculate criterion in image domain - MSELoss
            out_img_real = complex2real(out_img_complex)
            target_img = complex2real(target_img.unsqueeze(1))
            target_img = self.crop(target_img)
            train_loss = self.loss(out_img_real, target_img) 

        self.log("step/train_loss", train_loss)

        self.trianing_step_outputs.append({'loss': train_loss, 'caseid': caseid})

        return train_loss

    def validation_step(self, batch, batch_idx):
        caseid, casename, target_img, kdata, ktraj, smap, dcomp, remainder_mask, val_mask = batch

        # must remove batch dimension from data consistency params
        target_img, kdata, ktraj, smap, dcomp = target_img.squeeze(0), kdata.squeeze(0), ktraj.squeeze(0), smap.squeeze(0), dcomp.squeeze(0)
        remainder_mask, val_mask = remainder_mask.squeeze(0), val_mask.squeeze(0)

        input_kdata = kdata * remainder_mask
        ref_kspace_tensor = kdata * val_mask
        img = self.op_adj(input_kdata, smap, ktraj, dcomp)

        out_img, nw_kspace_output, mu_penalty, thresh = self.model(img, smap, ktraj, dcomp, remainder_mask, val_mask, kdata)

        out_img_complex = self.crop(out_img)

        if self.current_epoch % 2 == 0:
            save_gif(out_img_complex, self.gif_root + f'pred_out_img_complex_{self.current_epoch}.gif')
        
        if self.self_supervised:
            # Calculate criterion on the kspace
            val_loss = 0.5 * (torch.norm(ref_kspace_tensor - nw_kspace_output) / torch.norm(ref_kspace_tensor)) + \
                0.5 * (torch.norm(ref_kspace_tensor - nw_kspace_output, p=1) / torch.norm(ref_kspace_tensor, p=1))
        else:
            # Calculate criterion in image domain - MSELoss
            out_img_real = complex2real(out_img_complex)
            target_img = complex2real(target_img.unsqueeze(1))
            target_img = self.crop(target_img)
            val_loss = self.loss(out_img_real, target_img)

        self.log("step/val_loss", val_loss)
        self.validation_step_outputs.append({'val_loss': val_loss, 'caseid': caseid})

        return val_loss

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        '''
        Evaluate on complex number tensors the following measures (from medutils) NMSE, SSIM, PSNR
        '''
        caseid, casename, target_img, kdata, ktraj, smap, dcomp, remainder_mask, val_mask = batch
        kdata, ktraj, smap, dcomp = kdata.squeeze(0), ktraj.squeeze(0), smap.squeeze(0), dcomp.squeeze(0)
        remainder_mask, val_mask = remainder_mask.squeeze(0), val_mask.squeeze(0)

        input_kdata = kdata * remainder_mask

        img = self.op_adj(input_kdata, smap, ktraj, dcomp)

        out_img, nw_kspace_output, mu_penalty, thresh = self.model(img, smap, ktraj, dcomp, remainder_mask, val_mask, kdata)

        # Center crop before comparing in image domain
        out_img_complex = self.crop(out_img)
        out_img_complex = out_img_complex.squeeze(1)

        save_gif(out_img_complex, self.gif_root_test + f'inference_{casename[0]}.gif')

        target_complex = target_img.squeeze(0)
        target_complex = self.crop(target_complex)

        pred, tar = out_img_complex.detach().cpu().numpy(), target_complex.detach().cpu().numpy() # (tphases, 160, 160)

        comp_list = [pred]
        ref = tar
        new_list = normalize(comp_list=comp_list, ref=ref)
        norm_pred = new_list[0]

        nmse_score = nmse(norm_pred, tar, axes = (1,2))
        ssim_score = ssim(norm_pred, tar, axes = (1,2))
        psnr_score = psnr(norm_pred, tar, axes = (1,2))

        self.test_step_outputs.append({'nmse': nmse_score, 'ssim': ssim_score, 'psnr': psnr_score})

        return {'nmse': nmse_score, 'ssim': ssim_score, 'psnr': psnr_score}


    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        caseid, casename, target_img, kdata, ktraj, smap, dcomp, remainder_mask, val_mask = batch
        kdata, ktraj, smap, dcomp = kdata.squeeze(0), ktraj.squeeze(0), smap.squeeze(0), dcomp.squeeze(0)
        remainder_mask, val_mask = remainder_mask.squeeze(0), val_mask.squeeze(0)

        input_kdata = kdata * remainder_mask
        img = self.op_adj(input_kdata, smap, ktraj, dcomp)

        out_img, nw_kspace_output, mu_penalty, thresh = self.model(img, smap, ktraj, dcomp, remainder_mask, val_mask, kdata)

        # Center crop before comparing in image domain
        out_img_complex = self.crop(out_img)
        out_img_complex = out_img_complex.squeeze(1)

        target_complex = target_img.squeeze(0)
        target_complex = self.crop(target_complex)

        save_gif(target_complex, self.pred_path + f'pred_{casename[0]}.gif')
        pred, tar = out_img_complex.detach().cpu().numpy(), target_complex.detach().cpu().numpy() # (tphases, 160, 160)

        save_dict = {'casename': casename[0], 'pred': pred, \
                      'target': tar, 'remainder_mask': remainder_mask.detach().cpu().numpy(),\
                      'val_mask': val_mask.detach().cpu().numpy()}
        
        with open(self.pred_path + casename[0] + '.pkl', 'wb') as f:
            pkl.dump(save_dict, f)

        nmse_score = nmse(pred, tar, axes = (1,2))
        ssim_score = ssim(pred, tar, axes = (1,2))
        psnr_score = psnr(pred, tar, axes = (1,2))

        self.predict_step_outputs.append({'casename': casename[0], 'nmse': nmse_score, 'ssim': ssim_score, 'psnr': psnr_score})
        
        return {'nmse': nmse_score, 'ssim': ssim_score, 'psnr': psnr_score}


    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.1)
        optimizer = Prodigy(self.model.parameters(), lr=1.)

        return [optimizer]#, scheduler

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

        epoch_nmse_std = torch.FloatTensor(batch_losses_nmse).std()
        epoch_ssim_std = torch.FloatTensor(batch_losses_ssim).std()
        epoch_psnr_std = torch.FloatTensor(batch_losses_psnr).std()

        print(f'Mean values for:\nepoch_nmse: {epoch_nmse:.3f}\nepoch_ssim: {epoch_ssim:.3f}\nepoch_psnr: {epoch_psnr:.3f}')
        print(f'Std values for:\nepoch_nmse: {epoch_nmse_std:.3f}\nepoch_ssim: {epoch_ssim_std:.3f}\nepoch_psnr: {epoch_psnr_std:.3f}')

        self.test_step_outputs.clear()

        return {'epoch_nmse': epoch_nmse.item(), 'epoch_ssim' : epoch_ssim.item(), 'epoch_psnr' : epoch_psnr.item()}

    def on_predict_epoch_end(self):
        batch_losses_nmse = [x["nmse"]for x in self.predict_step_outputs]         # normalized mean squared error
        batch_losses_ssim = [x["ssim"]for x in self.predict_step_outputs]         # structural similarity index  
        batch_losses_psnr = [x["psnr"]for x in self.predict_step_outputs]         # peak signal to noise ratio

        epoch_nmse = torch.FloatTensor(batch_losses_nmse).mean()
        epoch_ssim = torch.FloatTensor(batch_losses_ssim).mean()
        epoch_psnr = torch.FloatTensor(batch_losses_psnr).mean()

        df = pd.DataFrame(self.predict_step_outputs)
        df.to_csv(self.pd_path)

        self.predict_step_outputs.clear()
        return {'epoch_nmse': epoch_nmse.item(), 'epoch_ssim' : epoch_ssim.item(), 'epoch_psnr' : epoch_psnr.item()}


    # For gradient debugging gradients
    # def on_before_optimizer_step(self, optimizer):
    #     total_norm = 0.0
    #     param_norms = []
    #     named_params = {}
    #     named_grad_pramas = {}
        
    #     for n, p in self.model.named_parameters():
    #         named_params[n] = p
    #         if p.grad is not None:
    #             param_norm = p.grad.detach().data.norm(2)
    #             named_grad_pramas[n] = param_norm.item()
    #             param_norms.append(param_norm.item())
    #             total_norm += param_norm.item() ** 2
    #     total_norm = total_norm ** (1. / 2)
    #     print(f'------------------------------------------')
    #     print(f'total_norm {total_norm}')
    #     # print(f'Params: {named_params}')
    #     print(f'Grads: {named_grad_pramas}')
    #     print(f'------------------------------------------')