import torch
import numpy as np
import os
import pickle as pkl

from PIL import Image
from medutils.optimization import CgSenseReconstruction
from medutils.measures import psnr, ssim, nmse
from fft import NUFFT_v3, NUFFTAdj_v3
from dataloader import KspaceDataModule
from tqdm import tqdm
from torchvision.transforms import CenterCrop

from utils import normalize, save_gif

# Define NUFFT operators and configs
nufft_config = {
    'img_dim': 256,     # image dimension
    'osf': 1,         # oversampling factor
    'kernel_width': 3,
    'sector_width': 8,
}

op = NUFFT_v3(nufft_config)
op_adj = NUFFTAdj_v3(configs=nufft_config)

# Only this is needed for training dataloader actually
ROOT =  f'<your_path_here>'
SAVE_RECONSTRUCTION_PATH = f'<your_path_here>'

# Leave rest of params here, just for information on the dataset used 
IMG_PATH = f'<your_path_here>'
LIMIT_NUM_SAMPLES = None
NUM_WORKERS = 8
SHUFFLE = False
BS = 1
TRANSFORM = {
    'train': torch.nn.Identity(),
    'val': torch.nn.Identity(),
    'test': torch.nn.Identity()
}

datamodule = KspaceDataModule(BS, TRANSFORM, ROOT, IMG_PATH, LIMIT_NUM_SAMPLES, NUM_WORKERS, 
                    SHUFFLE, op, op_adj, [nufft_config['img_dim'], nufft_config['img_dim']])

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

class OP(object):
    def __init__(self, op, op_adj, ktraj, smaps, dcomp) -> None:
        self.op = op
        self.op_adj = op_adj
        self.ktraj = ktraj
        self.smaps = smaps
        self.dcomp = dcomp 

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).cuda()
        return self.op(x, self.smaps, self.ktraj, self.dcomp).detach().cpu().numpy()

    def adjoint(self, y):
        if not torch.is_tensor(y):
            y = torch.from_numpy(y).cuda()
        return self.op_adj(y, self.smaps, self.ktraj, self.dcomp).detach().cpu().numpy()


if __name__ == "__main__":

    crop = CenterCrop(160)
    nmse_all = []
    ssim_all = []
    psnr_all = []

    nmse_test = []
    ssim_test = []
    psnr_test = []

    for batch in tqdm(train_loader, desc='Train dataloader'):

        caseid, casename, target_img, kdata, ktraj, smap, dcomp, mask, mask = batch
        # must remove batch dimension from data consistency params
        kdata, ktraj, smap, dcomp =  kdata.squeeze(0).cuda(), ktraj.squeeze(0).cuda(), smap.squeeze(0).cuda(), dcomp.squeeze(0).cuda()
        cur_op = OP(op, op_adj, ktraj, smap, dcomp)
        cgsense = CgSenseReconstruction(op=cur_op, max_iter=6)

        res = torch.from_numpy(cgsense.solve(kdata))
        res = res.squeeze()
        res = crop(res)
        target_img = target_img.squeeze()
        target_img = crop(target_img)
        pred, tar = res.detach().cpu().numpy(), target_img.detach().cpu().numpy()

        comp_list = [pred]
        ref = tar
        new_list = normalize(comp_list=comp_list, ref=ref)
        norm_pred = new_list[0]

        nmse_score = nmse(norm_pred, tar, axes = (1,2))
        ssim_score = ssim(norm_pred, tar, axes = (1,2))
        psnr_score = psnr(norm_pred, tar, axes = (1,2))

        nmse_all.append(nmse_score)
        ssim_all.append(ssim_score)
        psnr_all.append(psnr_score)

        pickle = {'caseid': caseid[0],
                  'casename': casename[0],
                  'pred': pred,
                  'target': tar}

        save_gif(res, SAVE_RECONSTRUCTION_PATH + f'{casename[0]}.gif')
        with open(SAVE_RECONSTRUCTION_PATH + f'{casename[0]}.pkl', 'wb') as f:
            pkl.dump(pickle, f)

    for batch in tqdm(val_loader, desc='Validation dataloader'):

        caseid, casename, target_img, kdata, ktraj, smap, dcomp, mask, mask = batch
        # must remove batch dimension from data consistency params
        kdata, ktraj, smap, dcomp =  kdata.squeeze(0).cuda(), ktraj.squeeze(0).cuda(), smap.squeeze(0).cuda(), dcomp.squeeze(0).cuda()
        cur_op = OP(op, op_adj, ktraj, smap, dcomp)
        cgsense = CgSenseReconstruction(op=cur_op, max_iter=6)

        res = torch.from_numpy(cgsense.solve(kdata))
        res = res.squeeze()
        res = crop(res)
        target_img = target_img.squeeze()
        target_img = crop(target_img)
        pred, tar = res.detach().cpu().numpy(), target_img.detach().cpu().numpy()
        comp_list = [pred]
        ref = tar
        new_list = normalize(comp_list=comp_list, ref=ref)
        norm_pred = new_list[0]

        nmse_score = nmse(norm_pred, tar, axes = (1,2))
        ssim_score = ssim(norm_pred, tar, axes = (1,2))
        psnr_score = psnr(norm_pred, tar, axes = (1,2))

        nmse_all.append(nmse_score)
        ssim_all.append(ssim_score)
        psnr_all.append(psnr_score)

        pickle = {'caseid': caseid[0],
                  'casename': casename[0],
                  'pred': pred,
                  'target': tar}

        save_gif(res, SAVE_RECONSTRUCTION_PATH + f'{casename[0]}.gif')
        with open(SAVE_RECONSTRUCTION_PATH + f'{casename[0]}.pkl', 'wb') as f:
            pkl.dump(pickle, f)

    for batch in tqdm(test_loader, desc='Test dataloader'):

        caseid, casename, target_img, kdata, ktraj, smap, dcomp, mask, mask = batch
        # must remove batch dimension from data consistency params
        kdata, ktraj, smap, dcomp =  kdata.squeeze(0).cuda(), ktraj.squeeze(0).cuda(), smap.squeeze(0).cuda(), dcomp.squeeze(0).cuda()
        cur_op = OP(op, op_adj, ktraj, smap, dcomp)
        cgsense = CgSenseReconstruction(op=cur_op, max_iter=6)

        res = torch.from_numpy(cgsense.solve(kdata))
        res = res.squeeze()
        res = crop(res)
        target_img = target_img.squeeze()
        target_img = crop(target_img)
        pred, tar = res.detach().cpu().numpy(), target_img.detach().cpu().numpy()

        comp_list = [pred]
        ref = tar
        new_list = normalize(comp_list=comp_list, ref=ref)
        norm_pred = new_list[0]

        nmse_score = nmse(norm_pred, tar, axes = (1,2))
        ssim_score = ssim(norm_pred, tar, axes = (1,2))
        psnr_score = psnr(norm_pred, tar, axes = (1,2))

        nmse_all.append(nmse_score)
        ssim_all.append(ssim_score)
        psnr_all.append(psnr_score)
        nmse_test.append(nmse_score)
        ssim_test.append(ssim_score)
        psnr_test.append(psnr_score)

        pickle = {'caseid': caseid[0],
                  'casename': casename[0],
                  'pred': pred,
                  'target': tar}

        save_gif(res, SAVE_RECONSTRUCTION_PATH + f'{casename[0]}.gif')
        with open(SAVE_RECONSTRUCTION_PATH + f'{casename[0]}.pkl', 'wb') as f:
            pkl.dump(pickle, f)

    nmse_all, ssim_all, psnr_all = np.array(nmse_all), np.array(ssim_all), np.array(psnr_all)
    nmse_test, ssim_test, psnr_test = np.array(nmse_test), np.array(ssim_test), np.array(psnr_test)

    result_all = f'NMSE all: {nmse_all.mean()}, SSIM all: {ssim_all.mean()}, PSNR all: {psnr_all.mean()}\nNMSE test: {nmse_test.mean()}, SSIM test: {ssim_test.mean()}, PSNR test: {psnr_test.mean()}'
    print(f'NMSE all: {nmse_all.mean():.3f}, SSIM all: {ssim_all.mean():.3f}, PSNR all: {psnr_all.mean():.3f}\n\n')
    print(f'NMSE test: {nmse_test.mean():.3f}+/-{nmse_test.std():.3f}, SSIM test: {ssim_test.mean():.3f}+/-{ssim_test.std():.3f}, PSNR test: {psnr_test.mean():.3f}+/-{psnr_test.std():.3f}')

    with open(SAVE_RECONSTRUCTION_PATH + 'results.txt', 'w') as f:
        f.write(result_all)