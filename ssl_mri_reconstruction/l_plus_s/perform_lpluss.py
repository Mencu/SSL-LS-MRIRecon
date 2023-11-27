import torch
import numpy as np
import os
import pickle as pkl
import pandas as pd

from PIL import Image
from medutils.measures import psnr, ssim, nmse
from fft import NUFFT_v3, NUFFTAdj_v3
from dataloader import KspaceDataModule
from tqdm import tqdm
from torchvision.transforms import CenterCrop

# Define NUFFT operators and configs
nufft_config = {
    'img_dim': 256,     # image dimension
    'osf': 1,         # oversampling factor
    'kernel_width': 3,
    'sector_width': 8,
}

op = NUFFT_v3(nufft_config)
op_adj = NUFFTAdj_v3(configs=nufft_config)

# Only this needed for training dataloader actually
ROOT =  f'<your_path_here>'
SAVE_RECONSTRUCTION_PATH = f'<your_path_here>'

PD_PATH = SAVE_RECONSTRUCTION_PATH + f'results.csv'

# Leave rest of params here, just for information on the dataset used 
IMG_PATH = f'<your_path_here>'
LIMIT_NUM_SAMPLES = None
NUM_WORKERS = 8
SHUFFLE = False
COILS = [2, 3, 4, 5, 6, 7] # as in Wenqi's config
NUM_CARDIAC_CYCLES = 4
TRANSIENT_MAGNETIZATION = 600
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


class LplusS(object):
    def __init__(self, op, op_adj, op_temp, max_iter, lambda_L, lambda_S, gamma=1, tol=0.0025):
        self.op_temp = op_temp
        self.op = op
        self.op_adj = op_adj
        self.max_iter = max_iter
        self.lambda_L = lambda_L
        self.lambda_S = lambda_S
        self.gamma = gamma
        self.tol = tol

    def solve(self, y, csm, traj, dcf):
        # M = self.op.adjoint(y)
        # y.shape = [nslc, nt, nc, nspokes*nFE]
        # csm.shape = [nslc, nc, nx, ny]
        # traj.shape = [nslc, nt, 2, nspokes*nFE]
        # dcf.shape = [nslc, nt, 1, nspokes*nFE]
        M = self.op_adj(y, csm, traj, dcf).squeeze(1)
        nt, nx, ny = M.shape
        M = torch.reshape(M, (nt, nx*ny))
        # M0 = np.zeros_like(M)
        M0 = torch.zeros_like(M)
        # assert M.ndim == 3
        # nt, ny, nx = M.shape
        Lpre = M.clone()
        S = torch.zeros_like(Lpre)

        it = 0

        while it < self.max_iter and torch.linalg.norm((M-M0).flatten()) > self.tol * torch.linalg.norm(M0.flatten()):
            # Low-rank update
            M0 = M.clone()
            # tmp = torch.reshape(M - S, (nt, nx * ny))
            u, s, vh = torch.linalg.svd((M - S), full_matrices=False)

            # Soft-Thresholding of singular values
            s = torch.relu(s - s[0] * self.lambda_L)

            # Update low-rank component
            s = torch.diag_embed(s).type(u.dtype)
            L = torch.mm(torch.mm(u, s), vh)

            # Sparse update
            S = self.op(torch.reshape(M - Lpre, (nt,1,nx, ny)),csm, traj, dcf)
            S = torch.relu(torch.abs(S) - self.lambda_S) * (S / (1e-10 + torch.abs(S)))
            S = self.op_adj(S,csm, traj, dcf)
            S = torch.reshape(S, (nt, nx*ny))
            # S = torch.max(torch.abs(S), torch.scalar_tensor(self.lambda_S, device=S.device).expand_as(S)) * S / (torch.abs(S) + 1e-10)
        
            # Data consistency
            resk = self.op_adj(self.op(torch.reshape(L + S, (nt, 1, nx, ny)),  csm, traj, dcf) -  y,  csm, traj, dcf).squeeze(1)
            M = L + S - self.gamma * torch.reshape(resk, (nt, nx * ny))

            Lpre = L.clone()
            it += 1

        L = torch.reshape(L, (nt, nx, ny))
        S = torch.reshape(S, (nt, nx, ny))
        M = torch.reshape(M, (nt, nx, ny))
        return L, S, M

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

def normalize(comp_list, ref):
    for i in range(len(comp_list)):
        i_flat = comp_list[i].flatten() #[:,35:120,35:120]
        i_flat = np.concatenate((i_flat.real, i_flat.imag))
        a = np.stack([i_flat, np.ones_like(i_flat)], axis=1)
        b = ref.flatten()
        b = np.concatenate((b.real, b.imag))
        x = np.linalg.lstsq(a, b, rcond=None)[0]
        comp_list[i] = comp_list[i] * x[0] + x[1]

        return comp_list

if __name__ == "__main__":

    max_iter = 50
    lambda_L=0.003
    lambda_S=0.1

    crop = CenterCrop(160)
    nmse_all = []
    ssim_all = []
    psnr_all = []

    nmse_test = []
    ssim_test = []
    psnr_test = []

    csv_res = []

    for batch in tqdm(train_loader, desc='Train dataloader'):
        caseid, casename, target_img, kdata, ktraj, smap, dcomp, mask = batch
        # must remove batch dimension from data consistency params
        kdata, ktraj, smap, dcomp =  kdata.squeeze(0).cuda(), ktraj.squeeze(0).cuda(), smap.squeeze(0).cuda(), dcomp.squeeze(0).cuda()
        # kdata, ktraj, smap, dcomp =  kdata.cuda(), ktraj.cuda(), smap.cuda(), dcomp.cuda()
        # print(f'kdata: {kdata.shape}, smap: {smap.shape}, ktraj {ktraj.shape}, dcomp {dcomp.shape}')
        
        op_temp = OP(op, op_adj, ktraj, smap, dcomp)
        l_plus_s = LplusS(op, op_adj, op_temp, max_iter, lambda_L, lambda_S, gamma=1, tol=0.0025)

        L, S, M = l_plus_s.solve(kdata, smap, ktraj, dcomp)
        res = L+S
        res = res.squeeze()
        res_compare = crop(res)
        res = res.detach().cpu().numpy()

        target_img = target_img.squeeze()
        target_img = crop(target_img)
        pred, tar = res_compare.detach().cpu().numpy(), target_img.detach().cpu().numpy()


        nmse_score = nmse(pred, tar, axes = (1,2))
        ssim_score = ssim(pred, tar, axes = (1,2))
        psnr_score = psnr(pred, tar, axes = (1,2))

        nmse_all.append(nmse_score)
        ssim_all.append(ssim_score)
        psnr_all.append(psnr_score)

        pickle = {'caseid': caseid[0],
                  'casename': casename[0],
                  'pred': pred,
                  'target': tar}

        csv_res.append({'caseid': caseid[0],'casename': casename[0], 'nmse': nmse_score, 'ssim': ssim_score, 'psnr': psnr_score})

        with open(SAVE_RECONSTRUCTION_PATH + f'{casename[0]}.pkl', 'wb') as f:
            pkl.dump(pickle, f)

    for batch in tqdm(val_loader, desc='Validation dataloader'):

        caseid, casename, target_img, kdata, ktraj, smap, dcomp, mask = batch
        # must remove batch dimension from data consistency params
        kdata, ktraj, smap, dcomp =  kdata.squeeze(0).cuda(), ktraj.squeeze(0).cuda(), smap.squeeze(0).cuda(), dcomp.squeeze(0).cuda()
        # kdata, ktraj, smap, dcomp =  kdata.cuda(), ktraj.cuda(), smap.cuda(), dcomp.cuda()
        op_temp = OP(op, op_adj, ktraj, smap, dcomp)
        l_plus_s = LplusS(op, op_adj, op_temp, max_iter, lambda_L, lambda_S, gamma=1, tol=0.0025)

        L, S, M = l_plus_s.solve(kdata, smap, ktraj, dcomp)
        res = L+S
        res = res.squeeze()
        res_compare = crop(res)
        res = res.detach().cpu().numpy()
        target_img = target_img.squeeze()
        target_img = crop(target_img)
        pred, tar = res_compare.detach().cpu().numpy(), target_img.detach().cpu().numpy()

        nmse_score = nmse(pred, tar, axes = (1,2))
        ssim_score = ssim(pred, tar, axes = (1,2))
        psnr_score = psnr(pred, tar, axes = (1,2))

        nmse_all.append(nmse_score)
        ssim_all.append(ssim_score)
        psnr_all.append(psnr_score)

        pickle = {'caseid': caseid[0],
                  'casename': casename[0],
                  'pred': pred,
                  'target': tar}

        csv_res.append({'caseid': caseid[0],'casename': casename[0], 'nmse': nmse_score, 'ssim': ssim_score, 'psnr': psnr_score})

        with open(SAVE_RECONSTRUCTION_PATH + f'{casename[0]}.pkl', 'wb') as f:
            pkl.dump(pickle, f)

    for batch in tqdm(test_loader, desc='Test dataloader'):

        caseid, casename, target_img, kdata, ktraj, smap, dcomp, mask = batch
        # must remove batch dimension from data consistency params
        kdata, ktraj, smap, dcomp =  kdata.squeeze(0).cuda(), ktraj.squeeze(0).cuda(), smap.squeeze(0).cuda(), dcomp.squeeze(0).cuda()
        # kdata, ktraj, smap, dcomp =  kdata.cuda(), ktraj.cuda(), smap.cuda(), dcomp.cuda()
        op_temp = OP(op, op_adj, ktraj, smap, dcomp)
        l_plus_s = LplusS(op, op_adj, op_temp, max_iter, lambda_L, lambda_S, gamma=1, tol=0.0025)

        L, S, M = l_plus_s.solve(kdata, smap, ktraj, dcomp)
        res = L+S
        res = res.squeeze()
        res_compare = crop(res)
        res = res.detach().cpu().numpy()
        target_img = target_img.squeeze()
        target_img = crop(target_img)
        pred, tar = res_compare.detach().cpu().numpy(), target_img.detach().cpu().numpy()

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

        csv_res.append({'caseid': caseid[0],'casename': casename[0], 'nmse': nmse_score, 'ssim': ssim_score, 'psnr': psnr_score})

        with open(SAVE_RECONSTRUCTION_PATH + f'{casename[0]}.pkl', 'wb') as f:
            pkl.dump(pickle, f)

    nmse_all, ssim_all, psnr_all = np.array(nmse_all), np.array(ssim_all), np.array(psnr_all)
    nmse_test, ssim_test, psnr_test = np.array(nmse_test), np.array(ssim_test), np.array(psnr_test)

    result_all = f'NMSE all: {nmse_all.mean()}, SSIM all: {ssim_all.mean()}, PSNR all: {psnr_all.mean()}\nNMSE test: {nmse_test.mean()}, SSIM test: {ssim_test.mean()}, PSNR test: {psnr_test.mean()}'
    print(f'NMSE all: {nmse_all.mean()}, SSIM all: {ssim_all.mean()}, PSNR all: {psnr_all.mean()}')
    print(f'NMSE test: {nmse_test.mean()}, SSIM test: {ssim_test.mean()}, PSNR test: {psnr_test.mean()}')

    with open(SAVE_RECONSTRUCTION_PATH + 'results.txt', 'w') as f:
        f.write(result_all)

    df = pd.DataFrame(csv_res)
    df.to_csv(PD_PATH)