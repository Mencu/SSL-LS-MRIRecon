import glob
import numpy as np
import pickle
import torch
import pytorch_lightning as pl

from tqdm import tqdm
from sklearn import model_selection
from torch.utils.data import Dataset, DataLoader
from typing import Callable, List
from merlinth.complex import complex2real, real2complex
from utils import uniform_selection
from medutils.visualization import ksave, imsave



class KspaceLoader(Dataset):
    def __init__(self, split = None, fnames = None, img_path: str = None, transform: Callable = torch.nn.Identity(), batch_size = 64, 
                coil_select:List = [0], num_cardiac_cycles: int = 1, transient_magnetization = 600, target_phases=30, op=None, op_adj=None, img_dim=None):

        self.split = split
        self.fnames = fnames
        self.img_path = img_path
        self.transform = transform
        self._batch_size = batch_size
        self.coil_select = coil_select
        self.num_cardiac_cycles = num_cardiac_cycles
        self.transient_magnetization = transient_magnetization
        self.op = op
        self.op_adj = op_adj
        self.img_dim = img_dim
        self.target_phases = target_phases
        self.weight_config = {
            'type': 'rectangle',
            'overlap': 0,
            'weight_norm': True,
        }

        torch.cuda.set_device(0) 
        self.device = torch.device('cuda')


        # TODO do this in-memory somehow, eliminate unused variables
        # # load all samples for the defined split into memory
        # self.samples = []
        # for file_path in tqdm(fnames, desc=f'{self.split} dataset'):
        #     with open(file_path, 'rb') as f:
        #         self.samples.append(pickle.load(f))


    def __getitem__(self, item):
        """
        All loaded pickles are dicts with the following data:
            {'caseid' : caseid, 
            'target_img' : target_img, 
            'img' : img, 
            'full_kdata' : full_kdata, 
            'kdata' : kdata, 
            'ktraj' : ktraj, 
            'smap' : smap, 
            'dcomp' : dcomp, 
            'tshots' : tshots, 
            'num_spokes' : new_nspokes,
            'mask': mask}
        
        """
        
        file_path = self.fnames[item]
        data = None
        with open(file_path, 'rb') as f:
                data = pickle.load(f)
        # data = self.samples[item]

        caseid = data['caseid'] # int
        target_img = data['target_img'].astype(np.complex64)
        img = data['img'].astype(np.complex64)
        full_kdata = data['full_kdata'].astype(np.complex64)
        kdata = data['kdata'].astype(np.complex64)
        ktraj = data['ktraj'].astype(np.float32)
        smap = data['smap'].astype(np.complex64)
        dcomp = data['dcomp'].astype(np.complex64)
        tshots = np.array(data['tshots'])
        num_spokes = data['num_spokes'] # int
        mask = data['kmask']

        # convert to tensors for training
        target_img = torch.from_numpy(target_img)
        img = torch.view_as_real(torch.from_numpy(img).squeeze(1)).permute(3,1,2,0)
        full_kdata = torch.from_numpy(full_kdata)
        kdata = torch.from_numpy(kdata)
        ktraj = torch.from_numpy(ktraj)
        smap = torch.from_numpy(smap)
        dcomp = torch.from_numpy(dcomp)
        mask = torch.from_numpy(mask)

        return caseid, target_img, img, kdata, ktraj, smap, dcomp, mask

    def __len__(self):
        return len(self.fnames)


class KspaceDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 1, transforms=None, root=None, img_path=None, limit_num_samples=None, 
            num_workers=32, shuffle=True, coil_select=[0], num_cardiac_cycles=1, transient_magnetization=600, op=None, op_adj=None, img_dim=None):

        self.fpath = root
        if self.fpath is None:
            self.fpath = f'<your_path_here>'

        self.fnames = glob.glob(self.fpath + '*.pkl')

        if self.fnames == [] or self.fnames is None:
            raise BaseException(f'There has been no found at this location: {self.fpath}')

        print(f'Dataset contains {len(self.fnames)} samples')

        self.batch_size = batch_size
        self.transforms = transforms
        self.root = root
        self.img_path = img_path
        self.limit_num_samples = limit_num_samples
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.coil_select = coil_select
        self.num_cardiac_cycles = num_cardiac_cycles
        self.transient_magnetization = transient_magnetization
        self.op = op
        self.op_adj = op_adj
        self.img_dim = img_dim

        # self.train_split, self.val_split, self.test_split = self.fnames[:1], self.fnames[:1], self.fnames[:1]

        # only use 90% for train-val, 10% is always test
        # from which train is 80% and val is 20%
        train_test_split = int(0.9 * len(self.fnames))
        self.train_split, self.val_split = model_selection.train_test_split(self.fnames[:train_test_split], test_size=0.2, shuffle=self.shuffle)
        self.test_split = self.fnames[train_test_split:]

        # Limit all predefined paths to the number of limited samples
        if self.limit_num_samples is not None:
            self.train_split = self.train_split[:self.limit_num_samples]
            self.val_split = self.val_split[:self.limit_num_samples]
            self.test_split = self.test_split[:self.limit_num_samples]

        print(f'Processed split:\nTrain\t{len(self.train_split)}\nVal\t{len(self.val_split)}\nTest\t{len(self.test_split)}')

        self.train_dataset = KspaceLoader('train', self.train_split, self.img_path, batch_size=self.batch_size, transform=self.transforms['train'], 
                                        coil_select=self.coil_select, num_cardiac_cycles=self.num_cardiac_cycles, transient_magnetization=self.transient_magnetization,
                                        op=self.op, op_adj=self.op_adj, img_dim=self.img_dim)

        self.validation_dataset = KspaceLoader('val', self.val_split, self.img_path, batch_size=self.batch_size, transform=self.transforms['val'], 
                                        coil_select=self.coil_select, num_cardiac_cycles=self.num_cardiac_cycles, transient_magnetization=self.transient_magnetization,
                                        op=self.op, op_adj=self.op_adj, img_dim=self.img_dim)

        self.test_dataset = KspaceLoader('test', self.test_split, self.img_path, batch_size=self.batch_size, transform=self.transforms['test'], 
                                        coil_select=self.coil_select, num_cardiac_cycles=self.num_cardiac_cycles, transient_magnetization=self.transient_magnetization,
                                        op=self.op, op_adj=self.op_adj, img_dim=self.img_dim)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)
