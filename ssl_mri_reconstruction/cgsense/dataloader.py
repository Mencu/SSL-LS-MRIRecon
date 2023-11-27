import glob
import numpy as np
import pickle
import torch
import pytorch_lightning as pl
import medutils

from tqdm import tqdm
from sklearn import model_selection
from torch.utils.data import Dataset, DataLoader
from typing import Callable, List


class KspaceLoader(Dataset):
    def __init__(self, split = None, fnames = None, img_targets: dict = None, transform: Callable = None,
                op=None, op_adj=None, img_dim=None):

        self.split = split
        self.fnames = fnames
        self.img_targets = img_targets
        self.transform = transform
        self.op = op
        self.op_adj = op_adj
        self.img_dim = img_dim

        # load all samples for the defined split into memory
        self.samples = []
        for file_path in tqdm(fnames, desc=f'{self.split} dataset'):
            with open(file_path, 'rb') as f:
                self.samples.append(pickle.load(f))

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
        data = self.samples[item]

        # Match caseid to target images
        caseid = data['caseid'] # int


        for (id, name) in  self.img_targets.keys():
            casename = name
            if id == caseid:
                target_img = self.img_targets[(id, name)].astype(np.complex64)
                break


        full_kdata = data['full_kdata'].astype(np.complex64)
        kdata = data['kdata'].astype(np.complex64)
        ktraj = data['ktraj'].astype(np.float32)
        smap = data['smap'].astype(np.complex64)
        dcomp = data['dcomp'].astype(np.float32)
        tshots = np.array(data['tshots'])
        num_spokes = data['num_spokes'] # int
        mask = data['kmask']
        remainder_mask = data['remainder_mask']
        val_mask = data['val_mask']

        # convert to tensors for training
        target_img = torch.from_numpy(target_img)
        full_kdata = torch.from_numpy(full_kdata)
        kdata = torch.from_numpy(kdata)
        ktraj = torch.from_numpy(ktraj)
        smap = torch.from_numpy(smap)
        dcomp = torch.from_numpy(dcomp)
        mask = torch.from_numpy(mask)

        return caseid, casename, target_img, kdata, ktraj, smap, dcomp, mask, mask

    def __len__(self):
        return len(self.fnames)


class KspaceDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 1, transforms=None, root=None, img_path=None, limit_num_samples=None, 
            num_workers=32, shuffle=True, op=None, op_adj=None, img_dim=None):

        self.fpath = root
        if self.fpath is None:
            self.fpath = f'<your_path_here>'

        self.fnames = glob.glob(self.fpath + '*.pkl')

        if self.fnames == [] or self.fnames is None:
            raise BaseException(f'There has been no found at this location: {self.fpath}')

        print(f'Dataset contains {len(self.fnames)} samples')

        self.targets = {}
        targets = glob.glob(img_path + '*.pkl')
        for file_path in tqdm(targets, desc=f'Targets'):
            with open(file_path, 'rb') as f:
                sample_dict = pickle.load(f)
                self.targets[(sample_dict['caseid'], sample_dict['casename'])] = sample_dict['target']

        self.batch_size = batch_size
        self.transforms = transforms
        self.root = root
        self.img_path = img_path
        self.limit_num_samples = limit_num_samples
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.op = op
        self.op_adj = op_adj
        self.img_dim = img_dim
        
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

        self.train_dataset = KspaceLoader('train', self.train_split, self.targets, transform=self.transforms['train'], 
                                        op=self.op, op_adj=self.op_adj, img_dim=self.img_dim)

        self.validation_dataset = KspaceLoader('val', self.val_split, self.targets, transform=self.transforms['val'], 
                                        op=self.op, op_adj=self.op_adj, img_dim=self.img_dim)

        self.test_dataset = KspaceLoader('test', self.test_split, self.targets, transform=self.transforms['test'], 
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
