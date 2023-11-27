"""
This is an old dataloader that was used to bin data and preprocess it.
The new dataloader loads directly preprocessed pickled data
"""


import glob
import os
import numpy as np
import h5py
import medutils
import torch
import pytorch_lightning as pl

from sklearn import model_selection
from torch.utils.data import Dataset, DataLoader
from typing import Callable, List
from fft import compute_dcf


class KspaceLoader(Dataset):
    def __init__(self, split = None, fnames = None, img_path: str = None, transform: Callable = torch.nn.Identity(), batch_size = 64, 
                coil_select:List = [0], num_cardiac_cycles: int = 1, transient_magnetization = 600, target_phases=30, op=None, op_adj=None, img_dim=None):

        if img_path is None:
            img_path = f'<your_path_here>'

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
        # target phases?
        self.target_phases = target_phases
        self.weight_config = {
            'type': 'rectangle',
            'overlap': 0,
            'weight_norm': True,
        }
        torch.cuda.set_device(0) 
        self.device = torch.device('cuda')


    def __getitem__(self, item):
        print(item)
        data = self.get_patient_data(item)

        target_img = data['full_img'].astype(np.complex64)
        full_kdata = data['full_kdata'].astype(np.complex64)
        kdata = data['kdata'].astype(np.complex64)
        ktraj = data['ktraj'].astype(np.float32)
        smap = data['smap'].astype(np.complex64)
        tshots = np.array(data['tshots'])
        RR = data['RR']

        # print(kdata.shape)
        _, nc, nspokes, nFE = kdata.shape

        # 1: BINNING
        self.target_phases = 30
        spoke_id_list = []

        t_frame = RR / self.target_phases

        for i in range(self.target_phases):
            tstart = t_frame * i
            tend = t_frame * (i+1)
            idxshots = np.where((tshots >= tstart) & (tshots < tend))[0]
            spoke_id_list.append(idxshots)
        
        spoke_num = [len(x) for x in spoke_id_list]
        max_spoke_num = max(spoke_num)

        kdata_binned = np.zeros((self.target_phases, nc, max_spoke_num, nFE), dtype=np.complex64)
        ktraj_binned = np.zeros((self.target_phases, 2, max_spoke_num, nFE), dtype=np.float32)

        for i in range(self.target_phases):
            kdata_binned[i, :, :spoke_num[i], :] = kdata[:, :, spoke_id_list[i], :]
            ktraj_binned[i, :, :spoke_num[i], :] = ktraj[:, spoke_id_list[i], :]

        _, num_coils, num_spokes, num_freq_enq = kdata.shape


        kdata = kdata_binned
        ktraj = ktraj_binned
        nt, _, new_nspokes, _ = kdata.shape

        # print(f'binned kdata shape: {kdata.shape}\n ktraj: {ktraj.shape}')

        # ttarget = [i / self.target_phases * RR for i in range(self.target_phases)]
        # ttarget = [x * data['RR'] for x in ttarget]
        # print(tshots[0])
        # TODO remove this after binning
        # weight = self.weight_generation(tshots=tshots, ttarget=ttarget, RR=RR, weight_config=self.weight_config).astype(np.float32)
        # kdata = weight * kdata      

        # 2: DCOMP calculation
        kdata = torch.from_numpy(kdata).reshape((self.target_phases, num_coils, -1)).to(self.device)
        # smap = np.ones((1, nufft_config['img_dim'], nufft_config['img_dim'])).astype(np.complex64)
        smap = torch.from_numpy(smap).to(self.device)
        ktraj = torch.from_numpy(ktraj).reshape(self.target_phases,2,-1).to(self.device)

        # calculate decomposition field func
        weight_shapes = (nt, new_nspokes)
        dcomp = compute_dcf(ktraj, smap, weight_shapes, self.img_dim, self.op, self.op_adj, self.device)

        # TODO make faster allocation
        caseid = data['caseid']
        img = self.op_adj.forward(kdata*torch.sqrt(dcomp), smap, ktraj, dcomp).squeeze()
        # kdata = torch.from_numpy(kdata.astype(np.complex64))
        # smap = torch.from_numpy(smap)
        tshots = torch.from_numpy(tshots)
        target_img = torch.from_numpy(target_img)
        full_kdata = torch.from_numpy(full_kdata)

        return caseid, target_img, img, full_kdata, kdata*torch.sqrt(dcomp), ktraj, smap, dcomp, tshots, new_nspokes

    def __len__(self):
        return len(self.fnames)

    def get_patient_data(self, item, img_dim=(160,160)):
        case_name = self.fnames[item].split('/')[-2]
        print(case_name)
        h5_data = h5py.File(self.fnames[item], 'r', libver='latest', swmr=True)
        img_data = h5py.File(self.img_path + case_name + '.h5', 'r', libver='latest', swmr=True)['target']['reconstruction'][:]

        # load coil sensitivity maps (`csm`)
        csm = h5_data['csm_real'][self.coil_select] + 1j*h5_data['csm_imag'][self.coil_select]

        # `full_kdata` is the continously acquired k-space data
        full_kdata = h5_data['fullkdata_real'][self.coil_select] + 1j * h5_data['fullkdata_imag'][self.coil_select]
        # `full_kpos` contains the position of the spokes in the k-space, i.e., trajectory
        full_kpos = h5_data['fullkpos'][()]

        # crop csm to img_dim
        csm = medutils.visualization.center_crop(csm, img_dim)
        csm_rss = medutils.mri.rss(csm, coil_axis=0)
        csm = np.nan_to_num(csm/csm_rss)

        _, n_total_spokes, nFE = full_kdata.shape

        # patient-related binning info
        ecg = h5_data['ECG'][0]
        acq_t0 = h5_data['read_marker'][0][0] # original acq_t0

        # TR based on read_marker
        TR = (h5_data['read_marker'][0][-1] - h5_data['read_marker'][0][0] 
                + h5_data['read_marker'][0][1] - h5_data['read_marker'][0][0]) / n_total_spokes
        h5_data.close()
        # binning
        # binned_data = self.binning_to_target_cycles(ecg, acq_t0, TR, n_total_spokes, all_cycles=False)
        binned_data = self.binning_to_target_cycles(1, ecg, acq_t0, TR, n_total_spokes, all_cycles=True)

        # undersampled data
        tshots = binned_data['tshots']
        idxshots = binned_data['idxshots']
        target_RR = binned_data['target_RR']
        # ttarget = [x * target_RR for x in self.ttarget] # also for fully sampled data
        # num_cycles = binned_data['num_cycles']

        tstart_eachcycle = binned_data['tstart_eachcycle']
        tend_eachcycle = binned_data['tend_eachcycle']
        RR_eachcycle = binned_data['RR_eachcycle']
        tshots_eachcycle = binned_data['tshots_eachcycle']
        idxshots_eachcycle = binned_data['idxshots_eachcycle']

        tstart_eachcycle = binned_data['tstart_eachcycle']
        tend_eachcycle = binned_data['tend_eachcycle']
        RR_eachcycle = binned_data['RR_eachcycle']
        tshots_eachcycle = binned_data['tshots_eachcycle']
        idxshots_eachcycle = binned_data['idxshots_eachcycle']

        if self.split == "train" or self.split == "val":

            kdata = full_kdata[:, idxshots, :][None] # 1, nc, nSpokes, nFE
            ktraj = full_kpos[:, idxshots, :] * (2 * np.pi) 
            ktraj = np.roll(ktraj, shift=1, axis=0) # 2, nSpokes, nFE

            # print(np.max(np.abs(kdata)))
            # normalize k-space data
            # print(np.max(np.abs(kdata)))
            kdata = kdata / np.max(np.abs(kdata)) #* 255

            data = {
                'caseid': item,
                'full_img': img_data,
                'full_kdata': full_kdata,
                'kdata': kdata,# / (np.abs(kdata).max()*0.001), # 1, nc, nSpokes, nFE
                'smap': csm, # nc, img_dim, img_dim
                'ktraj': ktraj, # 2, nSpokes, nFE
                'tshots': tshots, # nSpokes
                'RR': target_RR,  # 1
            }

            return data
        
        else:
            
            kdata = []
            ktraj = []
            for i in range(len(RR_eachcycle)):
                kdata.append(full_kdata[:, tshots_eachcycle[i], :][None])
                ktraj.append(np.roll(full_kpos[:, tshots_eachcycle[i], :] * (2 * np.pi), shift=1, axis=0))

            data = {
                'caseid': item,
                'kdata': kdata ,#/ (1e-3 * np.abs(kdata).max()),     # nRR * [1, nc, nSpokes, nFE]
                'ktraj': ktraj,     # nRR * [2, nSpokes, nFE]
                'tshots': tshots_eachcycle, # nRR * nSpokes
                'RR': RR_eachcycle, # nRR
            }

            return data


    def binning_to_target_cycles(self, target_cycles, ecg, acq_t0, TR, n_spokes, all_cycles=False):
        """Retrospective binning of cardiac data
        Args:
            ecg: array with the timings of the ECG R-wave detections
            acq_t0: start time of acquisitions
            TR: repetition time
            n_spokes: number of shots
        Returns:
            data_binning(dict):
                'target_RR': target RR interval
                'num_cycles': number of cardiac cycles
                'tshots': relative time stamps of the shots
                'idxshots': indices of the shots in the original data (sorted by the relative time in target_RR)
        """

        # discard some shots at the beginning of the data to avoid the 
        # influcence of transient magnetization
        Ndiscard = np.ceil(self.transient_magnetization / TR)

        # calculate the real time stamp for each shots
        shot_time = acq_t0 + np.arange(n_spokes) * TR

        # initialize the list for output shots
        tshots_out = []
        idxshots_out = []

        tshots_eachcycle = [] # for each RR interval
        idxshots_eachcycle = []
        tstart_eachcycle = []
        tend_eachcycle = []
        RR_eachcycle = []

        # binning to an averaved heart beat period
        target_RR = np.mean(ecg[1:] - ecg[:-1])

        num_cycles = 0        # loop for each RR interval (inversed order)
        for idx_ecg_inv in range(0, min(ecg.size, self.num_cardiac_cycles), target_cycles):
            if idx_ecg_inv < ecg.size - target_cycles:
                tperiod_start = ecg[-1 - idx_ecg_inv - target_cycles]
                tperiod_end = ecg[-1 - idx_ecg_inv]
            current_RR = tperiod_end - tperiod_start

            shots_idx_absolute = np.where(np.logical_and(shot_time > tperiod_start, shot_time < tperiod_end))[0]

            # transient magnetization
            shots_idx_absolute = shots_idx_absolute[shots_idx_absolute >= Ndiscard]

            if not all_cycles:
                shots_idx_absolute = []

            tcurrent_shots = shot_time[shots_idx_absolute]
            tshots_relative = (tcurrent_shots - tperiod_start) * (target_RR / current_RR)

            # count cardiac cycles
            if tshots_relative != []:
                if tshots_relative[0] < 0.05 * target_RR:
                    num_cycles += 1
                else:
                    num_cycles += 1 - tshots_relative[0] / target_RR
            
            tstart_eachcycle.insert(0, tperiod_start)
            tend_eachcycle.insert(0, tperiod_end)
            RR_eachcycle.insert(0, current_RR)
            tshots_eachcycle.insert(0, (tcurrent_shots - tperiod_start)/current_RR)
            idxshots_eachcycle.insert(0, shots_idx_absolute)


            tshots_out.extend(list(tshots_relative))
            idxshots_out.extend(list(shots_idx_absolute))
        
        (tshots, idxshots) = list(zip(*sorted(zip(tshots_out, idxshots_out))))

        binned_data = {
            'target_RR': target_RR,
            'num_cycles': num_cycles,
            'tshots': list(tshots),
            'idxshots': list(idxshots),
            'tstart_eachcycle': list(tstart_eachcycle),
            'tend_eachcycle': list(tend_eachcycle),
            'RR_eachcycle': list(RR_eachcycle),
            'tshots_eachcycle': list(tshots_eachcycle),
            'idxshots_eachcycle': list(idxshots_eachcycle)
        }
        return binned_data

    def weight_generation(self, tshots, ttarget, RR, weight_config) -> np.ndarray:
        """
        Args:
            tshots: relative time stamps of the shots
            ttarget: target number of phases to be reconstructed
            sigma: standard deviation of the Gaussian distribution
            RR: target RR interval
            weight_norm: normalize the weights to 1


        Returns:
            weight (nn.Tensor): weight distribution for spokes of kdata -> torch.float32
        """
        # TODO implement: also cut off zero values in kdata -> indices to do kdata[indices]?
        type = weight_config['type']
        weight_norm = weight_config['weight_norm']
        weights = np.zeros([len(ttarget), len(tshots)])
        if type == 'Gaussian':
            sigma = weight_config['sigma']
            sigma = sigma * RR
            # weights = np.zeros([len(tshots)])
            for tidx, mu in enumerate(ttarget):
                tdistance = np.minimum(np.abs(tshots - mu), np.abs(RR - np.abs(tshots - mu)))
                w = np.exp(-tdistance**2/(2*sigma**2)) #/ np.sqrt(2*np.pi*sigma**2)
                # TODO: find a suitable sigma
                # TODO: not strictly same as the definition of the Gaussian
                # normalize the weights
                if weight_norm:
                    dt = np.array(list(tshots[1:]) + [RR]) - np.array(tshots)
                    # w /= (np.sum(w * dt)**2)
                    w /= np.sum(w * dt)
                    w *= np.sum(dt)
                    # w /= np.sum(w)
                weights[tidx, :] = w
            weights = np.expand_dims(weights,(1,3))

        elif type == 'rectangle':
            overlap = weight_config['overlap']
            window_size = ttarget[1] * (1 + overlap)
            for tidx, mu in enumerate(ttarget):
                tdistance = np.minimum(np.abs(tshots - mu), np.abs(RR - np.abs(tshots - mu)))
                w = 1.0 * (tdistance < window_size/2)

                if weight_norm:
                    dt = np.array(list(tshots[1:]) + [RR]) - np.array(tshots)
                    # w /= (np.sum(w * dt)**2)
                    w /= np.sum(w * dt)
                    w *= np.sum(dt)
                    # w /= np.sum(w)
                weights[tidx, :] = w
            weights = np.expand_dims(weights,(1,3))
        

        return weights



class KspaceDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64, transforms=None, root=None, img_path=None, limit_num_samples=None, 
            num_workers=32, shuffle=True, coil_select=[0], num_cardiac_cycles=1, transient_magnetization=600, op=None, op_adj=None, img_dim=None):

        self.fpath = root
        if self.fpath is None:
            self.fpath = f'<your_path_here>'

        self.fnames = glob.glob(self.fpath + '*/*.h5')

        if self.fnames == [] or self.fnames is None:
            raise BaseException(f'There has been no found at this location: {self.hclouds}')

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
