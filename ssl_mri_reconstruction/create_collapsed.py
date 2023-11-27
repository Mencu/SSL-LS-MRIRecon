import pickle
import glob
import os
import numpy as np
import h5py
import medutils
import torch
import tqdm

from PIL import Image
from typing import Callable, List
from zs_ssl.fft import compute_dcf
from zs_ssl.fft import NUFFT_v3, NUFFTAdj_v3
from zs_ssl.utils import uniform_selection

ROOT = f'<your_path_here>'
IMG_PATH = f'<your_path_here>'
TARGET_PATH1 = f'<your_path_here>'
TARGET_PATH2 = f'<your_path_here>'

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

weight_config = {
            'type': 'rectangle',
            'overlap': 0.5,
            'weight_norm': True,
        }


# Define NUFFT operators and configs
nufft_config = {
    'img_dim': 256,     # image dimension
    'osf': 1,         # oversampling factor TODO 1 (dim 256) and 2 (dim 128)
    'kernel_width': 3,
    'sector_width': 8,
}

op = NUFFT_v3(nufft_config)
op_adj = NUFFTAdj_v3(configs=nufft_config)

img_dim = [nufft_config['img_dim'], nufft_config['img_dim']]
COILS = [2, 3, 4, 5, 6, 7]
TRANSIENT_MAGNETIZATION = 600
NUM_CARDIAC_CYCLES = 4


def max_eigenvalue(M, y, max_iter=30, verbose=False):
    b_k = y.clone().to(device)

    if verbose:
        iter_list = tqdm.tqdm(range(max_iter))
    else:
        iter_list = range(max_iter)
    
    for i in iter_list:
        # calculate the matrix-by-vector product Ab
        b_k1 = M(b_k).to(device)

        # calculate the norm
        b_k1_norm = torch.linalg.norm(b_k1)
        b_k_norm = torch.linalg.norm(b_k)

        # spectral radius
        eig = torch.sum(torch.conj(b_k1) * b_k) / b_k_norm
        # print(eig)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm
    return eig

def weight_generation(tshots, ttarget, RR, weight_config) -> np.ndarray:
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
    weights = np.zeros([len(ttarget), tshots.shape[1]])
    if type == 'Gaussian':
        sigma = weight_config['sigma']
        sigma = sigma * RR
        for tidx, mu in enumerate(ttarget):
            tdistance = np.minimum(np.abs(tshots[tidx] - mu), np.abs(RR - np.abs(tshots[tidx] - mu)))
            w = np.exp(-tdistance**2/(2*sigma**2)) 
            # normalize the weights
            if weight_norm:
                dt = np.array(list(tshots[tidx][1:]) + [RR]) - np.array(tshots[tidx])
                w /= np.sum(w * dt)
                w *= np.sum(dt)
            weights[tidx, :] = w
        weights = np.expand_dims(weights,(1,3))

    elif type == 'rectangle':
        overlap = weight_config['overlap']
        window_size = ttarget[1] * (1 + overlap)
        for tidx, mu in enumerate(ttarget):
            tdistance = np.minimum(np.abs(tshots[tidx] - mu), np.abs(RR - np.abs(tshots[tidx] - mu)))
            w = 1.0 * (tdistance < window_size/2)

            if weight_norm:
                dt = np.array(list(tshots[tidx][1:]) + [RR]) - np.array(tshots[tidx])
                w /= np.sum(w * dt)
                w *= np.sum(dt)
            weights[tidx, :] = w

        weights = np.expand_dims(weights,(1,3))
    

    return weights

def binning_to_target_cycles(target_cycles, ecg, acq_t0, TR, n_spokes, all_cycles=False):
    r"""Retrospective binning of cardiac data
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
    Ndiscard = np.ceil(TRANSIENT_MAGNETIZATION / TR)

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
    for idx_ecg_inv in range(0, min(ecg.size, NUM_CARDIAC_CYCLES), target_cycles):
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

def get_patient_data(item, img_dim=(256,256), files=None):
    case_name = files[item].split('/')[-2]
    h5_data = h5py.File(files[item], 'r', libver='latest', swmr=True)
    try:
        img_data = h5py.File(IMG_PATH + case_name + '.h5', 'r', libver='latest', swmr=True)['target']['reconstruction'][:]
    except:
        return None, None

    # load coil sensitivity maps (`csm`)
    csm = h5_data['csm_real'][COILS] + 1j*h5_data['csm_imag'][COILS]

    # `full_kdata` is the continously acquired k-space data
    full_kdata = h5_data['fullkdata_real'][COILS] + 1j * h5_data['fullkdata_imag'][COILS]
    # `full_kpos` contains the position of the spokes in the k-space, i.e., trajectory
    full_kpos = h5_data['fullkpos'][()]

    # crop csm to img_dim
    csm = medutils.visualization.center_crop(csm, img_dim)          # TODO look at csm before and after
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
    binned_data = binning_to_target_cycles(1, ecg, acq_t0, TR, n_total_spokes, all_cycles=True)

    # undersampled data
    tshots = binned_data['tshots']
    idxshots = binned_data['idxshots']
    target_RR = binned_data['target_RR']

    kdata = full_kdata[:, idxshots, :][None] # 1, nc, nSpokes, nFE
    ktraj = full_kpos[:, idxshots, :] * (2 * np.pi)
    ktraj = np.roll(ktraj, shift=1, axis=0) # 2, nSpokes, nFE

    data = {
        'caseid': item,
        'casename': case_name,
        'full_img': img_data,
        'full_kdata': full_kdata,
        'kdata': kdata,# / (np.abs(kdata).max()*0.001), # 1, nc, nSpokes, nFE
        'smap': csm, # nc, img_dim, img_dim
        'ktraj': ktraj, # 2, nSpokes, nFE
        'tshots': tshots, # nSpokes
        'RR': target_RR,  # 1
    }

    return data, case_name

if __name__ == "__main__":
    file_paths = glob.glob(ROOT + '*/*.h5')
    img_paths = glob.glob(IMG_PATH + '*.h5')

    for i in tqdm.tqdm(range(len(file_paths))):
        print(f'Start processing for patient: {i}')
        data, case_name = get_patient_data(i, files=file_paths)
        if data is None:
            continue

        target_img = data['full_img'].astype(np.complex64)
        full_kdata = data['full_kdata'].astype(np.complex64)
        kdata = data['kdata'].astype(np.complex64)
        ktraj = data['ktraj'].astype(np.float32)
        smap = data['smap'].astype(np.complex64)
        tshots = np.array(data['tshots'])
        RR = data['RR']

        _, nc, nspokes, nFE = kdata.shape

        # 1: BINNING
        target_phases = 30

        spoke_id_list = []

        t_frame = RR / target_phases

        print(f'Collapse kspaces')

        for i in range(target_phases):
            tstart = t_frame * i
            tend = t_frame * (i+1)
            idxshots = np.where((tshots >= tstart) & (tshots < tend))[0]
            spoke_id_list.append(idxshots)

        spoke_num = [len(x) for x in spoke_id_list]
        min_spoke_num = min(spoke_num)                  # NOTE min to not complicate with dcomp and masking at expense of some information

        kdata_binned = np.zeros((target_phases, nc, min_spoke_num, nFE), dtype=np.complex64)
        ktraj_binned = np.zeros((target_phases, 2, min_spoke_num, nFE), dtype=np.float32)
        tshots_binned = np.zeros((target_phases, min_spoke_num,),dtype=np.float32)

        ttarget = [i / target_phases for i in range(target_phases)]
        ttarget = [x * RR for x in ttarget]

        for i in range(target_phases):
            kdata_binned[i, :, :min_spoke_num, :] = kdata[:, :, spoke_id_list[i][:min_spoke_num], :]
            ktraj_binned[i, :, :min_spoke_num, :] = ktraj[:, spoke_id_list[i][:min_spoke_num], :]
            tshots_binned[i, :min_spoke_num] = tshots[spoke_id_list[i][:min_spoke_num]]

        # _, num_coils, num_spokes, num_freq_enq = kdata.shape

        kdata = kdata_binned
        ktraj = ktraj_binned
        nt, _, new_nspokes, _ = kdata.shape

        _, num_coils, num_spokes, num_freq_enq = kdata.shape

        # 2: DCOMP calculation
        kdata = torch.from_numpy(kdata).reshape(target_phases, num_coils, -1).to(device)
        # Create masks
        kdata_abs = kdata.abs()
        kmask = torch.ones_like(kdata_abs)

        smap = torch.from_numpy(smap).to(device)
        ktraj = torch.from_numpy(ktraj).reshape(target_phases,2,-1).to(device)

        stacked_ktraj = ktraj

        # calculate decomposition field func
        weight_shapes = (target_phases, num_spokes)
        unmasked_dcomp = compute_dcf(stacked_ktraj, smap, weight_shapes, img_dim, op, op_adj, device)

        # Apply dcomp to data
        unmasked_kdata = kdata * torch.sqrt(unmasked_dcomp) 
        # for unmasked trajectories

        # Ccompute normalization
        unmasked_eig = []
        for phase in range(unmasked_kdata.shape[0]):
            A = lambda x: op(x.to(device), smap.to(device), stacked_ktraj[phase:phase+1].to(device), unmasked_dcomp[phase:phase+1].to(device))
            AT = lambda x: op_adj(x.to(device), smap.to(device), stacked_ktraj[phase:phase+1].to(device), unmasked_dcomp[phase:phase+1].to(device))

            def M(x):
                return A(AT(x))

            y = unmasked_kdata[phase:phase+1].clone() #.reshape(nFrames, nCh, nShots * nFE)
            b_k = torch.rand(*y.shape) + 1j*torch.rand(*y.shape)

            eig_M = torch.abs(max_eigenvalue(M, b_k, max_iter=30,verbose=False))
            unmasked_eig.append(eig_M)

        # Global normalization of phases
        # Try normalization with the maximum and with mean eigenvalue
        new_dcomp_max = unmasked_dcomp / torch.max(torch.Tensor([unmasked_eig]))
        new_dcomp_mean = unmasked_dcomp / torch.mean(torch.Tensor([unmasked_eig]))
        
        unmasked_kdata_max = unmasked_kdata / torch.sqrt(torch.max(torch.Tensor([unmasked_eig])))
        unmasked_kdata_mean = unmasked_kdata / torch.sqrt(torch.mean(torch.Tensor([unmasked_eig])))   

        caseid = data['caseid']

        unmasked_img_max = op_adj(unmasked_kdata_max, smap, ktraj, new_dcomp_max).detach().cpu().numpy()
        unmasked_img_mean = op_adj(unmasked_kdata_mean, smap, ktraj, new_dcomp_mean).detach().cpu().numpy()
        unmasked_img = torch.zeros((30,1,256,256))

        # Convert everything to numpy arrays
        if torch.is_tensor(kdata):
            kdata = kdata.detach().cpu().numpy()
        if torch.is_tensor(unmasked_kdata):
            unmasked_kdata = unmasked_kdata.detach().cpu().numpy()
        if torch.is_tensor(smap):
            smap = smap.detach().cpu().numpy()
        if torch.is_tensor(ktraj):
            ktraj = ktraj.detach().cpu().numpy()
        if torch.is_tensor(stacked_ktraj):
            stacked_ktraj = stacked_ktraj.detach().cpu().numpy()
        if torch.is_tensor(unmasked_dcomp):
            unmasked_dcomp = unmasked_dcomp.detach().cpu().numpy()
        if torch.is_tensor(target_img):
            target_img = target_img.detach().cpu().numpy()        
        if torch.is_tensor(unmasked_img):
            unmasked_img = unmasked_img.squeeze(1).detach().cpu().numpy()
        if torch.is_tensor(kmask):
            kmask = kmask.detach().cpu().numpy()
        if torch.is_tensor(unmasked_kdata_max):
            unmasked_kdata_max = unmasked_kdata_max.detach().cpu().numpy()
        if torch.is_tensor(unmasked_kdata_mean):
            unmasked_kdata_mean = unmasked_kdata_mean.detach().cpu().numpy()
        if torch.is_tensor(new_dcomp_max):
            new_dcomp_max = new_dcomp_max.detach().cpu().numpy()
        if torch.is_tensor(new_dcomp_mean):
            new_dcomp_mean = new_dcomp_mean.detach().cpu().numpy()

        # Save samples with maximum eigenvalue normalization
        unmasked_kdata = unmasked_kdata_max
        unmasked_dcomp = new_dcomp_max
        # Caluclate masks for zs-ssl
        unmasked_kdata = unmasked_kdata.reshape(unmasked_kdata.shape[0], unmasked_kdata.shape[1], num_spokes, -1)
        kmask = kmask.reshape(unmasked_kdata.shape[0], unmasked_kdata.shape[1], unmasked_kdata.shape[2], -1)
        remainder_mask, val_mask = uniform_selection(unmasked_kdata, kmask, rho=0.2)
        
        # Make all kdata and masks of shape (tphases, ncoils, num_spokes * nFE)
        kmask = kmask.reshape(unmasked_kdata.shape[0], unmasked_kdata.shape[1], -1)
        remainder_mask = remainder_mask.reshape(unmasked_kdata.shape[0], unmasked_kdata.shape[1], -1)
        val_mask = val_mask.reshape(unmasked_kdata.shape[0], unmasked_kdata.shape[1], -1)
        unmasked_kdata = unmasked_kdata.reshape(unmasked_kdata.shape[0], unmasked_kdata.shape[1], -1)

        res1 = {'caseid' : caseid, 
            'casename':case_name,
            'target_img' : target_img, 
            'img' : unmasked_img_max, 
            'full_kdata' : full_kdata, 
            'kdata' : unmasked_kdata, 
            'ktraj' : stacked_ktraj, 
            'smap' : smap, 
            'dcomp' : unmasked_dcomp, 
            'tshots' : tshots, 
            'num_spokes' : num_spokes,
            'kmask' : kmask,
            'remainder_mask': remainder_mask,
            'val_mask': val_mask,
            }

        unmasked_kdata_shape = unmasked_kdata.shape
        print(f'kdata dimension: {unmasked_kdata.reshape(unmasked_kdata_shape[0],unmasked_kdata_shape[1], -1, 256).shape}')

        with open(TARGET_PATH1 + case_name + '.pkl', 'wb') as f:
            pickle.dump(res1, f)


        # Save samples with mean eigenvalue normalization
        unmasked_kdata = unmasked_kdata_mean
        unmasked_dcomp = new_dcomp_mean
        # Caluclate masks for zs-ssl
        unmasked_kdata = unmasked_kdata.reshape(unmasked_kdata.shape[0], unmasked_kdata.shape[1], num_spokes, -1)
        kmask = kmask.reshape(unmasked_kdata.shape[0], unmasked_kdata.shape[1], unmasked_kdata.shape[2], -1)
        remainder_mask, val_mask = uniform_selection(unmasked_kdata, kmask, rho=0.2)
        
        # Make all kdata and masks of shape (tphases, ncoils, num_spokes * nFE)
        kmask = kmask.reshape(unmasked_kdata.shape[0], unmasked_kdata.shape[1], -1)
        remainder_mask = remainder_mask.reshape(unmasked_kdata.shape[0], unmasked_kdata.shape[1], -1)
        val_mask = val_mask.reshape(unmasked_kdata.shape[0], unmasked_kdata.shape[1], -1)
        unmasked_kdata = unmasked_kdata.reshape(unmasked_kdata.shape[0], unmasked_kdata.shape[1], -1)

        res2 = {'caseid' : caseid, 
            'target_img' : target_img, 
            'img' : unmasked_img_mean, 
            'full_kdata' : full_kdata, 
            'kdata' : unmasked_kdata, 
            'ktraj' : stacked_ktraj, 
            'smap' : smap, 
            'dcomp' : unmasked_dcomp, 
            'tshots' : tshots, 
            'num_spokes' : num_spokes,
            'kmask' : kmask,
            'remainder_mask': remainder_mask,
            'val_mask': val_mask,
            }

        with open(TARGET_PATH2 + case_name + '.pkl', 'wb') as f:
            pickle.dump(res2, f)