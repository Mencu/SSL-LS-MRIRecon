import numpy as np
import torch

from PIL import Image

def uniform_selection(input_data, input_mask, rho=0.2, small_acs_block=(4, 4)):
    """
    :param rho: Sampling ratio for uniform sampling (int), percent in the loss mask
    """
    # input expects nrow x ncol x ncoil, but we have tphases x ncoils x nrow x ncol
    # Process it per phase

    ncoil, nrow, ncol = input_data.shape[1], input_data.shape[2], input_data.shape[3]

    trn_mask = np.zeros_like(input_mask)
    loss_mask = np.zeros_like(input_mask)

    # calculate masks per phase
    for phase in range(input_mask.shape[0]):
        inp_mask = input_mask[phase]

        # is this correct? should this be also done per coil?
        center_kx = int(find_center_ind(input_data[phase], axes=(0, 2)))
        center_ky = int(find_center_ind(input_data[phase], axes=(0, 1)))

        temp_mask = np.copy(inp_mask)
        temp_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
        center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

        pr = np.ndarray.flatten(temp_mask[0])       # take only nrow, ncols, NO ncoils
        # p makes it uniformly distributed among positive entries in mask
        ind = np.random.choice(np.arange(nrow * ncol),
                                size=int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))

        [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))

        # loss_mask = np.zeros_like(input_mask)
        loss_mask[phase][:,ind_x, ind_y] = 1

        trn_mask[phase] = inp_mask - loss_mask[phase]

    return trn_mask, loss_mask


def norm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor : It can be in image space or k-space.
    axes :  The default is (0, 1, 2).
    keepdims : The default is True.
    Returns
    -------
    tensor : applies l2-norm .
    """
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)

    if not keepdims: return tensor.squeeze()

    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace : nrow x ncol x ncoil.
    axes :  The default is (1, 2, 3).
    Returns
    -------
    the center of the k-space
    """

    center_locs = norm(kspace, axes=axes).squeeze()

    return np.argsort(center_locs)[-1:]


def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/tensor for mapping ind.
    Returns
    -------
    list of >=2D indices containing non-zero locations
    """

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]

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