import torch
import torchkbnufft as tkbn
import kspace_ops as kso
import numpy as np

from torch.autograd import Variable, grad
from torch import nn


# Model adapted from https://github.com/js3611/Deep-MRI-Reconstruction/blob/d8a40efd892e57799c3413630e5cb92d5b035cf8/cascadenet/network/model.py#L23
# Original paper https://arxiv.org/pdf/1704.02422.pdf
class CascadeNet(nn.Module):
    """
    Model for Dynamic MRI Reconstruction using Convolutional Neural Networks
    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image, kspace_data, mask], each of shape (batch_size, 2, width, height, n_seq)
    Returns                                                                                  convert complex/real
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, 2, width, height, n_seq)
    """
    def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5, op=None, op_adj=None):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CascadeNet, self).__init__()

        if op is None or op_adj is None:
            raise NotImplementedError('One of the Fourier functions is not implemented in the model!')

        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks
        self.op = op
        self.op_adj = op_adj

        self.bcrnn = BCRNNlayer(n_ch, nf, ks)
        self.conv1_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv1_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv2_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv2_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv4_x = nn.Conv2d(nf, n_ch, ks, padding = ks//2)
        self.relu = nn.ReLU(inplace=True)

        dcs = []
        for i in range(nc):
            dcs.append(kso.DataConsistencyInKspace(op=self.op, op_adj=self.op_adj))
        self.dcs = dcs

    def forward(self, x, k, ktraj, smaps, dcomp, test=False):
        """
        x   - input in image domain, of shape (n, 2, nx, ny, n_seq)
        k   - initially sampled elements in k-space
        ktraj - corresponding nonzero location
        smaps - sensitivity maps for each coil
        dcomp - density compensation function
        test - True: the model is in test mode, False: train mode
        """
        net = {}

        n_batch, n_ch, width, height, n_seq = x.size()
        size_h = [n_seq*n_batch, self.nf, width, height]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            hid_init = Variable(torch.zeros(size_h)).cuda()

        for j in range(self.nd-1):
            net['t0_x%d'%j]=hid_init

        for i in range(1,self.nc+1):
            x = x.permute(4,0,1,2,3)
            x = x.contiguous()

            # TODO waste of memory with the 'net' dictionary, do this concisely
            net[f't{(i-1)}_x0'] = net[f't{(i-1)}_x0'].view(n_seq, n_batch,self.nf,width, height)
            net[f't{i}_x0'] = self.bcrnn(x, net[f't{(i-1)}_x0'], test)
            net[f't{i}_x0'] = net[f't{i}_x0'].view(-1,self.nf,width, height)

            net[f't{i}_x1'] = self.conv1_x(net[f't{i}_x0'])
            net[f't{i}_h1'] = self.conv1_h(net[f't{(i-1)}_x1'])
            net[f't{i}_x1'] = self.relu(net[f't{i}_h1']+net[f't{i}_x1'])

            net[f't{i}_x2'] = self.conv2_x(net[f't{i}_x1'])
            net[f't{i}_h2'] = self.conv2_h(net[f't{(i-1)}_x2'])
            net[f't{i}_x2'] = self.relu(net[f't{i}_h2']+net[f't{i}_x2'])

            net[f't{i}_x3'] = self.conv3_x(net[f't{i}_x2'])
            net[f't{i}_h3'] = self.conv3_h(net[f't{(i-1)}_x3'])
            net[f't{i}_x3'] = self.relu(net[f't{i}_h3']+net[f't{i}_x3'])

            net[f't{i}_x4'] = self.conv4_x(net[f't{i}_x3'])

            x = x.view(-1,n_ch,width, height)
            net[f't{i}_out'] = x + net[f't{i}_x4']

            net[f't{i}_out'] = net[f't{i}_out'].view(-1,n_batch, n_ch, width, height)
            net[f't{i}_out'] = net[f't{i}_out'].permute(1,2,3,4,0)
            net[f't{i}_out'].contiguous()
            net[f't{i}_out'] = self.dcs[i-1].perform(net[f't{i}_out'], k, ktraj, smaps, dcomp)
            x =  net[f't{i}_out']

            # clean up i-1
            # if test:
            to_delete = [ key for key in net if f't{(i-1)}' in key ]

            for elt in to_delete:
                del net[elt]

            torch.cuda.empty_cache()

        return net[f't{i}_out']

class DnCn3D(nn.Module):
    def __init__(self, n_channels=2, nc=5, nd=5, op=None, op_adj=None, **kwargs):
        super(DnCn3D, self).__init__()
        self.nc = nc
        self.nd = nd
        self.op = op
        self.op_adj = op_adj
        print('Creating D{}C{} (3D)'.format(nd, nc))
        conv_blocks = []
        dcs = []

        conv_layer = conv_block

        for i in range(nc):
            conv_blocks.append(conv_layer(n_channels, nd, conv_dim=2, **kwargs))
            dcs.append(kso.DataConsistencyInKspaceUnbatched(op=self.op, op_adj=self.op_adj))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = nn.ModuleList(dcs)

    def forward(self, x, k, ktraj, smaps, dcomp, mask):
        # n_seq, n_batch,self.nf,width, height
        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            x = self.dcs[i].perform(x, k, ktraj, smaps, dcomp, mask)

        return x



def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)


def relu():
    return nn.ReLU(inplace=True)


def conv_block(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=None):

    # convolution dimension (2D or 3D)
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
        # pad = dilation
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    def conv_i():
        conv_i = conv(nf,   nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)
        torch.nn.init.xavier_uniform(conv_i.weight)
        return conv_i#conv(nf,   nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)
    torch.nn.init.xavier_uniform(conv_1.weight)
    torch.nn.init.xavier_uniform(conv_n.weight)

    # relu
    nll = relu if nl == 'relu' else lrelu

    layers = [conv_1, nll()]
    for i in range(nd-2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)


class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations
    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    hidden_iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # add iteration hidden connection
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_iteration, hidden):
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        ih_to_ih = self.ih2ih(hidden_iteration)

        hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)

        return hidden


class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer
    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode
    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)

    def forward(self, input, input_iteration, test=False):
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            hid_init = Variable(torch.zeros(size_h)).cuda()

        output_f = []
        output_b = []
        # forward
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[i], input_iteration[i], hidden)
            output_f.append(hidden)

        output_f = torch.cat(output_f)

        # backward
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[nt - i - 1], input_iteration[nt - i -1], hidden)

            output_b.append(hidden)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output