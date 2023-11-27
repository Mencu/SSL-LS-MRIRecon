import torch
import numpy as np
import data_consistency as dcp

from merlinth.complex import complex2real, real2complex
from merlinth.models.cnn import Real2chCNN
from PIL import Image
from torch import nn

class ScaleLayer(nn.Module):
   def __init__(self, init_value=1e-1):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, input):
       return input * self.scale


def res_convs(conv, nf=64, ks=3, pad_dilconv=1, dilation=1, is_relu=False, is_scaling=False):

    conv_1 = conv(nf, nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)
    torch.nn.init.normal_(conv_1.weight,mean=0.0,std=0.05)

    if is_relu:
        relu1 = nn.ReLU()
    else:
        relu1 = nn.Identity()

    conv_2 = conv(nf, nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)
    torch.nn.init.normal_(conv_2.weight,mean=0.0,std=0.05)

    if is_scaling:
        scale = ScaleLayer(0.1)
    else:
        scale = nn.Identity()

    return nn.Sequential(conv_1, relu1, scale)


class ResNet(nn.Module):
    def __init__(self, n_ch, nb_res_blocks, nf=64, ks=3, dilation=1, conv_dim=2, n_out=2):
        super(ResNet, self).__init__()
        # convolution dimension (2D or 3D)
        if conv_dim == 2:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d

        # output dim: If None, it is assumed to be the same as n_ch
        if not n_out:
            n_out = n_ch

        # dilated convolution
        pad_conv = "same"
        if dilation > 1:
            # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
            # pad = dilation
            pad_dilconv = dilation
        else:
            pad_dilconv = pad_conv

        self.nb_res_blocks = nb_res_blocks

        # First Layer
        self.conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
        torch.nn.init.normal_(self.conv_1.weight,mean=0.0,std=0.05)

        # Last Layer
        self.conv_n = conv(nf, nf, ks, stride=1, padding=pad_conv, bias=True)
        torch.nn.init.normal_(self.conv_n.weight,mean=0.0,std=0.05)

        self.residual = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)
        torch.nn.init.normal_(self.conv_n.weight,mean=0.0,std=0.05)

        res_blocks = []
        for i in np.arange(1, nb_res_blocks + 1):
            res_blocks.append(nn.Sequential(res_convs(conv,nf,ks,pad_dilconv,dilation,is_relu=True,is_scaling=False),\
                res_convs(conv,nf,ks,pad_dilconv,dilation,is_relu=False,is_scaling=True)))

        self.res_blocks = nn.ModuleList(res_blocks)


    def forward(self,x):
        # In tf "FirstLayer" scope
        x_init = self.conv_1(x)
        x = x_init
        for i in range(self.nb_res_blocks):
            x_cnn = self.res_blocks[i](x)
            x = x + x_cnn
        # In tf "LastLayer" scope
        x = self.conv_n(x) + x_init
        # In tf "Residual" scope
        x = self.residual(x)
        return x

class UnrolledNet(nn.Module):
    def __init__(self, in_ch, nb_unroll_blocks, nb_res_blocks, op, op_adj, out_ch=2):
        super(UnrolledNet, self).__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nb_unroll_blocks = nb_unroll_blocks
        self.nb_res_blocks = nb_res_blocks
        self.op = op
        self.op_adj = op_adj

        self.denoiser = Real2chCNN(num_layer=self.nb_res_blocks)
        self.mu_penalty = nn.Parameter(torch.FloatTensor([.05]))
        self.dc = dcp.dc_block()

    def forward(self, input_x, sens_maps, ktraj, dcomp, trn_mask, loss_mask):
        x = input_x

        for i in range(self.nb_unroll_blocks):
            x = self.denoiser(x)
            rhs = input_x + self.mu_penalty * x
            x = self.dc(rhs, sens_maps, ktraj, dcomp, trn_mask, self.mu_penalty, self.op, self.op_adj)

        nw_kspace_output = dcp.SSDU_kspace_transform(x, sens_maps, ktraj, dcomp, loss_mask, self.op, self.op_adj)

        return x, nw_kspace_output, rhs, self.mu_penalty