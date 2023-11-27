import torch
import numpy as np

from torch import nn
from merlinth.complex import complex2real, real2complex

# Adapted fro Yaman et al.'s ZS-SSL
class data_consistency():
    """
    Data consistency class can be used for:
        -performing E^h*E operation in the paper
        -transforming final network output to kspace
    """

    def __init__(self, ktraj, smaps, dcomp, mask, op, op_adj):
        self.shape_list = mask.shape
        self.ktraj = ktraj
        self.sens_maps = smaps
        self.mask = mask
        self.dcomp = dcomp
        self.op = op
        self.op_adj = op_adj

    def EhE_Op(self, img, mu):
        """
        Performs (E^h*E+ mu*I) x
        """
        kspace = self.op(img, self.sens_maps, self.ktraj, self.dcomp)
        
        masked_kspace = kspace * self.mask

        img_recon = self.op_adj(masked_kspace, self.sens_maps, self.ktraj, self.dcomp)
        ispace = img_recon + mu * img
        return ispace

    def SSDU_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        and selects only loss mask locations(\Lambda) for computing loss
        """
        kspace = self.op(img, self.sens_maps, self.ktraj, self.dcomp)

        masked_kspace = kspace * self.mask

        return masked_kspace

    def Supervised_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        """
        kspace = self.op(img, self.sens_maps, self.ktraj, self.dcomp)
        return kspace

class dc_block(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, rhs, sens_maps, ktraj, dcomp, mask, mu, op, op_adj):
        """
        DC block employs conjugate gradient for data consistency,
        """

        mu = mu.type(torch.complex64).to(torch.device('cuda'))
        if rhs.dtype != torch.complex64:
            rhs = real2complex(rhs)

        Encoder = data_consistency(ktraj, sens_maps, dcomp, mask, op, op_adj)

        def body(i, rsold, x, r, p, mu):
            Ap = Encoder.EhE_Op(p, mu)
            alpha = torch.complex(torch.Tensor([rsold / torch.sum(torch.conj(p) * Ap).real]), torch.Tensor([0.])).to(torch.device('cuda'))

            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = (torch.sum(torch.conj(r) * r)).real
            beta = rsnew / rsold
            beta = torch.complex(torch.Tensor([beta]), torch.Tensor([0.])).to(torch.device('cuda'))
            p = r + beta * p            # rsnew instead of r??

            return i + 1, rsnew, x, r, p, mu

        x = torch.zeros_like(rhs)
        i, r, p = 0, rhs, rhs
        rsold = torch.sum(torch.conj(r) * r).float()

        # CG loop 
        # TODO code this cleaner
        while i < 6:
            i, rsold, x, r, p, mu = body(i, rsold, x, r, p, mu)

        cg_out = x

        return cg_out


class DC_wenqi(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, rhs, sens_maps, ktraj, dcomp, mask, mu, op, op_adj, init_kspace):
        kspace = op(rhs, sens_maps, ktraj, dcomp)
        masked_kspace = kspace * mask - init_kspace
        img_recon = op_adj(masked_kspace, sens_maps, ktraj, dcomp)
        return img_recon


def SSDU_kspace_transform(nw_output, sens_maps, ktraj, dcomp, mask, op, op_adj):
    """
    This function transforms unrolled network output to k-space at only unseen locations in training (\Lambda locations)
    """

    Encoder = data_consistency(ktraj, sens_maps, dcomp, mask, op, op_adj)
    nw_output_kspace = Encoder.SSDU_kspace(nw_output)

    return nw_output_kspace


def supervised_kspace_transform(img, sens_maps, ktraj, dcomp, mask, op, op_adj):
    """
    This function transforms unrolled network output to k-space at only unseen locations in training (\Lambda locations)
    """

    Encoder = data_consistency(ktraj, sens_maps, dcomp, mask, op, op_adj)
    nw_output_kspace = Encoder.Supervised_kspace(img)

    return nw_output_kspace

######################################################################################################################################
# Not really in active use, only for testing purposes
# Adapted form https://github.com/midas-tum/merlin_dev/blob/master/pytorch/merlinth/layers/complex_cg.py

class ComplexCG(torch.autograd.Function):
    @staticmethod
    def dotp(data1, data2):
        if data1.is_complex():
            mult = torch.conj(data1) * data2
        else:
            mult = data1 * data2
        return torch.sum(mult)

    @staticmethod
    def solve(x0, M, tol, max_iter):
        x = torch.zeros_like(x0)
        r = x0.clone()
        p = x0.clone()

        rTr = torch.norm(r).pow(2)

        it = 0
        while rTr > tol and it < max_iter:
            it += 1
            q = M(p)
            alpha = rTr / ComplexCG.dotp(p, q)
            x += alpha * p
            r -= alpha * q

            rTrNew = torch.norm(r).pow(2)

            beta = rTrNew / rTr

            p = r.clone() + beta * p
            rTr = rTrNew.clone()
        return x

    @staticmethod
    def forward(ctx, A, AH, max_iter, tol, lambdaa, x, y, *constants, mask, mu):
        ctx.tol = tol
        ctx.max_iter = max_iter
        ctx.A = A
        ctx.AH = AH

        def M(p):
            return AH(A(p, *constants), *constants) + lambdaa * p

        rhs = AH(y, *constants) + lambdaa * x
        ctx.save_for_backward(x, rhs, *constants, lambdaa)

        return ComplexCG.solve(rhs, M, ctx.tol, ctx.max_iter)

    @staticmethod
    def backward(ctx, grad_x):
        x, rhs, *constants, lambdaa = ctx.saved_tensors

        def M(p):
            return ctx.AH(ctx.A(p, *constants), *constants) + lambdaa * p

        Qe  = ComplexCG.solve(grad_x, M, ctx.tol, ctx.max_iter)
        QQe = ComplexCG.solve(Qe,     M, ctx.tol, ctx.max_iter)

        grad_x = lambdaa * Qe

        grad_lambdaa = ComplexCG.dotp(Qe, x).sum() - \
                       ComplexCG.dotp(QQe, rhs).sum()
        grad_lambdaa = torch.real(grad_lambdaa)

        output = None, None, None, None, grad_lambdaa, grad_x, None, *[None for _ in constants]
        return output
        
class CGClass(torch.nn.Module):
    def __init__(self, A, AH, max_iter=10, tol=1e-10):
        super().__init__()
        self.A = A
        self.AH = AH
        self.max_iter = max_iter
        self.tol = tol

        self.cg = ComplexCG
    
    def forward(self, lambdaa, x, y, *constants, mask, mu):
        out = torch.zeros_like(x)

        for n in range(x.shape[0]):
            cg_out = self.cg.apply(self.A, self.AH, self.max_iter, self.tol, lambdaa, x[n::1], y[n::1], *[c[n::1] for c in constants], mask, mu)
            out[n] = cg_out[0]
        return out
