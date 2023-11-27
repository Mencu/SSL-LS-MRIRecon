import torch

from torch import nn
from merlinth.complex import complex2real, real2complex

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
        masked_kspace = kspace * self.mask          # mask 0 spokes auf 1 setzen - drin behalten
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
            alpha = torch.complex(torch.Tensor([rsold / torch.sum(torch.conj(p) * Ap).float()]), torch.Tensor([0.])).to(torch.device('cuda'))

            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = (torch.sum(torch.conj(r) * r)).float()
            beta = rsnew / rsold
            beta = torch.complex(torch.Tensor([beta]), torch.Tensor([0.])).to(torch.device('cuda'))
            p = r + beta * p

            return i + 1, rsnew, x, r, p, mu

        x = torch.zeros_like(rhs)
        i, r, p = 0, rhs, rhs
        rsold = torch.sum(torch.conj(r) * r).float()
        while i < 10:
            i, rsold, x, r, p, mu = body(i, rsold, x, r, p, mu)

        cg_out = x

        return complex2real(cg_out)

def SSDU_kspace_transform(nw_output, sens_maps, ktraj, dcomp, mask, op, op_adj):
    """
    This function transforms unrolled network output to k-space at only unseen locations in training (\Lambda locations)
    """

    nw_output = real2complex(nw_output)

    def ssdu_map_fn(input_elems):
        nw_output_enc, ktraj, sens_maps_enc, dcomp, mask_enc, op, op_adj = input_elems
        Encoder = data_consistency(ktraj, sens_maps_enc, dcomp, mask_enc, op, op_adj)
        nw_output_kspace = Encoder.SSDU_kspace(nw_output_enc)

        return nw_output_kspace

    masked_kspace = ssdu_map_fn((nw_output, ktraj, sens_maps, dcomp, mask, op, op_adj))

    return masked_kspace


class Masked_DCPM(torch.nn.Module):
    def __init__(self, A, AH, weight_init=1.0, weight_scale=1.0, requires_grad=True, **kwargs):
        super().__init__()

        self.A = A
        self.AH = AH

        self.weight_scale = weight_scale
        self.weight_init = weight_init
        self._weight = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32)*weight_init)
        self._weight.requires_grad_(requires_grad)
        self._weight.proj = lambda: self._weight.data.clamp_(1e-4, 1000)

        max_iter = kwargs.get('max_iter', 10)
        tol = kwargs.get('tol', 1e-10)
        self.prox = CGClass(A, AH, max_iter=max_iter, tol=tol)

    @property
    def weight(self):
        return self._weight * self.weight_scale

    def forward(self, inputs, scale=1.0):
        x = inputs[0]
        y = inputs[1]
        constants = inputs[2:]
        lambdaa = 1.0 / torch.max(self.weight * scale, torch.ones_like(self.weight)*1e-9)
        return self.prox(lambdaa, x, y, *constants)

    def __repr__(self):
        return f'DCPD(lambda_init={self.weight_init:.4g}, weight_scale={self.weight_scale}, requires_grad={self._weight.requires_grad})'


# Not used, just for testing purposes
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