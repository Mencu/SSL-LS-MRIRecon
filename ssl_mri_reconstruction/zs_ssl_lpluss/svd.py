import torch


class CustomSVD(torch.autograd.Function):
    """
    A function that calculates a stable SVD with an epsilon in the backward pass to prevent exploding gradients.
    Adapted from tensorflow/python/ops/linalg_grad.py
    """

    @staticmethod
    def _SafeReciprocal(x, eps=1e-20):
        return x / (x ** 2 + eps)
    
    @staticmethod
    def forward(ctx, input, full_matrices=False):
        U, S, Vh = torch.linalg.svd(input, full_matrices=full_matrices)
        ctx.save_for_backward(input, U, S, Vh)
        ctx.full_matrices = full_matrices
        return U, S, Vh
    
    @staticmethod
    def backward(ctx, grad_u, grad_s, grad_vh):
        # Implement safe backward from tensorflow 

        input, U, S, Vh  = ctx.saved_tensors
        B, M, N = input.shape

        # NOTE V is back to original
        V = Vh.adjoint()
        grad_v = grad_vh.adjoint()

        # lines 860
        S = S.type(input.dtype)

        # lines 863-869
        use_adjoint = False 
        if M > N:
            # Compute the gradient for A^H = V * S^T * U^H, and (implicitly) take the
            # Hermitian transpose of the gradient at the end.
            use_adjoint = True
            M, N = N, M
            U, V = V, U
            grad_u, grad_v = grad_v, grad_u

        # lines 877-878
        S_mat = torch.diag_embed(S)
        grad_s_mat = torch.diag_embed(grad_s)
        S2 = S ** 2

        # lines 888-893
        F = CustomSVD._SafeReciprocal(torch.unsqueeze(S2, -2) - torch.unsqueeze(S2, -1))
        mask = torch.eye(M, device=S.device).repeat(B, 1, 1).bool()
        F[mask] = 0 + 1j * 0
        S_inv_mat = torch.diag_embed(CustomSVD._SafeReciprocal(S))

        # lines 895-902
        v1 = V[..., :, :M]
        grad_v1 = grad_v[..., :, :M]

        u_gu = U.adjoint() @ grad_u
        v_gv = v1.adjoint() @ grad_v1

        f_u = F * u_gu
        f_v = F * v_gv

        # line 904-908
        term1_nouv = grad_s_mat + (f_u + f_u.adjoint()) @ S_mat + S_mat @ (f_v + f_v.adjoint())
        term1 = U @ (term1_nouv @ v1.adjoint())
        
        if M == N:
            grad_a_before_transpose = term1
        else:
            gv1t = grad_v1.adjoint() # NOTE maybe here also conj()?
            gv1t_v1 = gv1t @ v1
            term2_nous = gv1t - (gv1t_v1 @ v1.adjoint())

            if ctx.full_matrices:
                v2 = V[..., :, M:N]
                grad_v2 = grad_v[..., :, M:N]

                v1t_gv2 = v1.adjoint() @ grad_v2
                term2_nous -= v1t_gv2 @ v2.adjoint()
        
        u_s_inv = U @ S_inv_mat
        term2 = u_s_inv @ term2_nous

        grad_a_before_transpose = term1 + term2        
        
        if torch.is_complex(input):
            eye = torch.eye(S.shape[-1], dtype=input.dtype, device=S.device).repeat(B, 1, 1)
            l = eye * v_gv
            term3_nouv = S_inv_mat @ (l.adjoint() - l)
            term3 = 1 / 2. * U @ (term3_nouv @ v1.adjoint())

            grad_a_before_transpose += term3

        if use_adjoint:
            grad_a = grad_a_before_transpose.adjoint()
        else:
            grad_a = grad_a_before_transpose

        return grad_a

