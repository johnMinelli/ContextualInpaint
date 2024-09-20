import torch
from torch import nn

def logmeanexp(beta, tensor, dim, ignore_negative_inf=False, keepdim=False):
    n = torch.tensor(tensor.size(dim))
    if ignore_negative_inf:
        no_neg_inf = torch.sum((torch.isinf(tensor) & (tensor < 0)).to(torch.int), dim=dim)
        n = n - no_neg_inf
    lse = 1/beta * torch.logsumexp(beta * tensor, dim=dim, keepdim=keepdim)
    return lse - 1 / beta * torch.log(n)


class Energy(nn.Module):
    def __init__(self, a, b, beta_a=4, beta_b=4, normalize=True, mask_diagonale=False):
        super(Energy, self).__init__()
        if normalize:
            a = nn.functional.normalize(a, dim=-1)
            b = nn.functional.normalize(b, dim=-1)
        self.register_buffer('a', a)
        self.register_buffer('b', b)
        self.mask_diagonale = mask_diagonale
        self.normalize = normalize
        self.beta_a = beta_a
        self.beta_b = beta_b

    def forward(self, x):
        if self.normalize:
            x = nn.functional.normalize(x, dim=-1)
        attn = torch.einsum('...if,...jf->...ij', x, torch.concat([self.a, self.b], dim=0))
        if self.mask_diagonale:
            assert attn.shape[-1] == attn.shape[-2]
            attn[..., range(attn.size(-1)), range(attn.size(-1))] = -torch.inf

        a_energy = -logmeanexp(self.beta_a, attn[...,:len(self.a)], dim=-1)  # queries attending to the first part of the memory
        b_energy = -logmeanexp(self.beta_b, attn[...,len(self.a):], dim=-1)  # queries attending to the second part of the memory

        return a_energy, b_energy, attn


class BorderEnergy(Energy):
    def __init__(self, a, b, beta_a=4, beta_b=4, beta_border=4, normalize=True, mask_diagonale=False):
        super(BorderEnergy, self).__init__(a, b, beta_a, beta_b, normalize=normalize, mask_diagonale=mask_diagonale)
        self.beta_border = beta_border

    def forward(self, x, return_dict=False):
        a_energy, b_energy, attn = super().forward(x)
        union_energy = -1/self.beta_border*torch.logaddexp(-self.beta_border*a_energy, -self.beta_border*b_energy) + 1/self.beta_border*torch.log(torch.tensor(2))  # queries attending to whole memory
        border_energy = a_energy + b_energy - 2*union_energy
        if return_dict:
            return border_energy, {
                'attn': attn,
                'a_energy': a_energy,
                'b_energy': b_energy,
                'onesided_energy': a_energy-b_energy,
                'union_energy': union_energy,
            }
        else:
            return border_energy


class OneSidedEnergy(Energy):
    def __init__(self, a, b, beta_a, beta_b, normalize=True, mask_diagonale=False):
        super(OneSidedEnergy, self).__init__(a, b, beta_a, beta_b, normalize=normalize, mask_diagonale=mask_diagonale)

    def forward(self, x, return_dict=False):
        a_energy, b_energy, attn = super().forward(x)
        one_sided_energy = a_energy - b_energy
        if return_dict:
            return one_sided_energy, {
                'attn': attn,
                'a_energy': a_energy,
                'b_energy': b_energy,
            }
        else:
            return one_sided_energy


class HopfieldEnergy(Energy):
    def __init__(self, a, beta_a, normalize=True):
        super(HopfieldEnergy, self).__init__(a, a, beta_a, beta_a, normalize=normalize)

    def forward(self, x, mask_diagonale=False, return_dict=False):
        a_energy, _, attn = super().forward(x, mask_diagonale)
        maxnorm = torch.max(torch.functional.norm(self.a, dim=-1))
        if self.normalize:
            torch.allclose(maxnorm, torch.tensor(1.))
        if self.normalize:
            x = torch.nn.functional.normalize(x, dim=-1)
        energy = a_energy + 1/2 * torch.einsum('...f,...f->...', x, x) + 1/2 * maxnorm**2
        if return_dict:
            return energy, {
                'attn': attn,
                'a_energy': a_energy,
            }
        else:
            return energy
