import torch
import torch.nn.functional as F
from models.simple_diffusion.tools import get_index_from_list


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def generate_noise_schedule(T):
    T = 300
    betas = linear_beta_schedule(T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), mode='constant', value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    return T, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance, betas, sqrt_recip_alphas


c_T, c_sqrt_alphas_cumprod, c_sqrt_one_minus_alphas_cumprod, c_posterior_variance, c_betas, c_sqrt_recip_alphas \
    = generate_noise_schedule(T=300)
