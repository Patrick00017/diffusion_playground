from models.simple_diffusion.noise_schedule import c_sqrt_alphas_cumprod, c_sqrt_one_minus_alphas_cumprod, \
    c_posterior_variance
from models.simple_diffusion.tools import get_index_from_list
import torch


def forward_diffusion_sample(x_0, t, device='cpu'):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    :param device: str
    :param x_0:
    :param t: tensor[1]
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(c_sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        c_sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(
        device), noise.to(device)
