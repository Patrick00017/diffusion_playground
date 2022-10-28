import os.path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.simple_diffusion.tools import get_index_from_list, show_tensor_image
from models.simple_diffusion.noise_schedule import c_T, c_betas, c_sqrt_one_minus_alphas_cumprod, c_sqrt_recip_alphas, \
    c_posterior_variance
from models.simple_diffusion.model import model
import matplotlib.pyplot as plt
from models.simple_diffusion.dataset import load_transformed_dataset
from models.simple_diffusion.loss import l1_loss
from models.simple_diffusion.forward_diffusion import forward_diffusion_sample

IMG_SIZE = 32


@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(c_betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        c_sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(c_sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(c_posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image():
    device = 'cpu'
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(c_T / num_images)

    for i in range(0, c_T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, i // stepsize + 1)
            show_tensor_image(img.detach().cpu())
    plt.show()


def train():
    weight_path = './weights/simple_diffusion_cifar_10.pth'
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # prepare the datasets
    dataset = load_transformed_dataset()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
    optimizer = Adam(model.parameters(), lr=0.001)

    epochs = 100  # Try more!
    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            imgs, _ = batch
            optimizer.zero_grad()
            t = torch.randint(0, c_T, (batch_size,), device=device).long()
            imgs, noise = forward_diffusion_sample(imgs, t)
            noise_pred = model(imgs, t)
            loss = l1_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                sample_plot_image()
                torch.save(model.state_dict(), weight_path)


if __name__ == '__main__':
    train()
