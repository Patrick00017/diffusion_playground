import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from models.simple_diffusion.dataset import load_transformed_dataset
from models.simple_diffusion.forward_diffusion import forward_diffusion_sample
from models.simple_diffusion.tools import show_tensor_image
from models.simple_diffusion.noise_schedule import c_T
import numpy as np

if __name__ == '__main__':
    batch_size = 12
    dataset = load_transformed_dataset()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # # Simulate forward diffusion
    image = next(iter(train_loader))[0]
    print(image.shape)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(c_T / num_images)

    for idx in range(0, c_T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images + 1, (idx // stepsize) + 1)
        image, noise = forward_diffusion_sample(image, t)
        print(image.shape)
        show_tensor_image(image)
    plt.show()
