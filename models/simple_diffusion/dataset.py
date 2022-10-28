import torch
import torchvision
from torch.utils.data import ConcatDataset
from torchvision import transforms


def load_transformed_dataset():
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2 - 1))
    ])
    train_data = torchvision.datasets.CIFAR10(root="../../datasets", download=True, train=True,
                                              transform=data_transforms)
    test_data = torchvision.datasets.CIFAR10(root="../../datasets", download=True, train=False,
                                             transform=data_transforms)
    return ConcatDataset([train_data, test_data])
