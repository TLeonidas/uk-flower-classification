import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

def load_data(data_dir):
    """Load training, validation, and test datasets."""
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    dataloaders = {
        "train": DataLoader(train_data, batch_size=128, shuffle=True),
        "valid": DataLoader(valid_data, batch_size=64, shuffle=False),
        "test": DataLoader(test_data, batch_size=64, shuffle=False)
    }

    return train_data, dataloaders

def process_image(image_path):
    """Process an image for model inference."""
    image = Image.open(image_path)
    image = image.resize((256, 256)).crop((16, 16, 240, 240))
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))

    return torch.tensor(np_image).float()
