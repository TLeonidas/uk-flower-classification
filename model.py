import torch
from torchvision import models

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))

    # Load ResNet50 instead of VGG
    model = models.resnet50(pretrained=True)

    # Replace classifier with saved one
    model.fc = checkpoint['classifier']  # ResNet uses `fc`, not `classifier`
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
