import torch
from torch import nn
from torchvision import models

def build_model(arch="vgg16", hidden_units=512):
    """Build a pre-trained model (VGG16 or ResNet50) with a custom classifier."""
    
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        input_size = 25088  # VGG16's classifier input size
        model.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )

    elif arch == "resnet50":
        model = models.resnet50(pretrained=True)
        input_size = model.fc.in_features  # ResNet50's FC input size
        model.fc = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )

    else:
        raise ValueError("Only 'vgg16' and 'resnet50' are supported.")

    # Freeze feature extractor layers
    for param in model.parameters():
        param.requires_grad = False

    return model

def save_checkpoint(model, optimizer, epochs, filepath):
    """Save model checkpoint."""
    checkpoint = {
        "arch": model.__class__.__name__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_to_idx": model.class_to_idx,
        "epochs": epochs
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath):
    """Load a model checkpoint and map it to the appropriate device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    model.to(device)
    return model
