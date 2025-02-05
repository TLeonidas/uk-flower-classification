import torch
from torch import nn, optim
from torchvision import models

def build_model():
    """Build and return a VGG16 model with the same architecture."""
    model = models.vgg16(pretrained=True)

    # Freeze feature extractor layers
    for param in model.parameters():
        param.requires_grad = False

    # Define new classifier (SAME as in Notebook)
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    return model

def save_checkpoint(model, optimizer, epochs, filepath):
    """Save model checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_to_idx": model.class_to_idx,
        "epochs": epochs
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    return model
