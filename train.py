import argparse
import torch
from torch import optim, nn
from model import build_model, save_checkpoint
from utils import load_data

def train_model(data_dir, arch, learning_rate, hidden_units, epochs, save_dir, gpu):
    """Train the model using the specified architecture and hyperparameters."""
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    # Load data
    train_data, dataloaders = load_data(data_dir)

    # Load model (NOW supports multiple architectures)
    model = build_model(arch, hidden_units)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        
        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}.. Training Loss: {running_loss:.3f}")

    # Save model
    model.class_to_idx = train_data.class_to_idx
    save_checkpoint(model, optimizer, epochs, f"{save_dir}/checkpoint.pth")
    print(f"Model saved successfully at {save_dir}/checkpoint.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a new model.")
    parser.add_argument("data_dir", type=str, help="Path to dataset")
    parser.add_argument("--arch", type=str, default="vgg16", choices=["vgg16", "resnet50"], help="Model architecture")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Hidden units in classifier")
    parser.add_argument("--epochs", type=int, default=7, help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory to save the model")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()
    train_model(args.data_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.save_dir, args.gpu)
