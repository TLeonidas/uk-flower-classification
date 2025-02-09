from model import load_checkpoint
from utils import process_image

def predict(image_path, checkpoint="checkpoint.pth", top_k=5, category_names="cat_to_name.json", gpu=False):
    """Predict the class of an image using a pretrained model."""
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    # Load model directly from checkpoint
    model = load_checkpoint(checkpoint)
    model.to(device)
    model.eval()

    # Process image
    image = process_image(image_path).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)

    # Extract top predictions
    probs, indices = torch.exp(output).topk(top_k)
    probs = probs.cpu().numpy().squeeze()
    indices = indices.cpu().numpy().squeeze()

    # Convert indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]

    # Load category names
    if category_names:
        with open(category_names, "r") as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name.get(cls, cls) for cls in classes]

    return probs, classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict an image class using a pretrained model")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K predictions")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Category mapping file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()
    probs, classes = predict(args.image_path, "checkpoint.pth", args.top_k, args.category_names, args.gpu)
    
    print("\n **Predictions:**")
    for i, (prob, flower) in enumerate(zip(probs, classes), 1):
        print(f"{i}. {flower}: {prob*100:.2f}%")
