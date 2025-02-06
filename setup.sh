#!/bin/bash

# Step 1: Install Dependencies
echo "📦 Installing dependencies..."
pip install torch torchvision numpy matplotlib pillow argparse

# Step 2: Clone the Repository
echo "📂 Cloning repository..."
git clone https://github.com/TLeonidas/uk-flower-classification.git
cd uk-flower-classification

# Step 3: Download the Trained Checkpoint (if not found)
if [ ! -f "checkpoint.pth" ]; then
    echo "⬇️ Downloading trained model checkpoint..."
    git lfs pull
fi

# Step 4: Ask User for Image Path
echo "🖼️ Please enter the full path to the image for inference:"
read image_path

# Step 5: Run Inference on User's Image
python predict.py "$image_path" --gpu

echo "✅ Setup Complete! Inference has been run on: $image_path"
