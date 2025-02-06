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

# Step 4: Run Inference on a Sample Image
echo "🖼️ Running inference on sample image..."
python predict.py flowers/test/1/image_06743.jpg --gpu

echo "✅ Setup Complete! You can now use the model for inference."
