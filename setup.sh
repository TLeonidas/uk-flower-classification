#!/bin/bash

# Step 1: Install Dependencies
echo "📦 Installing dependencies..."
pip install torch==2.5.0 torchvision==0.16.0 numpy matplotlib pillow argparse

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

# Step 5: Check if Image Exists
if [ ! -f "$image_path" ]; then
    echo "❌ Error: The file '$image_path' does not exist."
    exit 1
fi

# Step 6: Run Inference on User's Image
python predict.py "$image_path" --gpu

# Step 7: Keep Terminal Open After Execution
echo "✅ Inference complete! Press Enter to exit."
read  # <--- This keeps the window open until the user presses Enter.
