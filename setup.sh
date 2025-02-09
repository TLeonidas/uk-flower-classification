#!/bin/bash

# Step 1: Install Dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip  # Ensure pip is up to date
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

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

# Convert Windows paths (backslashes) to Unix-style paths
image_path=$(echo $image_path | sed 's/\\/\//g')

# Debugging: Print the normalized path
echo "🔍 Checking for image at: $image_path"

# Step 5: Check if Image Exists
if [ ! -f "$image_path" ]; then
    echo "❌ Error: The file '$image_path' does not exist."
    echo "💡 Make sure you provide the **absolute path** to the image."
    exit 1
fi

# Step 6: Run Inference on User's Image
python predict.py "$image_path" --gpu
