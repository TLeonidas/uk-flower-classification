#!/bin/bash

# Step 0: Create and activate virtual environment
echo "Creating virtual environment..."
python -m venv flowerclass
source flowerclass/bin/activate

# Step 1: Install Dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Step 2: Clone the Repository
echo "Cloning repository..."
git clone https://github.com/TLeonidas/uk-flower-classification.git
cd uk-flower-classification

# Step 3: Check for the Trained Checkpoint
if [ ! -f "checkpoint.pth" ]; then
    echo "Error: checkpoint.pth not found in the repository."
    echo "Please make sure the model checkpoint is present."
    deactivate
    cd ..
    rm -rf flowerclass
    exit 1
fi

# Step 4: Ask User for Image Path
echo "Please enter the full path to the image for inference:"
read image_path

# Convert Windows paths (backslashes) to Unix-style paths
image_path=$(echo $image_path | sed 's/\\/\//g')

# Debugging: Print the normalized path
echo "Checking for image at: $image_path"

# Step 5: Check if Image Exists
if [ ! -f "$image_path" ]; then
    echo "Error: The file '$image_path' does not exist."
    echo "Make sure you provide the absolute path to the image."
    deactivate
    cd ..
    rm -rf flowerclass
    exit 1
fi

# Step 6: Run Inference on User's Image
python predict.py "$image_path" --gpu

# Step 7: Deactivate and remove virtual environment
echo "Cleaning up virtual environment..."
deactivate
cd ..
rm -rf flowerclass
