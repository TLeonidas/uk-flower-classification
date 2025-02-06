#!/bin/bash

# Step 1: Install Dependencies from requirements.txt
echo "📦 Installing dependencies..."
pip install --upgrade pip  # Ensure latest pip version
pip install -r requirements.txt || {
    echo "❌ Error: Failed to install dependencies. Check your Python & pip version."
    echo "Press Enter to exit."
    read
    exit 1
}

# Step 2: Clone the Repository (Check if Already Exists)
if [ -d "uk-flower-classification" ]; then
    echo "📂 Repository already exists. Pulling latest changes..."
    cd uk-flower-classification
    git pull origin main
else
    echo "📂 Cloning repository..."
    git clone https://github.com/TLeonidas/uk-flower-classification.git
    cd uk-flower-classification
fi

# Step 3: Download the Trained Checkpoint (if not found)
if [ ! -f "checkpoint.pth" ]; then
    echo "⬇️ Downloading trained model checkpoint..."
    git lfs pull
fi

# Step 4: Ask User for Image Path (Loop Until a Valid File is Provided)
while true; do
    echo "🖼️ Please enter the full path to the image for inference:"
    read image_path

    if [ -f "$image_path" ]; then
        break  # Valid path, exit loop
    else
        echo "❌ Error: The file '$image_path' does not exist. Please try again."
    fi
done

# Step 5: Verify PyTorch is Installed
if ! python -c "import torch" &> /dev/null; then
    echo "❌ Error: PyTorch is not installed. Please check requirements.txt and try again."
    echo "Press Enter to exit."
    read
    exit 1
fi

# Step 6: Run Inference on User's Image
python predict.py "$image_path" --gpu || {
    echo "❌ An error occurred while running inference. Please check your inputs."
    echo "Press Enter to exit."
    read
    exit 1
}

# Step 7: Keep Terminal Open After Execution
echo "✅ Inference complete! Press Enter to exit."
read
