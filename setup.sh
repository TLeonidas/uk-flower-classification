#!/bin/bash

# Step 1: Install Dependencies from requirements.txt
echo "📦 Installing dependencies..."
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

# Step 5: Run Inference on User's Image
python predict.py "$image_path" --gpu || {
    echo "❌ An error occurred while running inference. Please check your inputs."
    echo "Press Enter to exit."
    read
    exit 1
}

# Step 6: Keep Terminal Open After Execution
echo "✅ Inference complete! Press Enter to exit."
read
