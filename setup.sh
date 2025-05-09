#!/bin/bash

# Step 0: Create and activate virtual environment
echo "Creating virtual environment..."
python -m venv flowerclass

# Check if venv was created
if [ ! -d "flowerclass" ]; then
    echo "Failed to create virtual environment. Exiting..."
    read -n 1 -s -r -p "Press any key to exit..."
    exit 1
fi

# Activate depending on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source flowerclass/Scripts/activate
else
    source flowerclass/bin/activate
fi
echo "Virtual environment created and activated."

# Step 1: Install Dependencies
echo
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Step 2: Clone the Repository
echo
echo "Cloning repository..."
git clone https://github.com/TLeonidas/uk-flower-classification.git
cd uk-flower-classification

# Step 3: Check for the Trained Checkpoint
if [ ! -f "checkpoint.pth" ]; then
    echo
    echo "Error: checkpoint.pth not found in the repository."
    echo "Please make sure the model checkpoint is present and spelled correctly."
    read -n 1 -s -r -p "Press any key to exit..."
    deactivate
    cd ..
    rm -rf flowerclass
    exit 1
fi

# Step 4 & 5: Ask User for Image Path and Check If Exists
while true; do
    echo
    echo "Please enter the full path to the image for inference:"
    read image_path

    # Convert Windows-style backslashes to Unix-style forward slashes
    image_path=$(echo "$image_path" | sed 's/\\/\//g')

    echo "Checking for image at: $image_path"

    if [ -f "$image_path" ]; then
        break  # exit loop if the file exists
    else
        echo
        echo "Error: The file '$image_path' does not exist."
        echo "Make sure you provide the absolute path to the image."
        read -n 1 -s -r -p "Press any key to try again..."
    fi
done

# Step 6: Run Inference on User's Image
echo
echo "Running inference..."
python predict.py "$image_path" --gpu
read -n 1 -s -r -p "Inference complete. Press any key to clean up and exit..."

# Step 7: Deactivate and remove virtual environment
echo
echo "Cleaning up virtual environment..."
deactivate
cd ..
rm -rf flowerclass
