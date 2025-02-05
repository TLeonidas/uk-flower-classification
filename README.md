# UK Flower Classification with Deep Learning 🪷

This repository hosts my final project for the AI Programming with Python Nanodegree program by Udacity, earned through a scholarship from AWS.

## Project Overview
The project trains a deep learning model to classify images of 102 flower species in [this dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) using transfer learning with a VGG16 or ResNet50 convolutional neural network.

## Features
✔ Transfer learning using VGG16 and ResNet50  
✔ PyTorch implementation  
✔ Command-line training & inference (`train.py`, `predict.py`)  
✔ 85% validation accuracy

---

## 📂 Repository Structure

- train.py            # Train the model & save checkpoint
- predict.py          # Load model & classify an image
- model.py            # Model architecture & checkpoint functions
- utils.py            # Data loading & preprocessing
- cat_to_name.json    # Category name mapping
- checkpoint.pth      # Saved model checkpoint
- requirements.txt    # Dependencies
- README.md           # Project documentation

---

## ⚙️ Installation & Setup
1️⃣ **Clone the repository**  
```bash
git clone https://github.com/TLeonidas/uk-flower-classification.git
cd uk-flower-classification
```
2️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```
## Training the Model
```bash
python train.py flowers --epochs 7 --gpu
```
## Predicting an Image
```bash
python predict.py flowers/test/100/image_07939.jpg checkpoint.pth --gpu
```
## License
This project is open-source under the MIT License.
