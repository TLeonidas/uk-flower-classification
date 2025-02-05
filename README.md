# 🌸 UK Flower Classification with Deep Learning

This project is the final submission for the **AI Programming with Python Nanodegree** by **Udacity**, earned through a **scholarship from AWS**.

## 📌 Project Overview
This project trains a **deep learning model** to classify images of **102 flower species** using **transfer learning** with a **VGG16** convolutional neural network.

## 🚀 Features
✔ Transfer learning using **VGG16**  
✔ **PyTorch implementation**  
✔ **Command-line training & inference** (`train.py`, `predict.py`)  
✔ **85% validation accuracy**  

---

## 📂 File Structure
uk-flower-classification/
│── train.py            # Train the model & save checkpoint
│── predict.py          # Load model & classify an image
│── model.py            # Model architecture & checkpoint functions
│── utils.py            # Data loading & preprocessing
│── cat_to_name.json    # Category name mapping
│── checkpoint.pth      # Saved model checkpoint
│── requirements.txt    # Dependencies
│── README.md           # Project documentation

---

## ⚙️ Installation & Setup
1️⃣ **Clone the repository**  
```bash
git clone https://github.com/TLeonidas/uk-flower-classification.git
cd uk-flower-classification
```
2️⃣ **Install dependencies**
pip install -r requirements.txt

## Training the Model
python train.py flowers --epochs 7 --gpu

## Predicting an Image
python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --gpu

## License
This project is open-source under the MIT License.