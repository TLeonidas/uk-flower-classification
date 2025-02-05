# UK Flower Classification with Deep Learning 🪷

This repository hosts a trained deep learning model for classifying 102 species of UK endemic flowers using transfer learning with a VGG16-based neural network. The model achieves 85% validation accuracy.

#### **Dataset:** [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)  
#### **Trained Model:** Available in this repository soon (`checkpoint.pth`)  
#### **Inference:** Clone this repository and run inference instantly!  
---

### 📂 Repository Structure

- train.py            # Train the model & save checkpoint (Not necessary for inference)
- predict.py          # Load model & classify an image
- model.py            # Model architecture & checkpoint functions
- utils.py            # Data loading & preprocessing
- cat_to_name.json    # Category name mapping
- checkpoint.pth      # Saved model checkpoint
- requirements.txt    # Dependencies
- setup.sh            # Automates setup & inference
- README.md           # Documentation
---

## **Installation & Setup**
To get started, simply run:
```bash
bash setup.sh
```
## Running Inference
**After running setup.sh, classify a flower image:**  
```bash
python predict.py your_img_path/image.jpg --gpu
```
## License
This project is open-source under the MIT License.
