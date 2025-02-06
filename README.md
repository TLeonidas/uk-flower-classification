# UK Flower Classification with Deep Learning 🌺

This repository hosts a trained deep learning model for classifying 102 species of UK endemic flowers using transfer learning with a VGG16-based neural network. The model achieves 85% validation accuracy.

#### Dataset: [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)  
#### Trained Model: Available in this repository (`checkpoint.pth`)  
#### Inference: Clone this repository and run inference instantly!  
---

### 📂 Repository Structure
- .gitattributes      # Used to track *checkpoint.pth*
- LICENSE             # MIT License
- README.md           # Documentation
- cat_to_name.json    # Category name mapping 
- checkpoint.pth      # Saved model checkpoint
- model.py            # Model architecture & checkpoint functions
- predict.py          # Load model & classify an image
- requirements.txt    # Dependencies
- setup.sh            # Automates setup & inference
- train.py            # Train the model & save checkpoint (Not necessary for inference)
- utils.py            # Data loading & preprocessing
---

## **Installation & Setup**
To get started, simply run the following code after cloning the repository and changing the directory to it:
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
