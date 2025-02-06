# UK Flower Classification with Deep Learning 🌺

This repository hosts a trained deep learning model for classifying 102 species of UK endemic flowers by building a custom classifier on top of a VGG16-based neural network. The model achieves 85% validation accuracy.

**Dataset:** [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

**Trained Model:** Available in this repository (`checkpoint.pth`)

### 📂 Repository Structure
- .gitattributes      # Used to track *checkpoint.pth* and linguistic documentation
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
To get started, simply download the *setup.sh* file, change the working directory to where the file was downloaded, and run the following code:
```bash
cd <path_to_the_downloaded_file>
bash setup.sh
```
## Running Inference
**After running setup.sh, classify a flower image:**
```bash
python predict.py <your_img_path/image.jpg> --gpu
```
## License
This project is open-source under the MIT License.
