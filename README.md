# UK Flower Classification with Deep Learning 🌺

This repository hosts a trained deep learning model for classifying 102 species of UK endemic flowers, using a custom-built classifier on top of a VGG16-based neural network. The model achieves 85% validation accuracy.

This project was developed as the final project for the AI Programming with Python Nanodegree by Udacity. The trained model is hosted for inference, allowing users to classify flower species using an image.

**Dataset:** [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

**Trained Model:** Available in this repository (`checkpoint.pth`)

**Model Architecture:** VGG16-based with a custom classifier

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
To get started, simply download the `setup.sh` file, navigate to its location, and run:
```bash
cd <path_to_the_downloaded_file>
bash setup.sh
```
## Running Inference
**After running setup.sh, insert the full path to the image file, when prompted:**
```bash
</path_to/flower.jpg>
```
## License
This project is open-source under the MIT License.
