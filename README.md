# UK Flower Classification with Deep Learning 🌺

This repository hosts a trained deep learning model for classifying 102 species of UK endemic flowers, using a custom-built classifier on top of a Resnet50-based neural network. The model achieves 96% validation accuracy.

This project was originally developed as the final project for the AI Programming with Python Nanodegree by Udacity. The trained model is hosted for inference, allowing users to classify flower species using an image.

**Dataset:** [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

**Trained Model:** Available in this repository (`checkpoint.pth`)

**Model Architecture:** Resnet50-based with a custom classifier

### 📂 Repository Structure
- .gitattributes      # Used to track *checkpoint.pth* and linguistic documentation
- LICENSE             # MIT License
- README.md           # Documentation
- cat_to_name.json    # Maps category numbers to flower names
- checkpoint.pth      # Stored trained model weights
- model.py            # Loads the trained ResNet50 model
- predict.py          # Runs inference on an input image
- requirements.txt    # Dependencies
- setup.sh            # Automates setup & inference
---

## **Installation & Setup**
To get started, simply download the `setup.sh` file, navigate to its location in your Git CLI, and run:
```bash
bash setup.sh
```
## Running Inference
After running `setup.sh`, insert the full path to the image file, when prompted:
```bash
/path_to/flower.jpg
```
## License
This project is open-source under the MIT License.
