# PyTorch CIFAR-10 Image Classifier API

This project is an end-to-end implementation of an image classification model using PyTorch, deployed as a web API with FastAPI.

## Project Overview

This project builds and trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset and then serves the trained model via a RESTful API.

### Tech Stack
- **Python**
- **PyTorch**
- **FastAPI**
- **Uvicorn**
- **Pillow**

### Results
- The trained model achieved **54% accuracy** on the 10,000 test images after 2 epochs.

### How to Run
1. Clone the repository: `git clone <your-repo-url>`
2. Create and activate a Conda environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Train the model by running the Jupyter Notebook to generate the `cifar_net.pth` file.
5. Run the API: `python -m uvicorn main:app --reload`