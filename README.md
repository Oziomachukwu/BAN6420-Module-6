# Fashion MNIST Classification with CNN

## Overview
This project implements a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset using both **Python** and **R**. The implementation includes:
- A 6-layer CNN architecture
- Model training and evaluation
- Predictions on random test images
- Cross-language compatibility (Python/R)
- Robust path handling to ensure files save in the script directory

## Requirements

### Python
- Python 3.8+
- Packages:
    pip install tensorflow numpy matplotlib
### R 
- R 4.1+
- Packages
    install.packages(c("keras3", "this.path", "ggplot2", "dplyr", "tidyr"))
    remotes::install_github("rstudio/keras3")  # For Keras3
## Usage

### Python
Train the model:
  cd Python
  python fashion_mnist_cnn.py

Make predictions: 
  python fashio_predict.py

  
### R
Train the model:
  source("fashion_mnist_cnn.R")

Make predictions:
  source("fashion_predict.R")
  
## Key Features

Feature	            |Python Implementation	   |R Implementation
__________________________________________________________________
Model Format	      |.keras	                   |.keras (Keras3)
Visualization	      |Matplotlib	               |ggplot2
Path Handling	      |os module	               |this.path package
Data Serialization	|NumPy (.npy)	             |RDS (.rds)
Dependencies	      |TensorFlow/Keras	         |Keras3

## Notes
File Locations: All models/test data are saved in the same directory as the scripts.

### R Specifics:

- Uses Keras3 (modern API)
- Handles 1-based ↔ 0-based indexing internally
- Requires explicit loss/optimizer functions (e.g., loss_sparse_categorical_crossentropy())

Python Specifics:

- Uses standard Keras/TensorFlow
- Matplotlib for visualization
