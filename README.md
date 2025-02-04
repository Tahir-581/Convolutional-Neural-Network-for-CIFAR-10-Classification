# Convolutional Neural Network for CIFAR-10 Classification

## Overview
This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the CIFAR-10 dataset. The model is trained on 10 different categories of objects, achieving competitive accuracy through deep learning techniques.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Features
- Preprocessing: Normalization and one-hot encoding of labels.
- Model: CNN with convolutional, max-pooling, dropout, and dense layers.
- Training: Uses Adam optimizer with categorical cross-entropy loss.
- Evaluation: Accuracy and loss visualization over epochs.
- Prediction: Displays a sample test image with true and predicted labels.

## Installation
```bash
pip install tensorflow matplotlib numpy
```

## Usage
Run the script to train and evaluate the model:
```bash
python cnn_cifar10.py
```

## Model Architecture
- **Conv2D (32 filters, 3x3, ReLU)**
- **MaxPooling2D (2x2)**
- **Conv2D (64 filters, 3x3, ReLU)**
- **MaxPooling2D (2x2)**
- **Flatten**
- **Dense (128, ReLU, Dropout 0.5)**
- **Dense (10, Softmax)**

## Results
After training for 100 epochs with batch size 64, the model achieves:
- **Test Accuracy:** ~X% (depends on training outcome)
- **Visualization:** Training/Validation accuracy & loss plots included.

## Example Prediction
A random test image is classified, and the true vs. predicted labels are displayed.

## License
This project is open-source under the MIT License.

