## Description


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/baksho/ml-handson/blob/main/LICENSE)
[![Python 3.12.5](https://img.shields.io/badge/python-3.12.5-blue?logo=python&logoColor=ffffff)](https://www.python.org/downloads/release/python-3125/)
[![Pytorch 2.5.1](https://img.shields.io/badge/pytorch-2.5.1-orange?logo=pytorch&logoColor=ffffff)](https://pytorch.org/)
[![Keras 3.7.0](https://img.shields.io/badge/keras-3.7.0-red?logo=keras&logoColor=ffffff)](https://keras.io/)
[![TensorFlow 2.18.0](https://img.shields.io/badge/tensorflow-2.18.0-orange?logo=tensorflow&logoColor=ffffff)](https://www.tensorflow.org/)

This repository contains a Python-based implementation of a **Convolutional Neural Network (CNN)** for recognizing handwritten digits using the **MNIST dataset**. The project demonstrates how to preprocess data, build and train a CNN model, and evaluate its performance on handwritten digit recognition tasks.

#### Features
- **MNIST Dataset**: Utilizes the benchmark dataset of `28 x 28` grayscale images of handwritten digits (0–9).
- **CNN Architecture**: Implements a custom-designed CNN with layers such as Convolution, MaxPooling, and Dense for feature extraction and classification.
- **Training & Evaluation**: Includes training on the MNIST dataset and evaluation metrics such as accuracy to measure performance.
- **Visualization**:
  - Plots of model training/validation accuracy and loss curves.
  - Visualization of test predictions using 3x5 grid subplots with predicted and actual labels.
- **Model Save**: The weights of the trained model are saved separately as `digit_recog_mnist_cnn.keras` file.

#### Project Highlights
- **Preprocessing**: Normalizes image data for faster convergence during training.
- **Model Training**: Uses TensorFlow/Keras for defining and training the CNN model.
- **Prediction**: Predicts test labels and visualizes results for better understanding.
- **Hyperparameter Tuning**: Experiments with CNN hyperparameters like filter sizes, number of layers, and learning rates to optimize accuracy.

#### Requirements
Python 3.8+, TensorFlow 2.x, NumPy, Matplotlib, Pandas, Matplotlib, Seaborn, Scikit-Learn

Install the required dependencies using:
`pip install -r requirements.txt`

#### How to Run
1. Clone the repository:
   `git clone https://github.com/your-username/handwritten-digit-recognition.git`

#### Results
Achieved `~99%` accuracy with `~0.33` loss on the MNIST test dataset. The model effectively recognizes handwritten digits and generalizes well to unseen data.

#### Future Work
- Experiment with more advanced architectures (e.g., ResNet, VGG).
- Extend the project to recognize handwritten text or other digit datasets.
- Deploy the trained model as a web app using Flask/Django.

#### License
This project is licensed under the MIT License.
