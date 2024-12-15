## Description


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://github.com/baksho/ml-handson/blob/main/LICENSE)
[![Python 3.12.5](https://img.shields.io/badge/python-3.12.5-3670A0?style=for-the-badge&logo=python&logoColor=ffffff)](https://www.python.org/downloads/release/python-3125/)
[![NumPy 2.0.2](https://img.shields.io/badge/numpy-2.0.2-777BB4?style=for-the-badge&logo=numpy&logoColor=ffffff)](https://numpy.org/)
[![TensorFlow 2.18.0](https://img.shields.io/badge/tensorflow-2.18.0-FF6E00?style=for-the-badge&logo=tensorflow&logoColor=ffffff)](https://www.tensorflow.org/)
[![Keras 3.7.0](https://img.shields.io/badge/keras-3.7.0-D00000?style=for-the-badge&logo=keras&logoColor=ffffff)](https://keras.io/)
[![Matplotlib 3.9.3](https://img.shields.io/badge/matplotlib-3.9.3-3670A0?style=for-the-badge&logo=matplotlib&logoColor=ffffff
)](https://matplotlib.org/)
[![Pandas 2.2.3](https://img.shields.io/badge/pandas-2.2.3-130754?style=for-the-badge&logo=pandas&logoColor=ffffff
)](https://pandas.pydata.org/)
[![Seaborn 0.13.2](https://img.shields.io/badge/seaborn-0.13.2-7DB0BC?style=for-the-badge&logo=pandas&logoColor=ffffff
)](https://seaborn.pydata.org/)

 <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white" />

This repository contains a Python-based implementation of a **Convolutional Neural Network (CNN)** for recognizing handwritten digits using the **MNIST dataset**. The project demonstrates how to preprocess data, build and train a CNN model, and evaluate its performance on handwritten digit recognition tasks.

The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset.It is a dataset of `60,000` small square `28 × 28` pixel grayscale images of handwritten single digits between `0` and `9`.The task is to classify a given image of a handwritten digit into one of `10` classes representing integer values from `0` to `9`, inclusively. It can be simply imported from Keras Datasets using `from keras.datasets import mnist`.

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
Python 3.8+, TensorFlow 2.x, NumPy, Matplotlib, Pandas, Seaborn, Scikit-Learn

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
