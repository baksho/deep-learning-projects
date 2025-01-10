## Description


[![Python 3.12.5](https://img.shields.io/badge/python-3.12.5-3670A0?style=for-the-badge&logo=python&logoColor=ffffff)](https://www.python.org/downloads/release/python-3125/)
[![NumPy 2.0.2](https://img.shields.io/badge/numpy-2.0.2-4D77CF?style=for-the-badge&logo=numpy&logoColor=ffffff)](https://numpy.org/)
[![TensorFlow 2.18.0](https://img.shields.io/badge/tensorflow-2.18.0-E55B2D?style=for-the-badge&logo=tensorflow&logoColor=ffffff)](https://www.tensorflow.org/)
[![Keras 3.7.0](https://img.shields.io/badge/keras-3.7.0-D00000?style=for-the-badge&logo=keras&logoColor=ffffff)](https://keras.io/)
[![Matplotlib 3.9.3](https://img.shields.io/badge/matplotlib-3.9.3-3670A0?style=for-the-badge&logo=matplotlib&logoColor=ffffff
)](https://matplotlib.org/)
[![Pandas 2.2.3](https://img.shields.io/badge/pandas-2.2.3-130754?style=for-the-badge&logo=pandas&logoColor=ffffff
)](https://pandas.pydata.org/)
[![Scikit-learn 1.6.0](https://img.shields.io/badge/scikit--learn-1.6.0-F79939?style=for-the-badge&logo=scikit-learn&logoColor=ffffff)](https://scikit-learn.org/stable/)
[![Seaborn 0.13.2](https://img.shields.io/badge/seaborn-0.13.2-7DB0BC?style=for-the-badge&logo=pandas&logoColor=ffffff
)](https://seaborn.pydata.org/)
[![Jupyter 4.3.3](https://img.shields.io/badge/jupyter-4.3.3-F37821?style=for-the-badge&logo=jupyter&logoColor=ffffff)](https://jupyter.org/)

This repository contains a Python-based implementation of a **Artificial Neural Network (ANN)** for predicting the housing prices in Perth (Western Australia).

The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset.It is a dataset of `60,000` small square `28 × 28` pixel grayscale images of handwritten single digits between `0` and `9`.The task is to classify a given image of a handwritten digit into one of `10` classes representing integer values from `0` to `9`, inclusively. It can be simply imported from Keras Datasets using `from keras.datasets import mnist`.

#### About Dataset
This data was scraped from http://house.speakingsame.com/ and includes $33566$ data from $322$ Perth suburbs, resulting in an average of about $100$ rows per suburb. Below features are available in the **Perth Housing Dataset**. The dataset is available in [Kaggle](https://www.kaggle.com/datasets/syuzai/perth-house-prices).

- `ADDRESS` : Physical address of the property ( we will set to index )
- `SUBURB` : Specific locality in Perth; a list of all Perth suburb can be found here
- `PRICE` : Price at which a property was sold (AUD)
- `BEDROOMS` : Number of bedrooms
- `BATHROOMS` : Number of bathrooms
- `GARAGE` : Number of garage places
- `LAND_AREA` : Total land area (m^2)
- `FLOOR_AREA` : Internal floor area (m^2)
- `BUILD_YEAR` : Year in which the property was built
- `CBD_DIST` : Distance from the centre of Perth (m)
- `NEAREST_STN` : The nearest public transport station from the property
- `NEAREST_STN_DIST` : The nearest station distance (m)
- `DATE_SOLD` : Month & year in which the property was sold
- `POSTCODE` : Local Area Identifier
- `LATITUDE` : Geographic Location (lat) of ADDRESS
- `LONGITIDE` : Geographic Location (long) of ADDRESS
- `NEAREST_SCH` : Location of the nearest School
- `NEAREST_SCH_DIST` : Distance to the nearest school
- `NEAREST_SCH_RANK` : Ranking of the nearest school

#### How to Run
1. **Clone the repository**:
   ```bash
   git clone https://github.com/baksho/deep-learning-projects.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd deep-learning-projects/House-price-prediction-using-ANN

3. **Run the `Jupyter Notebook` named `Handwritten_digit_recognition_with_CNN.ipynb` to train the CNN model**:
    -  Launch Jupyter Notebook:
       ```bash
       jupyter notebook
       ```
    - Navigate to the notebook `House_price_prediction_using_ANN.ipynb` and open it.
    - Execute all cells sequentially.
    - After training the model, all the weights will be saved in `house_price_prediction_model.keras` file.

4. **Run the python file to test the model for an random user input**:
   ```bash
   python predict_house_price.py
   ```

#### Results
Achieved `~99%` accuracy with `~0.33` loss on the MNIST test dataset. The model effectively recognizes handwritten digits and generalizes well to unseen data.

#### Reference
For better conceptual understanding on how a **Artificial Neural Network** works, please refer to my other GitHub repository: **[dl-nutshell](https://github.com/baksho/dl-nutshell/tree/main)**.

#### License
This project is licensed under the MIT License.
