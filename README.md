# Covid Image Classification with Chest X-Ray

Using Machine Learning and Deep Learning techniques so that we can train our CNN model for automating the whole diagnosis method to detect COVID-19 virus, using significant features given by the most linked features that are taken into consideration when evaluating the target.

## Table of Contents
* [Introduction](#introduction)
* [Dataset General info](#dataset-general-info)
* [Evaluation](#evaluation)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Run Example](#run-example)
* [Sources](#sources)

## Introduction

The progress of artificial intelligence has effectively contributed to the areas of life that have helped humans in all aspects of their lives, and this effective progress has invested, through its algorithms and technologies, the great development of resources and technologies that helped to implement these technologies as quickly as possible and with high efficiency that could be used in real time, and here, I single out computer vision technologies that have effectively contributed to various aspects of life that have supported humans in their work, and have also had a major and effective role in medical development and assisting doctors in diagnosing diseases, analyzing pests, detecting tumors, etc., even reaching advanced stages in supporting robots to perform operations, surgical procedures accuracy, especially the micro ones, and other effective factors of using artificial intelligence.
Hence, and based on this interesting introduction, I present to you my Computer Vision project, which is to explore lung diseases, especially during the COVID-19 period the whole world is suffering from COVID-19 disease and its proper and timely diagnosis is the need of the hour. So for building an efficient AI-based diagnostic system, whether they are Viruses, Covid, or a Normal lung.
Various advanced techniques have been used to deal with images and their exceptions, and to reach the best possible result, using a lot of effective algorithm and techniques with a good analysis (EDA), and comparing between them using logical thinking, and put my suggestions for solving it with the best possible ways and the current capabilities using Machine Learning and Deep Learning.

Hoping to improve it gradually in the coming times.

## Dataset General info

**General info about the dataset:**

**Context**
* The whole world is suffering from COVID-19 disease and its proper and timely diagnosis is the need of the hour. So for building an efficient AI-based diagnostic system we have collected Chest ray images from different sources so that we can train our CNN model for automating the whole diagnosis method. Therefore we have collected Chest ray images from different sources and research papers and combined them to create one comprehensive dataset that can be used by research community.

* This dataset is also used in COVID Lite paper which has shown significant results by building novel CNN based solution.

**Content**

* This dataset consists of a posteroanterior (PA) view of chest X-ray images comprising Normal, Viral, and CVOID-19 affected patients. There are total 1709 CXR images.
    
## Evaluation

The evaluation metric for this competition is [Mean F1-Score](https://en.wikipedia.org/wiki/F-score). The F1 score, commonly used in information retrieval, measures accuracy using statistics precision and recall.

## Technologies

* Programming language: Python.
* Libraries: Numpy, Matplotlib, Pandas, Seaborn, plotly, tensorflow, sklearn, Pillow, copy, os-sys, opencv, plotly, tqdm, keras. 
* Application: Jupyter Notebook.

## Setup

To run this project setup the following libraries on your local machine using pip on the terminal after installing Python:\
'''\
pip install numpy\
pip install matplotlib\
pip install pandas\
pip install seaborn\
pip install tensorflow\
pip install Pillow\
pip install pycopy-copy\
pip install os-sys\
pip install opencv-python\
pip install plotly\
pip install tqdm\
pip install scikit-learn\
pip install keras

'''\
To install these packages with conda run:\
'''

conda install -c anaconda numpy\
conda install -c conda-forge matplotlib\
conda install -c anaconda pandas\
conda install -c anaconda seaborn\
conda install -c conda-forge tensorflow\
conda install -c anaconda pillow\
conda install -c conda-forge pgcopy\
conda install -c jmcmurray os\
conda install -c conda-forge opencv\
conda install -c plotly plotly\
conda install -c conda-forge tqdm\
conda install -c anaconda scikit-learn\
conda install -c conda-forge keras

'''

## Features

* I present to you my project in computer vision, which is to explore lung diseases, especially during the COVID-19 period the whole world is suffering from COVID-19 disease and its proper and timely diagnosis is the need of the hour. So for building an efficient AI-based diagnostic system, whether they are Viruses, Covid, or a Normal lung. with various advanced techniques have been used to deal with images and their exceptions, and to reach the best possible result, with a good analysis (EDA), and comparing between them using logical thinking, and put my suggestions for solving it with the best possible ways and the current capabilities using Machine Learning.

### To Do:

**Briefly about the process of the project work, here are (some) insights that I took care of it:**

* Explored the dataset
- Here I Got benefits by using CSV file provided in this competition by joining image path in front of each image name, so we did flow from dataframe instead of flow from directory.

* Exploratory Data Analysis (EDA)
- Got more info about the data.
- Displayed the distribution of each class.
- Checked and manipulated with images by visualizing each class of them.
- Checked images shape, images color type, and data type.

* Preprocess the dataset.
- Splited the data into train, validation, and test set for better evaluation, checked the distribution of labeled data in all these splits, and recreated datafarmes (Train, Validation, and Test).
- Removed useless columns and prepare the dataset.
- Checked files validation.
- Did label encoder for object data.
- Dealt with unbalanced class by computing classes weight and entering it into algorithms in a fitted way.
- Did data augmentation, visualize its effect using one example, and in all splits.
- Ensure data augmentation using visualization and arrays.
* Modelling

** Custom CNN

- I used custom CNN with DenseNet, VGG, CNN without Batch Normalization, and CNN with Batch Normalization architecture. Manipulated bias, used He Normal for kernel initializer and ReLU for activation function for the hidden layers, Glorot Uniform for kernel initializer and softmax for activation function in the output layer, also MINI-BATCH, Adam, AdamW optimizer, and etc. Manipulated different learning rate, used categorical crossentropy loss algorithm, also I used callbacks (Early Stopping and Learning Rate Scheduler).

** Transfer Learning
- I used DenseNet, VGG16, VGG19, Inception, and ResNet with different algorithms architecture. Manipulated hyperparameters tuning, He Normal for kernel initializer and ReLU for activation function for the hidden layers, GlorotUniform for kernel initializer and softmax for activation function for the output layer, also MINI-BATCH, Adam, AdamW optimizer, and etc. Manipulated different learning rate, used categorical crossentropy loss algorithm, also I used callbacks (Early Stopping and ReduceLROnPlateau).

- I tried not to train the algorithm layer, but lately I found that in our case training some layer will be good for some algorithms, and results would be better.\
Finally, I used confusion matrix, accuracy score, classification report, evaluating it on the test set and visualize the predicted probabilities.

## Run Example
To run and show analysis, insights, correlation, and results between any set of features of the dataset, here is a simple example of it:

* Note: you have to use a jupyter notebook to open this code file.

1. Run the importing stage.

2. Load the dataset from Kaggle.

3. Select which cell you would like to run and view its output.

5. Run the rest of cells to end up in the training process and visualizing results.

4. Run Selection/Line in Python Terminal command (Shift+Enter).

## Sources
This data was taken from kaggle's competition:\
https://www.kaggle.com/t/80b4d31ea077422ebd9c95c46f1f9e52

