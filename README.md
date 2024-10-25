# Butterfly Species Classifier using CNN

A deep learning project that classifies 75 different species of butterflies using a Convolutional Neural Network (CNN) implemented in PyTorch.

## Overview

This project implements a custom CNN architecture to classify butterfly images. The model is trained on a dataset containing images of 75 different butterfly species, making it a multi-class classification problem.

## Features

- Custom CNN architecture with multiple convolutional layers
- Data augmentation using random affine transformations, rotations, and horizontal flips
- Training and validation split (80/20)
- Performance metrics including accuracy and F1 score
- GPU acceleration support

## Requirements

- torch
- torchvision
- torchmetrics
- numpy
- pandas
- Pillow
- matplotlib
- tqdm
- torchinfo


## Model Architecture

The classifier consists of:
- 5 convolutional blocks with increasing channel dimensions (64 → 1024)
- Each block includes:
  - Convolutional layer
  - LeakyReLU activation
  - Batch normalization
  - MaxPooling
  - Dropout (in deeper layers)
- Final fully connected layers reducing to 75 classes

## Training

- Batch Size: 128
- Learning Rate: 0.0001
- Optimizer: AdamW with weight decay
- Loss Function: Cross Entropy Loss
- Number of Epochs: 20
- Data Augmentation:
  - Random rotations (-30° to +30°)
  - Random affine transformations
  - Random horizontal flips (40% probability)

## Usage

1. Clone the repository
2. Install the required dependencies
3. Organize your butterfly dataset in the specified structure
4. Run the Jupyter notebook

## Dataset

The dataset should be organized as follows:
- Training images in `data/butterfly_data/train/`
- Testing images in `data/butterfly_data/test/`
- Training labels in `Training_set.csv`
- Testing labels in `Testing_set.csv`

## Model Performance

The model's performance metrics include:
- Training and validation loss
- Accuracy
- F1 Score

Performance metrics are logged for each epoch during training.
