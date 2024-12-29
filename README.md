# Deep Learning HW3: Multi-Layer Perceptron (MLP) for Classification

This repository contains the implementation of a **Multi-Layer Perceptron (MLP)** for classification tasks as part of Homework 3 for a Deep Learning course. The MLP is a fully connected neural network trained using backpropagation to classify input data into multiple categories.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
  - [Installation](#installation)
  - [Running the Script](#running-the-script)
- [Example Output](#example-output)
- [Dependencies](#dependencies)
- [License](#license)

## Overview
A Multi-Layer Perceptron (MLP) is a fundamental building block of deep learning. This implementation demonstrates a simple MLP model for classification tasks, focusing on:

1. Implementing backpropagation from scratch.
2. Exploring the effects of hyperparameters such as learning rate and hidden layer size.
3. Training on datasets to achieve high classification accuracy.

## Features
- **Flexible Architecture:** Adjust the number of hidden layers and neurons per layer.
- **Activation Functions:** Includes ReLU for non-linear transformations.
- **Training Metrics:** Tracks loss and accuracy during training.
- **Efficient Backpropagation:** Implements gradient-based optimization from scratch.

## Usage

### Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/sivaciov/Deep-Learning-HW3.git
cd Deep-Learning-HW3
```

### Running the Script
Ensure you have Python 3.6 or later installed. Run the `mlp_classifier.py` script as follows:

```bash
python mlp_classifier.py --train_file <path_to_training_data> --test_file <path_to_test_data> --num_epochs <epochs> --learning_rate <lr> --hidden_size <size>
```

#### Command-line Arguments
- `--train_file`: Path to the training dataset (CSV or compatible format).
- `--test_file`: Path to the test dataset (CSV or compatible format).
- `--num_epochs`: Number of training epochs.
- `--learning_rate`: Learning rate for gradient descent.
- `--hidden_size`: Number of neurons in the hidden layer(s).

Example:
```bash
python mlp_classifier.py --train_file data/train.csv --test_file data/test.csv --num_epochs 50 --learning_rate 0.01 --hidden_size 128
```

## Example Output
The script will output the training and test performance metrics, such as loss and accuracy.

Sample output:
```
Epoch 1/50: Loss = 1.23, Accuracy = 65.0%
Epoch 50/50: Loss = 0.32, Accuracy = 92.5%
Final Test Accuracy: 91.8%
```

## Dependencies
This implementation uses only the Python standard library. However, the following optional library is recommended for enhanced performance:
- `numpy`

Install using:
```bash
pip install numpy
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to explore, adapt, and extend the code to fit your needs. Contributions and feedback are always welcome!
