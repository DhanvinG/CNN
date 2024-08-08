# Convolutional Neural Network for MNIST Digit Classification

## Overview

This project implements a convolutional neural network (CNN) to classify handwritten digits from the MNIST dataset. The CNN uses convolution and pooling layers to achieve high accuracy in digit recognition. The project is built using PyTorch and utilizes the Adam optimizer for training.

## Dataset

The MNIST dataset is a large collection of grayscale images of handwritten digits (0-9). It contains 60,000 training images and 10,000 testing images, each of size 28x28 pixels.

## Project Structure

- `CNN2.ipynb`: This Jupyter notebook contains the code for loading the dataset, building the CNN, training the model, and evaluating its performance.

## Requirements

To run this project, you need the following Python libraries:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`

You can install these dependencies using pip:

```bash
pip install torch torchvision numpy matplotlib
```

## Usage
Load the Dataset
The MNIST dataset is loaded using torchvision.datasets.MNIST.

## Preprocess the Data
The dataset is transformed into tensors and normalized.

## Build the CNN
A convolutional neural network is constructed with convolutional layers, pooling layers, and fully connected layers.

## Train the Network
The network is trained using the training dataset. The Adam optimizer is used to minimize the loss function during training.

## Evaluate the Network
The performance of the network is evaluated on the testing dataset, and the accuracy is calculated. Matplotlib is used to chart the results.

## Example
Here is a brief example of how to load the dataset and build a simple CNN:

```bash
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load the dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Define the CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12*12*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12*12*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
```
## Results
The CNN achieved an accuracy of approximately 98% on the MNIST testing dataset. The results are visualized using Matplotlib to chart the training and testing accuracy over epochs.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The MNIST dataset is sourced from the MNIST database of handwritten digits.
This project is inspired by various deep learning tutorials and documentation from PyTorch and other sources.
Repository
The complete code for this project is available in the GitHub repository: Convolutional Neural Network for MNIST Digit Classification
