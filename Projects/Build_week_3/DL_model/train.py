import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
from torch.nn import L1Loss, CrossEntropyLoss, BCELoss, Linear, ReLU, Sequential, Conv2d, Module, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from torchsummary import summary
from torchvision import transforms, datasets
import torch.nn.functional as F


from sklearn.metrics import accuracy_score, confusion_matrix

from model import grayscale_CNN
from data_handler import data_processor

#####  Load data  #####
root_dir = os.path.dirname(os.getcwd())
batchSize = 8
grayscale = True
rgb = False

train_loader, test_loader = data_processor(root_dir, batchSize, grayscale, rgb)


#####  cuda  #####

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#####  Define the model and variables for the training loop  #####

model = grayscale_CNN()

epochs = 3
print_every = 50
optimizer = Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


#####  Traing function #####

def train(model, optimizer, criterion, epochs, trainloader, testloader):
    train_loss = []
    val_loss = []
    accuracy = []

    for epoch in range(epochs):

        running_loss = 0

        print(f'Epoch: {epoch+1}/{epochs}')

        for i, (images, labels) in enumerate(iter(trainloader)):

            images, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % print_every == 0:
                print(f"\tIteration: {i}\t Loss: {running_loss/print_every:.4f}")
                running_loss = 0
    train_loss.append(running_loss/images.shape[0])

train(model,optimizer,criterion,epochs,train_loader,test_loader)
