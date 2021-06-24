import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, Conv2d, Module, BatchNorm2d, Dropout
import torch.nn.functional as F
from torchsummary import summary

# layer architecture for grayscale images
# dummy layers for now --> can be improved


# Class with sequential API

class grayscale_CNN(nn.Module):
    
    def __init__(self):

        super(grayscale_CNN, self).__init__()
        
        #first conv layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        #linear layer dimension = (( layersize - kernelsize +2(padding) ) / stride ) + 1

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 60 * 60, 3)


    def forward(self, x):
      # Convolution 1
      out = self.conv1(x)
      out = self.relu1(out)
      # Max pool 1
      out = self.maxpool1(out)

      # Convolution 2 
      out = self.conv2(out)
      out = self.relu2(out)
      # Max pool 2 
      out = self.maxpool2(out)

      # Resize
      out = out.view(out.size(0), -1)

      # Linear function (readout)
      out = self.fc1(out)
      out = F.softmax(out, dim=1)

      return out


# # layer architecture for rgb images
class rgb_CNN(Module):
    
    def __init__(self):

        super(rgb_CNN, self).__init__()
        
        #first conv layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        #linear layer dimension = (( layersize - kernelsize +2(padding) ) / stride ) + 1

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 60 * 60, 3)


    def forward(self, x):
      # Convolution 1
      out = self.conv1(x)
      out = self.relu1(out)
      # Max pool 1
      out = self.maxpool1(out)

      # Convolution 2 
      out = self.conv2(out)
      out = self.relu2(out)
      # Max pool 2 
      out = self.maxpool2(out)

      # Resize
      out = out.view(out.size(0), -1)

      # Linear function (readout)
      out = self.fc1(out)
      out = F.softmax(out, dim=1)

      return out

grayscale_model = grayscale_CNN()
rgb_model = rgb_CNN()

'''
summary(grayscale_model, (1, 255, 255)) #input size
summary(rgb_model, (3, 255, 255)) #input size
'''
