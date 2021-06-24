import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.pool1 = nn.MaxPool2d((2, 2))
        self.pool2 = nn.MaxPool2d((4, 4))
        self.drop_out1 = nn.Dropout2d(0.2)
        self.drop_out2 = nn.Dropout2d(0.4)

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(2, 2), stride=(1, 1), bias=True)
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(1, 1), bias=True)
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(4, 4), stride=(1, 1), bias=True)
        self.batchnorm3 = nn.BatchNorm2d(256)

        self.input = nn.Linear(256*14*14, 256, True)
        self.fc1 = nn.Linear(256, 128, True)
        self.fc2 = nn.Linear(128, 64, True)
        self.out = nn.Linear(64, 3)

    def forward(self, x):

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.batchnorm1(x)
        x = self.drop_out2(x)
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.batchnorm2(x)
        x = self.drop_out2(x)
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.batchnorm3(x)
        x = self.drop_out2(x)

        x = x.view(x.size()[0], -1)

        x = F.relu(self.input(x))
        x = self.drop_out2(x)
        x = F.relu(self.fc1(x))
        x = self.drop_out2(x)
        x = F.relu(self.fc2(x))
        x = self.drop_out1(x)
        x = self.out(x)
        return F.log_softmax(x, dim=1)
