from torchvision import models
import torch
from torch import nn
from train_resnet import train_model
from torch import optim
from torchsummary import summary
from collections import OrderedDict


model_mobile = models.mobilenet_v2(pretrained=True)
for param in model_mobile.parameters():
    param.requires_grad = False

num_ftrs = model_mobile.classifier[1].in_features

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_ftrs, 512)), ('do1', nn.Dropout(0.7)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(512, 256)), ('do2', nn.Dropout(0.7)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(256, 128)), ('do3', nn.Dropout(0.7)),
                          ('relu3', nn.ReLU()),
                          ('fc4', nn.Linear(128, 64)), ('do2', nn.Dropout(0.7)),
                          ('relu4', nn.ReLU()),
                          ('output', nn.Linear(64, 3))
                          ]))


model_mobile.classifier[1] = classifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()


optimizer_conv = optim.SGD(model_mobile.classifier[1].parameters(), lr=0.001, momentum=0.8)

exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_conv, mode='min', factor=0.1, patience=4, verbose=True, min_lr=0.00000001)


model_mobile_t = train_model(model_mobile, criterion, optimizer_conv, exp_lr_scheduler, device)

torch.save(model_mobile_t.state_dict(), '2mobile_state_dict3.pth')
torch.save(model_mobile_t, '2mobile_gestures3.pth')

summary(model_mobile, (3, 255, 255))




