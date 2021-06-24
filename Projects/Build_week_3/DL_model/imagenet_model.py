from torchvision import models
from collections import OrderedDict
from torch import nn
#import data_handler as dh


#give the path to the root directory here
# root_dir = ''       #use \\ if you are on windows
# batchSize = 2
# grayscale = True
# rgb = False

# train_loader, test_loader = dh.data_processor(root_dir, batchSize, grayscale, rgb)

model = models.resnet50(pretrained = True)
print(model)

print(f"The last layer of resnet model is: ", model.fc)


# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False


classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 1000)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(1000, 100)),
                          ('relu2', nn.ReLU()),  
                          ('fc3', nn.Linear(100, 3)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.fc = classifier # replace the classifier of resnet with our custom classifier
print(model)