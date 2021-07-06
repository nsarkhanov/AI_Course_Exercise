from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from torchvision import models
from torch import nn
from torch import optim
from torchsummary import summary
from collections import OrderedDict
import torch
import time
import copy

data_dir = '../images/'
batchSize = 128

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.RandomRotation(30),
        transforms.GaussianBlur(kernel_size=(3, 3)),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'validation']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'validation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
class_names = image_datasets['train'].classes


def train_model(model, criterion, optimizer, scheduler, device, num_epochs=40):
    train_loss = []
    eval_loss = []
    train_acc = []
    eval_acc = []
    since = time.time()
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        if epoch % 20 == 0:
            torch.save(model, f'Dense_gestures_epoch{epoch}.pth')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss, eval_loss, train_acc, eval_acc


model_dense = model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=True)
for param in model_dense.parameters():
    param.requires_grad = False

num_ftrs = model_dense.classifier.in_features

classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512, True),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(512, 256, True),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(256, 64, True),
    nn.Dropout(0.4),
    nn.ReLU(),
    nn.Linear(64, 3))


model_dense.classifier = classifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.Adam(model_dense.classifier.parameters(), lr=0.01)

exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_conv, milestones=[8, 16, 24, 32], gamma=0.1)


model_conv, train_loss, eval_loss, train_acc, eval_acc = train_model(model_dense, criterion, optimizer_conv, exp_lr_scheduler, device)

torch.save(model_conv.state_dict(), 'Dense_state_dict3.pth')
torch.save(model_conv, 'Dense_gestures3.pth')

summary(model_dense, (3, 255, 255))
