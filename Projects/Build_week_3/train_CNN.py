import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from torch.optim import Adam, SGD, lr_scheduler
from torch.autograd import Variable

import matplotlib.pyplot as plt

import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import data_handler_CNN as dhc
from cnn_model import CNN


def train(epochs, model, criterion, optimizer, lr_scheduler, device):
    train_losses = []
    val_losses = []
    acc_values = []
    acc_values_train = []
    lrs = []
    best_val = 2
    best_acc = 0.2
    min_train_val_loss = 20

    model = model.to(device)

    for epoch in range(epochs):

        tr_loss = 0
        correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(dhc.train_loader):
            images, labels = Variable(images), Variable(labels)

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output_train = model(images)
            pred_probs_train = F.softmax(output_train)

            pred_probs_train = pred_probs_train.to(device)

            loss_train = criterion(pred_probs_train, labels)

            for idx, pred_train in enumerate(pred_probs_train):
                if torch.argmax(pred_train) == labels[idx]:
                    correct_train += 1
                total_train += 1
            acc_train = round(correct_train / total_train, 3)
            acc_values_train.append(acc_train)
            if i % 200 == 0:
                print("Train accuracy: ", round(correct_train / total_train, 3), 'Epoch: ', epoch)

            loss_train.backward()

            optimizer.step()

            tr_loss += loss_train.item()
            if epoch % 1 == 0:
                train_losses.append(tr_loss / images.shape[0])

            val_loss = 0
            total_val = 0
            correct_val = 0
            with torch.no_grad():

                for j, (val_img, val_label) in enumerate(dhc.test_loader):

                    val_img, val_label = val_img.to(device), val_label.to(device)
                    output = model(val_img)
                    pred_probs_val = F.softmax(output)

                    pred_probs_val = pred_probs_val.to(device)

                    loss_val = criterion(pred_probs_val, val_label)
                    val_loss += loss_val.item()

                    for idx, pred in enumerate(pred_probs_val):
                        # print(torch.argmax(i), y[idx])
                        if torch.argmax(pred) == val_label[idx]:
                            correct_val += 1
                        total_val += 1
                    acc_val = round(correct_val / total_val, 3)
                    acc_values.append(acc_val)
            if i % 200 == 0:
                print("Accuracy Val: ", round(correct_val / total_val, 3), 'Epoch: ', epoch)
            if acc_val > best_acc:
                best_acc = acc_val
                torch.save(model, f"saved_models/best_acc_model3.pth")

            val_loss_perc = val_loss / val_img.shape[0]
            val_losses.append(val_loss_perc)
            if i % 20 == 0:
                print('Epoch : ', epoch, "\t Train loss: ", tr_loss / images.shape[0], "\t Validation loss: ", val_loss)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model, f"saved_models/best_loss_model3.pth")
        delta_train_val_loss = abs(tr_loss - val_loss_perc)
        if delta_train_val_loss < min_train_val_loss:
            torch.save(model, f"saved_models/min_delta_loss_model3.pth")
            min_train_val_loss = delta_train_val_loss

        lr_scheduler.step()

        lrs.append(optimizer.param_groups[0]["lr"])
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"lr:{current_lr}, epoch: {epoch}")

    torch.save(model, f"saved_models/test3.pth")
    print(lrs)
    #plt.plot(val_losses)
    #plt.plot(train_losses)
    #plt.plot(lrs)
    plt.plot(acc_values_train, 'ro')
    plt.plot(acc_values)
    plt.savefig('saved_models/metrics1.png')


lr = 1
epochs = 1
model = CNN()
criterion = nn.CrossEntropyLoss()
#optimizer = SGD(model.parameters(), lr=lr, momentum=0.5)
optimizer = Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train(epochs, model, criterion, optimizer, lr_scheduler, device)





