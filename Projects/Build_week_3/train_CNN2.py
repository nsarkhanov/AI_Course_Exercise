import numpy as np
import torch
import torch.nn as nn
import torchsummary as summary

from cnn_model import CNN
import data_handler_CNN as dhc

import torch.optim as optim
from matplotlib import pyplot as plt
# from sklearn.metrics import accuracy_score
from time import time

model = CNN()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)  # 0.005
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
# scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
criterion = nn.NLLLoss()

n_batches = 66
n_epochs = 30
print_every = 40


def train(model, optimizer, criterion, epochs, lr_schedular):
    best_val = 10
    best_acc = 0
    train_loss = []
    val_loss = []
    accuracy = []
    acc = 0.2
    for i in range(epochs):

        running_loss = 0
        for j, (x, y) in enumerate(dhc.train_loader):

            if torch.cuda.is_available():
                x = x.cuda()

                y = y.cuda()

                model = model.cuda()

            optimizer.zero_grad()

            output = model.forward(x)

            loss = criterion(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if j % print_every == 0 and j != 0:

                correct_test = 0
                total_test = 0
                validation_loss = 0

                for k, (val_x, val_y) in enumerate(dhc.test_loader):
                    if k == print_every:
                        break
                    with torch.no_grad():

                        val_x = val_x.cuda()
                        val_y = val_y.cuda()

                        output = model(val_x)
                        validation_loss += criterion(output, val_y)

                    val_y = val_y.detach().cpu()
                    for idx, l in enumerate(output):
                        if torch.argmax(l) == val_y[idx]:
                            correct_test += 1
                        total_test += 1

                # print(correct_test/total_test)
                acc = correct_test / total_test
                train_loss.append(running_loss / print_every)
                val_loss.append(validation_loss.item() / n_batches)
                accuracy.append(acc)
                if best_val > validation_loss / n_batches:
                    best_val = validation_loss / n_batches
                    torch.save(model, 'saved_models/best_loss_model.pth')

                if best_acc < correct_test / total_test:
                    best_acc = correct_test / total_test
                    torch.save(model, 'saved_models/best_acc_model.pth')

                print(
                    f"\tIteration: {j}\t Loss: {running_loss / print_every:.4f} \t Val_Loss: {validation_loss / n_batches} \t Val_Acc: {correct_test / total_test}")
                running_loss = 0

        lr_schedular.step(round(acc, 2))
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.savefig("loss.png")
    # torch.save(model, 'model.pth')


start = time()
train(model, optimizer, criterion, n_epochs, scheduler)
end = time()

print(f"The server took: {end - start:.4f} ")
