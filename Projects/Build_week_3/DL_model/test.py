'''
This file is only for testing things

*********** --------- ***********

The pre_processor function takes 4 aruguments and returns 2 (train_loader and test_loader)

- root_dir = root directory where test and validation folders are present
- batchSize = integer value for the batchsize

(if you want to train the model on graycale images set it True and vice-versa for rgb)
- grayscale = boolean 
- rgb = boolean

*********** --------- ***********

    '''

import torchvision
import matplotlib.pyplot as plt
import numpy as np
import data_handler as dh


#give the path to the root directory here
root_dir = 'C:\\Users\\User\\Desktop\\Strive_School\\Github\\Buildweek_03_Main_repo\\images'       #use \\ if you are on windows
batchSize = 32
grayscale = True
rgb = False

train_loader, test_loader = dh.data_processor(root_dir, batchSize, grayscale, rgb)
                               
# print(train_loader)
# print(test_loader)

#give the class names here
classes = ('dog', 'fish', 'rabbit')


#helper function to plot
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(images.shape)
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batchSize))) 
