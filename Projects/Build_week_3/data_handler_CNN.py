from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from canny import FourthDim

root_dir = './images'
batchSize = 128

train_transforms = transforms.Compose([
    transforms.Resize((255, 255), interpolation=Image.NEAREST),
    transforms.RandomRotation(30),
    FourthDim(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])])

test_transforms = transforms.Compose([
    transforms.Resize((255, 255), interpolation=Image.NEAREST),
    FourthDim(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])])

train_data = datasets.ImageFolder(root_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(root_dir + '/validation', transform=test_transforms)

train_loader = DataLoader(train_data, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batchSize, shuffle=False)

# examples = iter(train_loader)
#
# images, labels = examples.next()
#
# print(images.shape)
# print(labels.shape)




