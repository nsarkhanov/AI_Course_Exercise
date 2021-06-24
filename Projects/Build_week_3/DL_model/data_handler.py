#do the necessary imports

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def data_processor(root_dir, batchSize, grayscale, rgb):

    if rgb:
        train_transforms = transforms.Compose([transforms.Resize((255, 255),interpolation=Image.NEAREST),
                                            #transforms.RandomRotation(30),
                                            #transforms.RandomResizedCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5],
                                                                    [0.5, 0.5, 0.5])])

        test_transforms = transforms.Compose([transforms.Resize((255, 255),interpolation=Image.NEAREST),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5],
                                                                    [0.5, 0.5, 0.5])])
    elif grayscale:
        train_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                            transforms.Resize((255, 255),interpolation=Image.NEAREST),
                                            #transforms.RandomRotation(30),
                                            #transforms.RandomResizedCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5], [0.5])])

        test_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                            transforms.Resize((255, 255),interpolation=Image.NEAREST),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5], [0.5])])

    # Pass transforms in here and create dataloaders
    #train_data = datasets.ImageFolder(root_dir + '\\train', transform=train_transforms)
    #test_data = datasets.ImageFolder(root_dir + '\\validation', transform=test_transforms)

    train_data = datasets.ImageFolder(root_dir + '/images/train', transform=train_transforms)
    test_data = datasets.ImageFolder(root_dir + '/images/validation', transform=test_transforms)

    train_loader = DataLoader(train_data, batch_size=batchSize, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batchSize, shuffle=False)

    return train_loader, test_loader

#print('No errors')