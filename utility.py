import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch

def loadData(data_dir, train = True, batch_size = 64, shuffle = True):
    if train:
        data_transforms = transforms.Compose([transforms.RandomRotation(30), \
            transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), \
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        data_transforms = transforms.Compose([transforms.Resize(224), \
            transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    image_dataset = datasets.ImageFolder(data_dir, transform = data_transforms)
    #image_datasets['test'] = datasets.ImageFolder(test_dir, transform = test_transforms)

	# TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size = batch_size, shuffle = shuffle)
    #dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64)
    return image_dataset, dataloader

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Get dimensions
    box = image.getbbox()
    width = np.abs(box[2] - box[0])
    height = np.abs(box[1] - box[3])
    #aspect_ratio = width / height
    
    min_size = 256
    # step 1: resize the image to have shortest side of 256 pixels, while keeping its aspect ratio
    if width >= height:
        new_height = min_size
        scale_factor = new_height/height
        new_width = int(width * scale_factor)
    else:
        new_width = min_size
        scale_factor = new_width/width
        new_height = int(height * scale_factor)
    image = image.resize((new_width, new_height))
    
    # step 2: crop the image from the center to uniform width and height of 224x224 pixels
    crop_width, crop_height = 224, 224
    left = (new_width - crop_width)/2
    top = (new_height - crop_height)/2
    right = (new_width + crop_width)/2
    bottom = (new_height + crop_height)/2
    image = image.crop((left, top, right, bottom))
    
    print('Original input image width: {}, height: {}'.format(width, height))
    print('Resized output image width: {}, height: {}, cropped at: {}'.format(new_width, new_height, (left, top, right, bottom)))
    
    # step 3: normalize from PIL image to numpy array, and reform the array
    # with the first column corresponding to the color channel,
    # the second and third columns to the width and height
    np_image = np.array(image)/255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2, 0, 1))
    return np_image


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax