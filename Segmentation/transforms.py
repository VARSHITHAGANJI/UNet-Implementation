import torch
torch.manual_seed(17)
import torchvision.transforms as transforms

# Cropping, padding, resizing

def transform_normal():
    return transforms.Compose([transforms.Pad([5,5],fill=0,padding_mode='constant')])

def augmentations():
    return transforms.Compose([transforms.RandomVerticalFlip(p = 0.5),
                                transforms.RandomHorizontalFlip(p=0.3),
                                transforms.RandomRotation(degrees = (-50,50)),transforms.ToTensor(),
                                transforms.Normalize(mean = 0,std = 1, inplace=False)
                            ])
