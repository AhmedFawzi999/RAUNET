## IMPORTS
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
# import torchvision.transforms as transforms
import PIL.Image as Image
from torchvision import transforms, datasets

## Transformer to transform the images into tensors and normalize
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0, 0, 0], [1, 1, 1])
])

## load the Dataset Class
class Load_Dataset(Dataset):
    def __init__(self, filenames):
        self.file_names = filenames

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        
        ## Loading the Image and Transforming it into a Tensor
        img_file_name = self.file_names[idx]
        ori_image = load_image(img_file_name)
        image = x_transforms(ori_image)

        ## Loading the GroundTruth Image and then converting it into a mask with each pixel representing a class
        mask = load_mask(img_file_name)
        mask=np.array(mask)
        mask[mask==255]=1
        # Transforming into a Tensor
        mask=torch.from_numpy(mask).long()
        mask = mask.squeeze()

        return image, mask

## Load the Image
def load_image(path):
    img_x = Image.open(path).convert('RGB')
    
    return img_x

## Load the GroundTruth Image
def load_mask(path):
    new_path=path.replace('Images', 'Labels')
    mask = Image.open(new_path)
    return mask
