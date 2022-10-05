import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL.Image as Image

#### ORiginal ###
# x_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize([0, 0, 0], [1, 1, 1])
# ])
# y_trans=transforms.ToTensor()

# class Load_Dataset(Dataset):
#     def __init__(self, filenames):
#         self.file_names = filenames

#     def __len__(self):
#         return len(self.file_names)

#     def __getitem__(self, idx):
#         down_sample = 2
#         img_file_name = self.file_names[idx]
#         ori_image = load_image(img_file_name)
#         image = x_transforms(ori_image)
#         # image = F.max_pool2d(image, kernel_size=(down_sample, down_sample), stride=down_sample, padding=0)
#         # image = F.pad(image, (0, 0, 2, 2), 'constant', 0)

#         mask = load_mask(img_file_name)
#         # mask = mask[np.newaxis, :, :]
#         labels = torch.from_numpy(mask).float()
#         # labels = F.max_pool2d(labels, kernel_size=(down_sample, down_sample), stride=down_sample, padding=0)
#         # labels = F.pad(labels, (0, 0, 2, 2), 'constant', 0)
#         # labels = labels.squeeze()
#         # print(labels.shape)
#         # print(np.unique(labels),"hiiiiiiiiiiiiiiiiiii",image.shape,labels.shape)

#         return image, labels


# def load_image(path):
#     img_x = Image.open(path).convert("RGB")
#     # print(path)

#     return img_x



# def load_mask(path):
#     new_path=path.replace('Images', 'Labels')
#     # print(new_path)
#     mask = np.array(Image.open(new_path).convert("L"), dtype=np.float32)
#         #     image = np.array(Image.open(img_path).convert("RGB"))
#         # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
#     mask[mask == 255.0] = 1.0

#     return mask.astype(np.uint8)

##### lel RVSC
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import PIL.Image as Image

x_transforms = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])
y_trans = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])
class Load_Dataset(Dataset):
    def __init__(self, filenames):
        self.file_names = filenames

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        down_sample = 2
        img_file_name = self.file_names[idx]
        ori_image = load_image(img_file_name)
        image = x_transforms(ori_image)
        # print(np.shape(image))
        # image = F.max_pool2d(image, kernel_size=(down_sample, down_sample), stride=down_sample, padding=0)
        # image = F.pad(image, (0, 0, 2, 2), 'constant', 0)

        mask = load_mask(img_file_name)
        # mask = mask[np.newaxis, :, :]

        # print(np.shape(mask))

        mask=y_trans(mask)
        # labels = torch.from_numpy(mask).float()
        labels=mask
        # print(np.shape(labels))
        # labels = F.max_pool2d(labels, kernel_size=(down_sample, down_sample), stride=down_sample, padding=0)
        # labels = F.pad(labels, (0, 0, 2, 2), 'constant', 0)
        labels = labels.squeeze()
        
        # print(np.shape(labels))
        # print(labels.shape)
        # print(np.unique(labels),"hiiiiiiiiiiiiiiiiiii",image.shape,labels.shape)

        return image, labels


def load_image(path):
    img_x = Image.open(path).convert("RGB")
    # print(path)

    return img_x



def load_mask(path):
    new_path=path.replace('Images', 'Labels')
    # print(new_path)
    mask = Image.open(new_path).convert("L")
        #     image = np.array(Image.open(img_path).convert("RGB"))
        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

    return mask




# import torch
# import numpy as np
# import cv2
# from torch.utils.data import Dataset
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import PIL.Image as Image
# LABEL_TO_COLOR = {0:[255,0,0], 1:[0,255,0], 2:[0,0,255]}

# x_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0, 0, 0], [1, 1, 1])
# ])
# y_trans=transforms.ToTensor()

# class Load_Dataset(Dataset):
#     def __init__(self, filenames):
#         self.file_names = filenames

#     def __len__(self):
#         return len(self.file_names)

#     def __getitem__(self, idx):
#         down_sample = 2
#         img_file_name = self.file_names[idx]
#         ori_image = load_image(img_file_name)
#         image = x_transforms(ori_image)
#         # image = F.max_pool2d(image, kernel_size=(down_sample, down_sample), stride=down_sample, padding=0)
#         # image = F.pad(image, (0, 0, 2, 2), 'constant', 0)

#         mask = load_mask(img_file_name)
#         # mask = rgb2mask(np.array(mask))
#         mask=torch.from_numpy(mask).long()
#         # labels = F.max_pool2d(labels, kernel_size=(down_sample, down_sample), stride=down_sample, padding=0)
#         # labels = F.pad(labels, (0, 0, 2, 2), 'constant', 0)
#         mask = mask.squeeze()
#         # print(np.unique(labels),"hiiiiiiiiiiiiiiiiiii",image.shape,labels.shape)

#         return image, mask


# def load_image(path):
#     img_x = Image.open(path)
#     # print(path)

#     return img_x

# def rgb2mask(rgb):
    
#     mask = np.zeros((rgb.shape[0], rgb.shape[1]))

#     for k,v in LABEL_TO_COLOR.items():
#         mask[np.all(rgb==v, axis=2)] = k
        
#     return mask

# def load_mask(path):
#     new_path=path.replace('Images', 'Labels')
#     mask = Image.open(new_path)
    
#     return mask


# import torch
# import numpy as np
# import cv2
# from torch.utils.data import Dataset
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import PIL.Image as Image
# LABEL_TO_COLOR = {0:[255,0,0], 1:[0,255,0], 2:[0,0,255]}

# x_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0, 0, 0], [1, 1, 1])
# ])
# y_trans=transforms.ToTensor()

# class Load_Dataset(Dataset):
#     def __init__(self, filenames):
#         self.file_names = filenames

#     def __len__(self):
#         return len(self.file_names)

#     def __getitem__(self, idx):
#         down_sample = 2
#         img_file_name = self.file_names[idx]
#         ori_image = load_image(img_file_name)
#         image = x_transforms(ori_image)
#         # image = F.max_pool2d(image, kernel_size=(down_sample, down_sample), stride=down_sample, padding=0)
#         # image = F.pad(image, (0, 0, 2, 2), 'constant', 0)

#         mask = load_mask(img_file_name)
#         # mask = rgb2mask(np.array(mask))
#         mask=torch.from_numpy(mask).long()
#         # labels = F.max_pool2d(labels, kernel_size=(down_sample, down_sample), stride=down_sample, padding=0)
#         # labels = F.pad(labels, (0, 0, 2, 2), 'constant', 0)
#         mask = mask.squeeze()
#         # print(np.unique(labels),"hiiiiiiiiiiiiiiiiiii",image.shape,labels.shape)

#         return image, mask


# def load_image(path):
#     img_x = Image.open(path)
#     # print(path)

#     return img_x

# def rgb2mask(rgb):
    
#     mask = np.zeros((rgb.shape[0], rgb.shape[1]))

#     for k,v in LABEL_TO_COLOR.items():
#         mask[np.all(rgb==v, axis=2)] = k
        
#     return mask

# def load_mask(path):
#     new_path=path.replace('Images', 'Labels')
#     mask = Image.open(new_path)
    
#     return mask
