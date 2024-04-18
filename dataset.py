from einops import rearrange
from torch.utils.data import Dataset
import numpy as np
import os
from scipy import interpolate
import json
import csv
import torch
from pathlib import Path
import torchvision.transforms as transforms
from scipy.interpolate import interp1d
from typing import Callable, Optional, Tuple, Union
# from glob import glob
import os
from PIL import Image

def transform(x):
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(x)

class image_eeg_dataset(Dataset):
    def __init__(self, eeg_path, image_path, image_transform=transform, neg_per_eeg=20):
        super().__init__()
        loaded = torch.load(eeg_path)
        self.data = loaded['dataset']
        self.images = loaded['images']
        self.labels = loaded['labels']
        self.imagenet = image_path
        self.image_transform = image_transform
        self.data_len = 512
        # Compute size
        self.neg_per_img = neg_per_eeg
        self.size = len(self.data)*neg_per_eeg
        # print(self.size)

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        
        i = index//self.neg_per_img
        # print(i)
        eeg = self.data[i]['eeg'].float().t()
        eeg = eeg[20:460,:]
        eeg = np.array(eeg.transpose(0,1))

        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, self.data_len)
        f = interp1d(x, eeg)
        eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()
        eeg = eeg.unsqueeze(0)

        label = torch.tensor(self.data[i]["label"]).long()
        neg_label = label
        if label!=0 and label!=39:
            neg_label_low = np.random.randint(0,label)
            neg_label_high = np.random.randint(label+1,40)
            l = label if label<40-label else 40-label
            choice = np.random.binomial(1, p=l/(40-l))
            neg_label = neg_label_low if choice==0 else neg_label_high
        else:
            neg_label_ = np.random.randint(0,39)
            neg_label = neg_label_ if label==39 else 39-neg_label_
        # print(neg_label)
        # print(type(neg_label))
        image_dir = f'{self.imagenet}/{self.labels[neg_label]}'
        neg_img_dir = f'{image_dir}/{os.listdir(image_dir)[np.random.randint(0,len(os.listdir(image_dir)))]}'
        img = self.images[self.data[i]['image']]
        pos_img_dir = f'{self.imagenet}/{self.labels[label]}/{img}.JPEG'
        neg_img = Image.open(neg_img_dir).convert('RGB')
        pos_img = Image.open(pos_img_dir).convert('RGB')

        return {'eeg': eeg, 'pos_img': self.image_transform(pos_img), 'neg_img': self.image_transform(neg_img)}

# image_eeg_dataset = image_eeg_dataset('./DreamDiffusion/datasets/eeg_55_95_std.pth','./DreamDiffusion/datasets/imageNet_images')
# # img = Image.open('./DreamDiffusion/datasets/imageNet_images/n03452741/n03452741_17620.JPEG').convert('RGB')
# # print(img)
# for i in image_eeg_dataset:
#     print(i['eeg'].shape)
#     break