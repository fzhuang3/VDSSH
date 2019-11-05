import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import h5py
import torchvision.transforms as transforms
import pdb

class DatasetProcessingNUS_WIDE(Dataset):
    def __init__(self, img_path, data_path, img_filename, label_filename, transform=None):
        self.img_path =  img_path
        self.transform = transform
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)
