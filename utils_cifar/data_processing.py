import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import h5py
import torchvision.transforms as transforms
import pdb


class DatasetProcessingCIFAR_10(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()
        label_filepath = os.path.join(data_path, label_filename)
        fp_label = open(label_filepath, 'r')
        labels = [int(x.strip()) for x in fp_label]
        fp_label.close()
        self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([self.label[index]])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)


class DatasetProcessingNUS_WIDE(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = 'C:/Users/Ben/Desktop/NUS_WIDE/image'
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

class hdf5_NUS_WIDE(Dataset):
    def __init__(self, file_path, dimg, dlabel):
        self.file = file_path
        self.dimg = dimg
        self.dlabel = dlabel

    def __getitem__(self, index):
        with h5py.File(self.file, 'r',swmr=True) as db:
            # if self.dimg == 'train_img':
            #     img = db[self.dimg][index,...]
            #     if (0.5 > np.random.rand(1)):
            #         img = img[:,:,::-1].copy()
            #     img = torch.from_numpy(img)
            # else:
            #     img = torch.from_numpy(db[self.dimg][index,...])
            label = torch.from_numpy(db[self.dlabel][index,...])
            img = torch.from_numpy(db[self.dimg][index,...])
        return (img, label, index)

    def __len__(self):
        with h5py.File(self.file, 'r') as db:
            lens = len(db[self.dimg])
        return lens

class hdf5_NUS_WIDE_trial(Dataset):
    def __init__(self, file_path, dimg, dlabel,transform=None):
        self.file = file_path
        self.dimg = dimg
        self.dlabel = dlabel
        self.transform = transform

    def __getitem__(self, index):
        with h5py.File(self.file, 'r',swmr=True) as db:
            img = db[self.dimg][index,...]
            label = torch.from_numpy(db[self.dlabel][index,...])
        if self.transform is not None:
            img = self.transform(img)
        return (img, label, index)

    def __len__(self):
        with h5py.File(self.file, 'r') as db:
            lens = len(db[self.dimg])
        return lens

class DatasetProcessingMS_COCO(Dataset):
    def __init__(self, data_path, img_filename, label_filename, transform=None):
        self.img_path = data_path
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
