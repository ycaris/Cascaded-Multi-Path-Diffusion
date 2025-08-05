import os
import h5py
import random
import numpy as np
import pdb
import torch
import torchvision.utils as utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from dataloader_scripts.random_pair import paired_data_augmentation

class MRI_Train(Dataset):
    def __init__(self, mode='T1toT2'):
        # self.root = '/home/bo/Projects/AccDiffusion/Data/fastMRI/Proc_Data/'
        self.root = '/data22/user/yz2337/2024/AccDiffusion/Data/fastMRI/Proc_Data/'

        if mode == 'T1toT2':
            self.data_dir = os.path.join(self.root, 'Train', 'T1toT2')
        elif mode == 'T1toFLAIR':
            self.data_dir = os.path.join(self.root, 'Train', 'T1toFLAIR')

        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

    def __getitem__(self, index):
        index = index % len(self.data_files)

        filename = self.data_files[index]

        with h5py.File(filename, 'r') as f:
            image_org = f['image_T1'][...]
            image_trg = f['image_T2'][...]

        input = image_org
        target = image_trg

        if True:
            input, target = paired_data_augmentation(input, target)

        input = torch.from_numpy(input)
        target = torch.from_numpy(target)

        return target.unsqueeze(0), {'low_res': input.unsqueeze(0)}

    def __len__(self):
        return len(self.data_files) * 10


class MRI_Test(Dataset):
    def __init__(self, mode='T1toT2'):
        # self.root = '/home/bo/Projects/AccDiffusion/Data/fastMRI/Proc_Data/'
        self.root = '/data22/user/yz2337/2024/AccDiffusion/Data/fastMRI/Proc_Data/'

        if mode == 'T1toT2':
            self.data_dir = os.path.join(self.root, 'Test', 'T1toT2')
        elif mode == 'T1toFLAIR':
            self.data_dir = os.path.join(self.root, 'Test', 'T1toFLAIR')

        if mode == 'T1toT2':
            self.data_dir = os.path.join(self.root, 'Test_w_PriorCNN', 'T1toT2')
        elif mode == 'T1toFLAIR':
            self.data_dir = os.path.join(self.root, 'Test_w_PriorCNN', 'T1toFLAIR')

        self.data_files = sorted(
            [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

    def __getitem__(self, index):
        filename = self.data_files[index]

        with h5py.File(filename, 'r') as f:
            image_org = f['image_T1'][...]
            image_trg = f['image_T2'][...]
            image_prior = f['image_prior'][...]

        input = image_org
        target = image_trg
        prior = image_prior

        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        prior = torch.from_numpy(prior)

        return target.unsqueeze(0), {'low_res': input.unsqueeze(0)}, prior.unsqueeze(0)

    def __len__(self):
        return len(self.data_files)

if __name__ == '__main__':
    pass