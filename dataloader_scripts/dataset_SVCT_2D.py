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

class SVCT_Train(Dataset):
    def __init__(self):
        # self.root = '/home/bo/Projects/AccDiffusion/Data/SVCT/Processed_MAYO_SV_256_downrate6/'
        self.root = '/data22/user/yz2337/2024/AccDiffusion/Data/SVCT/Processed_MAYO_SV_256_downrate6/'

        self.data_dir = os.path.join(self.root, 'Train')
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

    def __getitem__(self, index):
        index = index % len(self.data_files)

        filename = self.data_files[index]

        with h5py.File(filename, 'r') as f:
            lv_CT = f['lv_CT'][...]
            fv_CT = f['fv_CT'][...]

        input = lv_CT
        target = fv_CT

        if True:
            input, target = paired_data_augmentation(input, target)

        input = torch.from_numpy(input)
        target = torch.from_numpy(target)

        return target.unsqueeze(0), {'low_res': input.unsqueeze(0)}

    def __len__(self):
        return len(self.data_files) * 3


class SVCT_Test(Dataset):
    def __init__(self):
        # self.root = '/home/bo/Projects/AccDiffusion/Data/SVCT/Processed_MAYO_SV_256_downrate6/'
        self.root = '/data22/user/yz2337/2024/AccDiffusion/Data/Processed_MAYO_SV_256_downrate6/'

        # self.data_dir = os.path.join(self.root, 'Test')
        self.data_dir = os.path.join(self.root, 'Test_w_PriorCNN')
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

    def __getitem__(self, index):
        filename = self.data_files[index]

        with h5py.File(filename, 'r') as f:
            lv_CT = f['lv_CT'][...]
            fv_CT = f['fv_CT'][...]
            fv_CT_prior = f['fv_CT_prior'][...]

        input = lv_CT
        target = fv_CT
        prior = fv_CT_prior

        input = torch.from_numpy(input)
        target = torch.from_numpy(target)
        prior = torch.from_numpy(prior)

        return target.unsqueeze(0), {'low_res': input.unsqueeze(0)}, prior.unsqueeze(0)

    def __len__(self):
        return len(self.data_files)


if __name__ == '__main__':
    pass