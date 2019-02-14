from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import numpy as np
import config

def split_datapack(datapack):
    return datapack['image'].cuda(), datapack['label'].cuda(), datapack['camera_id'].cuda()

class AdaptReID_Dataset(Dataset):
    def __init__(self, dataset_name, mode='source', transform=None):
        self.dataset_name = dataset_name
        self.mode = mode
        self.csv_path = config.get_csv_path(self.dataset_name)
        self.csv = self.get_csv()
        self.image_names = self.csv['image_path']
        self.image_labels = self.csv['id']
        self.transform = transform

        if self.mode == 'test' or self.mode == 'query':
            self.image_camera_ids = self.csv['camera'].values.astype('int')
        
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        input_image = self.get_image(idx)
        label = self.get_label(idx)
        
        if self.transform is not None:
            input_image = self.transform(input_image)
        
        datapack = {'image': input_image} 
        datapack['label'] = label
        datapack['camera_id'] = -1
        if self.mode == 'test' or self.mode == 'query':
            datapack['camera_id'] = self.image_camera_ids[idx]
        return datapack

    def get_csv(self):
        csv_name = '{}_list.csv'.format(self.mode)
        csv_f = os.path.join(self.csv_path, csv_name)
        return pd.read_csv(csv_f)

    def get_image(self, idx):
        dataset_path = config.get_dataset_path(self.dataset_name)
        image_path = os.path.join(dataset_path, self.image_names[idx])
        image = Image.open(image_path).convert('RGB')
        return image

    def get_label(self, idx):
        label = self.image_labels[idx]
        label = Variable(torch.tensor(label, dtype=torch.int64))
        return label