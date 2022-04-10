import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import glob


class HoleCenterDataset(Dataset):
    def __init__(self, root_dir='./data', split='train', img_wh=(2448, 2048), resize_fac=2):
        # TODO scale intrinsic according to resize_fac
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.resize_fac = resize_fac
        self.define_transforms()
        self.read_meta()

    def read_meta(self):
        jsonpaths = glob.glob(self.root_dir + '/*.json')
        jsonpaths.sort()
        self.img_name = [jsonpaths[i].split('data/')[1].split('.')[0] for i in range(len(jsonpaths))]

        self.labels = {}
        for json_idx, jsonpath in enumerate(jsonpaths):
            self.labels[json_idx] = []
            with open(jsonpath, 'r') as f:
                label_json = json.load(f)
                for idx_label in range(len(label_json['shapes'])):
                    self.labels[json_idx].append(label_json['shapes'][idx_label]['points'][0])  # wh
            self.labels[json_idx] = np.array(self.labels[json_idx]) / self.resize_fac / 4  # 再除以4是因为feature map是原本尺寸的1/4
            # 对应下面的crop
            self.labels[json_idx][:, 0] = self.labels[json_idx][:, 0] - (self.img_wh[0]-self.img_wh[1])//2

        np.random.seed(1)
        self.train_val_idx = np.random.permutation(len(jsonpaths))
        self.train_idx = self.train_val_idx[:int(len(jsonpaths)*0.6)]
        self.val_idx = self.train_val_idx[int(len(jsonpaths)*0.6):]

    def define_transforms(self):
        self.transform = T.Compose([
            T.ToTensor(),
            T.CenterCrop(self.img_wh[1]),
            T.Resize(self.img_wh[1]//self.resize_fac),
        ])

    def __len__(self):
        if self.split == 'train':
            return len(self.train_idx)
        if self.split == 'val':
            return len(self.val_idx)

    def __getitem__(self, idx):
        if self.split == 'train':
            idx = self.train_idx[idx]
        elif self.split == 'val':
            idx = self.val_idx[idx]
        img_name = self.img_name[idx]
        img = Image.open(os.path.join(self.root_dir, img_name + '.jpg'))
        img = self.transform(img)  #  TODO check 3 H W
        label = torch.from_numpy(self.labels[idx])
        
        sample = {'img': img,
                'label': label,
                }

        return sample