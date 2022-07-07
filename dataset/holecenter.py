from cv2 import split
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
        if self.split == 'test':
            # test 需要同时加载双目图像
            jsonpaths1 = glob.glob(self.root_dir + '/test/cam1*.json')
            jsonpaths2 = glob.glob(self.root_dir + '/test/cam2*.json')
            jsonpaths1.sort()
            jsonpaths2.sort()
            self.img1_name = [jsonpaths1[i].split(self.root_dir)[1][1:].split('.')[0] for i in range(len(jsonpaths1))]
            self.img2_name = [jsonpaths2[i].split(self.root_dir)[1][1:].split('.')[0] for i in range(len(jsonpaths2))]
            self.labels1 = {}
            self.labels2 = {}
            for json_idx, (jsonpath1, jsonpath2) in enumerate(zip(jsonpaths1, jsonpaths2)):
                self.labels1[json_idx] = []
                self.labels2[json_idx] = []
                with open(jsonpath1, 'r') as f:
                    label_json = json.load(f)
                    for idx_label in range(len(label_json['shapes'])):
                        self.labels1[json_idx].append(label_json['shapes'][idx_label]['points'][0])  # wh
                with open(jsonpath2, 'r') as f:
                    label_json = json.load(f)
                    for idx_label in range(len(label_json['shapes'])):
                        self.labels2[json_idx].append(label_json['shapes'][idx_label]['points'][0])  # wh
                self.labels1[json_idx] = np.array(self.labels1[json_idx])
                self.labels2[json_idx] = np.array(self.labels2[json_idx])
                # 对应下面的crop
                self.labels1[json_idx][:, 0] = self.labels1[json_idx][:, 0] - (self.img_wh[0]-self.img_wh[1])//2
                self.labels2[json_idx][:, 0] = self.labels2[json_idx][:, 0] - (self.img_wh[0]-self.img_wh[1])//2
                self.labels1[json_idx] = self.labels1[json_idx] / self.resize_fac / 4  # 再除以4是因为feature map是原本尺寸的1/4
                self.labels2[json_idx] = self.labels2[json_idx] / self.resize_fac / 4  # 再除以4是因为feature map是原本尺寸的1/4
        else:
            # train val时只需要加载单张图像
            if self.split == 'train':
                jsonpaths = glob.glob(self.root_dir + '/trainall/*.json')
            elif self.split == 'val':
                jsonpaths = glob.glob(self.root_dir + '/val/*.json')
            jsonpaths.sort()
            self.img_name = [jsonpaths[i].split(self.root_dir)[1][1:].split('.')[0] for i in range(len(jsonpaths))]

            self.labels = {}
            for json_idx, jsonpath in enumerate(jsonpaths):
                self.labels[json_idx] = []
                with open(jsonpath, 'r') as f:
                    label_json = json.load(f)
                    for idx_label in range(len(label_json['shapes'])):
                        self.labels[json_idx].append(label_json['shapes'][idx_label]['points'][0])  # wh
                self.labels[json_idx] = np.array(self.labels[json_idx])
                # 对应下面的crop
                self.labels[json_idx][:, 0] = self.labels[json_idx][:, 0] - (self.img_wh[0]-self.img_wh[1])//2
                self.labels[json_idx] = self.labels[json_idx] / self.resize_fac / 4  # 再除以4是因为feature map是原本尺寸的1/4

        np.random.seed(1)
        if self.split != 'test':
            # self.train_val_idx = np.random.permutation(len(jsonpaths))
            # self.train_idx = self.train_val_idx[:int(len(jsonpaths)*0.9)]
            # self.val_idx = self.train_val_idx[int(len(jsonpaths)*0.9):]
            self.train_idx = np.random.permutation(len(jsonpaths))
            self.val_idx = np.random.permutation(len(jsonpaths))

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
        if self.split == 'test':
            return len(self.img1_name)


    def __getitem__(self, idx):
        if self.split == 'test':
            img_name1 = self.img1_name[idx]
            img_name2 = self.img2_name[idx]
            img1 = Image.open(os.path.join(self.root_dir, img_name1 + '.jpg'))
            img2 = Image.open(os.path.join(self.root_dir, img_name2 + '.jpg'))
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            label1 = torch.from_numpy(self.labels1[idx])
            label2 = torch.from_numpy(self.labels2[idx])
            
            sample = {'img1': img1,
                      'img2': img2,
                      'label1': label1,
                      'label2': label2,
                    }
        else:
            if self.split == 'train':
                idx = self.train_idx[idx]
            elif self.split == 'val':
                idx = self.val_idx[idx]
            img_name = self.img_name[idx]
            img = Image.open(os.path.join(self.root_dir, img_name + '.jpg'))
            img = self.transform(img)
            label = torch.from_numpy(self.labels[idx])
            
            sample = {'img': img,
                    'label': label,
                    }

        return sample