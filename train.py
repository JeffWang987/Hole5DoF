from cv2 import split
import torch
from torch.nn import functional as F
import numpy as np
import math
import cv2
import os

from opt import get_opts

# datasets
from torch.utils.data import DataLoader
from torchvision import transforms as T
from dataset import HoleCenterDataset

# models
from models import CenterNet, CenterNetGT, centernetloss, CenterNetDecoder, geometric_filter

# optimizer
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

seed_everything(1234, workers=True)


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Hole5DoFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.net = CenterNet()
        self.count = 0
        
    def forward(self, x):
        return self.net(x)

    def setup(self, stage=None):
        """
        setup dataset for each machine
        """
        self.train_dataset = HoleCenterDataset(
            self.hparams.root_dir,
            split='train',
            resize_fac=self.hparams.resize_fac
        )
        self.val_dataset = HoleCenterDataset(
            self.hparams.root_dir,
            split='val',
            resize_fac=self.hparams.resize_fac
        )
        self.test_dataset = HoleCenterDataset(
            self.hparams.root_dir,
            split='test',
            resize_fac=self.hparams.resize_fac
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                        shuffle=False,
                        num_workers=self.hparams.num_workers,
                        batch_size=self.hparams.batch_size,
                        pin_memory=True)

    def configure_optimizers(self):
        self.optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        
        scheduler = CosineAnnealingLR(self.optimizer,
                                      T_max=self.hparams.num_epochs,
                                      eta_min=self.hparams.lr/1e2)

        return [self.optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch['img'], batch['label']
        pred_dict = self(images)
        gt_dict = CenterNetGT.generate(labels, self.hparams.resize_fac)
        loss = centernetloss(pred_dict, gt_dict)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/center_loss', loss['loss_cls'])
        self.log('train/reg_loss', loss['loss_center_reg'])

        total_loss = loss['loss_cls'] + loss['loss_center_reg']

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['img'], batch['label']
        pred_dict = self(images)
        gt_dict = CenterNetGT.generate(labels, self.hparams.resize_fac)
        loss = centernetloss(pred_dict, gt_dict)

        log = {'center_loss': loss['loss_cls'],
               'reg_loss': loss['loss_center_reg']}

        return log

    def validation_epoch_end(self, outputs):
        mean_center_loss = torch.stack([x['center_loss'] for x in outputs]).mean()
        mean_reg_loss = torch.stack([x['reg_loss'] for x in outputs]).mean()

        self.log('val/center_loss', mean_center_loss, prog_bar=True)
        self.log('val/reg_loss', mean_reg_loss, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        img1, img2, labels1, labels2 = batch['img1'], batch['img2'], batch['label1'], batch['label2']
        pred_dict1 = self(img1)
        pred_dict2 = self(img2)
        fmap1 = pred_dict1["cls"]
        fmap2 = pred_dict2["cls"]
        reg1 = pred_dict1["reg"]
        reg2 = pred_dict2["reg"]
        boxes1, scores1, classes1 = CenterNetDecoder.decode(fmap1, reg1, K=10)
        boxes2, scores2, classes2 = CenterNetDecoder.decode(fmap2, reg2, K=10)

        img1 = (img1[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # H W C
        img2 = (img2[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # H W C
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        labels1 = labels1[0].cpu().numpy() * 4  # 此处乘以4并不是因为缩放因子是4，而是因为centernet的heatmap把输入图像resize了1/4
        labels2 = labels2[0].cpu().numpy() * 4
        pred_labels1 = boxes1[0].cpu().numpy() * 4
        pred_labels2 = boxes2[0].cpu().numpy() * 4
        # for label1, lable2 in zip(labels1, labels2):
            # cv2.circle(img1, (int(label1[0]), int(label1[1])), radius=5, color=(255, 255, 255), thickness=-1)
            # cv2.circle(img2, (int(label2[0]), int(label2[1])), radius=5, color=(255, 255, 255), thickness=-1)
        # for pred_label1, pred_label2 in zip(pred_labels1, pred_labels2):
        #     cv2.circle(img1, (int(pred_label1[0]), int(pred_label1[1])), radius=5, color=(0, 255, 255), thickness=-1)
        #     cv2.circle(img2, (int(pred_label2[0]), int(pred_label2[1])), radius=5, color=(0, 255, 255), thickness=-1)
        # cv2.imwrite('./logs/' + self.hparams.exp_name + '/{}_label1.jpg'.format(self.count), img1)
        # cv2.imwrite('./logs/' + self.hparams.exp_name + '/{}_label2.jpg'.format(self.count), img2)
        geometric_filter(pred_labels2, pred_labels1, resize_fac=self.hparams.resize_fac, thres=3, image_left=img2, image_right=img1, show=True, id=self.count, log_dir='./logs/'+self.hparams.exp_name)
        self.count += 1



if __name__ == '__main__':
    hparams = get_opts()
    os.system('mkdir -p ./logs/{}/geo_filter'.format(hparams.exp_name))
    mnistsystem = Hole5DoFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              monitor='val/center_loss',
                              mode='min',
                              save_top_k=10)
    callbacks = [ckpt_cb]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      gpus=1)

    # trainer.fit(mnistsystem)
    # print(ckpt_cb.best_model_path)
    # testsystem = mnistsystem.load_from_checkpoint(ckpt_cb.best_model_path)

    os.system('rm ./logs/{}/geo_filter/*'.format(hparams.exp_name))
    testsystem = mnistsystem.load_from_checkpoint('/mnt/cfs/algorithm/xiaofeng.wang/jeff/code/MVS/BMI/Hole5DoF/ckpts/0707-trainvaltest-all/epoch=8.ckpt')



    trainer.test(testsystem)