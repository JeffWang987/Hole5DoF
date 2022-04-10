from cv2 import split
import torch
from torch.nn import functional as F

from opt import get_opts

# datasets
from torch.utils.data import DataLoader
from torchvision import transforms as T
from dataset import HoleCenterDataset

# models
from models import CenterNet, CenterNetGT, centernetloss

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

    def configure_optimizers(self):
        self.optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        
        scheduler = CosineAnnealingLR(self.optimizer,
                                      T_max=self.hparams.num_epochs,
                                      eta_min=self.hparams.lr/1e2)

        return [self.optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch['img'], batch['label']
        pred_dict = self(images)
        gt_dict = CenterNetGT.generate(labels)
        loss = centernetloss(pred_dict, gt_dict)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/center_loss', loss['loss_cls'])
        self.log('train/reg_loss', loss['loss_center_reg'])

        total_loss = loss['loss_cls'] + loss['loss_center_reg']

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['img'], batch['label']
        pred_dict = self(images)
        gt_dict = CenterNetGT.generate(labels)
        loss = centernetloss(pred_dict, gt_dict)

        log = {'center_loss': loss['loss_cls'],
               'reg_loss': loss['loss_center_reg']}

        return log

    def validation_epoch_end(self, outputs):
        mean_center_loss = torch.stack([x['center_loss'] for x in outputs]).mean()
        mean_reg_loss = torch.stack([x['reg_loss'] for x in outputs]).mean()

        self.log('val/center_loss', mean_center_loss, prog_bar=True)
        self.log('val/reg_loss', mean_reg_loss, prog_bar=True)


if __name__ == '__main__':
    hparams = get_opts()
    mnistsystem = Hole5DoFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              monitor='val/center_loss',
                              mode='max',
                              save_top_k=5)
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

    trainer.fit(mnistsystem)