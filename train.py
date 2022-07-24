#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2022/07/23 11:47:39
@Author  :   Baize
@Version :   1.0
@Contact :   
@License :   
@Desc    :   
'''
import warnings
warnings.filterwarnings('ignore')

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping,
    RichProgressBar
    )

from src.model import ConvModel
from src.datamodule import DevignDataModule

# hyper parameters
seed = 42
files = {
    'train': 'train.json',
    'valid': 'valid.json',
    'test': 'test.json'
}
emb_dim = 13
vocab_size = 159
pad_idx = 0
batch_size=128
n_epochs = 100

if __name__ == '__main__':
    if seed is not None:
        pl.seed_everything(seed, workers=True)
        
    model = ConvModel(vocab_size, emb_dim, pad_idx)
    dm = DevignDataModule(batch_size, files=files)
    
    # callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath='ckpt_points',
            filename='{epoch:02d}-{val_A:.4f}.pt',
            monitor='val_A',
            mode='max',
            save_weights_only=True
        ),
        EarlyStopping(
            monitor='val_A',
            patience=10,
            mode='max'
        ),
        RichProgressBar(leave=True)
    ]
    
    trainer = pl.Trainer(
        accelerator='gpu',
        gpus=1,
        max_epochs=n_epochs,
        callbacks=callbacks
    )
    
    trainer.fit(model, datamodule=dm)
    # Test
    trainer.test(model, datamodule=dm, ckpt_path='best')