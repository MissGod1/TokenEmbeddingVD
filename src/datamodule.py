#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   datamodule.py
@Time    :   2022/07/23 11:55:46
@Author  :   Baize
@Version :   1.0
@Contact :   
@License :   
@Desc    :   
'''

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional
import os
import pandas as pd
import gc
from torchtext.vocab import vocab

pad = '<pad>'
unk = '<unk>'

class DevignDataset(Dataset):
    def __init__(self, root, filename, max_len=512, cache='.cache') -> None:
        super(DevignDataset, self).__init__()
        self.file = os.path.join(root, filename)
        self.cache_file = os.path.join(cache, filename)
        self.vb_file = os.path.join(root, 'vocab')
        self.max_len = max_len
        if not os.path.exists(self.cache_file):
            self.data = pd.read_json(self.file)
            self.data = self.data[['tokenized', 'target']]
            os.makedirs(cache, exist_ok=True)
            self.data.to_parquet(self.cache_file, engine="fastparquet")
            gc.collect()
        else:
            self.data = pd.read_parquet(self.cache_file, engine="fastparquet")
        
        if not os.path.exists(self.vb_file):
            raise FileNotFoundError(f"{self.vb_file} is not exists.")
        else:
            with open(self.vb_file, 'rb') as fp:
                self.vb = pickle.load(fp)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        x, y = self.data.iloc[index]
        x = x[:self.max_len]
        x = self.vb(x)
        if len(x) < self.max_len:
            x = x + [self.vb[pad]]*(self.max_len - len(x))
        
        return torch.LongTensor(x), torch.tensor(y)

class DevignDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, root="data", files=None):
        super(DevignDataModule, self).__init__()
        self.batch_size = batch_size
        self.root = root
        self.files = files
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(DevignDataset(self.root, self.files['train']),
                          batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(DevignDataset(self.root, self.files['valid']),
                          batch_size=self.batch_size)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(DevignDataset(self.root, self.files['test']),
                          batch_size=self.batch_size)

if __name__ == '__main__':
    pass