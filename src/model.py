#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/07/23 11:03:38
@Author  :   Baize
@Version :   1.0
@Contact :   
@License :   
@Desc    :   
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch import Tensor
import pytorch_lightning as pl
from typing import Optional, Union, Dict, Sequence, Tuple, List
import torchmetrics

class ConvModel(pl.LightningModule):
    def __init__(self, vocab_size, emb_dim, pad_idx):
        super(ConvModel, self).__init__()
        self.num_out_kernel = 512
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim,
                                      padding_idx=pad_idx)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=self.num_out_kernel,
                      kernel_size=(9, emb_dim)), nn.ReLU(True))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.num_out_kernel, out_features=64),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(True),
            nn.BatchNorm1d(num_features=16),
            nn.Linear(in_features=16, out_features=2),
        )
        self.drop = nn.Dropout(p=0.5)
        self.init_metrics()
    
    def init_metrics(self):
        self.train_acc = torchmetrics.Accuracy(num_classes=2)
        self.val_acc = torchmetrics.Accuracy(num_classes=2)
        self.val_pre = torchmetrics.Precision(num_classes=2, ignore_index=0)
        self.val_re = torchmetrics.Recall(num_classes=2, ignore_index=0)
        self.val_f1 = torchmetrics.F1Score(num_classes=2, ignore_index=0)

    def forward(self, sentences, masks=None):
        emb = self.embedding(sentences) # [N, 500, 13]
        emb = emb.unsqueeze(dim=1) # [N, 1, 500, 13]
        cs = self.features(emb) # [N, 512, 503, 1]
        cs = cs.view(sentences.shape[0], self.num_out_kernel, -1) # [N, 512, 503]
        rs = self.drop(nn.functional.max_pool1d(cs, kernel_size=cs.shape[-1])) # [N, 512, 1]
        rs = rs.view(sentences.shape[0], self.num_out_kernel) # [N, 512]
        soft = self.classifier(rs) # [N, 2]
        return soft, rs

    # configure
    def configure_optimizers(
        self
    ) -> Optional[Union[Optimizer, Sequence[Optimizer], Dict, Sequence[Dict],
                        Tuple[List, List]]]:
        optimizer = Adam(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(
            self, batch, batch_idx
    ) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        inputs, targets = batch
        logits, _ = self(inputs)
        loss = F.cross_entropy(logits, targets)
        self.train_acc.update(logits, targets)
        self.log('loss', loss)
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits, _ = self(inputs)
        # loss = F.cross_entropy(logits, targets)
        # self.log('val_loss', loss)
        self.val_acc.update(logits, targets)
        self.val_pre.update(logits, targets)
        self.val_re.update(logits, targets)
        self.val_f1.update(logits, targets)
        # return loss

    def test_step(self, batch, batch_idx) -> Dict[str, Tensor]:
        inputs, targets = batch
        logits, _ = self(inputs)
        # loss = F.cross_entropy(logits, targets)
        # self.log('test_loss', loss)
        self.val_acc.update(logits, targets)
        self.val_pre.update(logits, targets)
        self.val_re.update(logits, targets)
        self.val_f1.update(logits, targets)
        # return loss

    def training_epoch_end(self, outputs):
        # return super().training_epoch_end(outputs)
        self.train_acc.reset()

    def validation_epoch_end(self, outputs):
        # return super().validation_epoch_end(outputs)
        self.log(f'val_A', self.val_acc.compute(), prog_bar=True, on_epoch=True)
        self.log(f'val_P', self.val_pre.compute(), prog_bar=True, on_epoch=True)
        self.log(f'val_R', self.val_re.compute(), prog_bar=True, on_epoch=True)
        self.log(f'val_F1', self.val_f1.compute(), prog_bar=True, on_epoch=True)
        self.val_acc.reset()
        self.val_pre.reset()
        self.val_re.reset()
        self.val_f1.reset()

    def test_epoch_end(self, outputs):
        # return super().test_epoch_end(outputs)
        self.log(f'A', self.val_acc.compute(), prog_bar=True, on_epoch=True)
        self.log(f'P', self.val_pre.compute(), prog_bar=True, on_epoch=True)
        self.log(f'R', self.val_re.compute(), prog_bar=True, on_epoch=True)
        self.log(f'F1', self.val_f1.compute(), prog_bar=True, on_epoch=True)
        self.val_acc.reset()
        self.val_pre.reset()
        self.val_re.reset()
        self.val_f1.reset()