"""
# Filename: criterion.py                                                       #
# Module: utils                                                                #
# Created Date: Tuesday, June 25th 2024, 1:56:28 pm                            #
# Author: Nisarg Joshi                                                         #
# -----                                                                        #
# Last Modified: Tue Jun 25 2024                                               #
# Modified By: Nisarg Joshi                                                    #
# -----                                                                        #
# Copyright (c) 2024 Nisarg Joshi @ HL Klemove India Pvt. Ltd.                 #
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):

    def __init__(self):
        super().__init__()

    def _ohem_loss(self, pred, target):
        n_min = target[target != self.ignore_index].numel() // 16
        loss = F.cross_entropy(
            pred,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
        ).view(-1)

        loss_hard = loss[loss >= self.ohem_threh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = torch.topk(loss_hard, n_min)

        return loss_hard.mean()

    def forward(
        self,
        classification: torch.Tensor,
        localization: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        pixel_cls_loss = self._ohem_loss(classification, targets["mask"])

        cls_loss = self._ohem_loss()

        return 0
